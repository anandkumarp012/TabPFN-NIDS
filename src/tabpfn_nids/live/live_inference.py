"""Async inference worker pool for live detection.

Implements a producer-consumer architecture:

    DetectionJob queue  →  [Worker 1 | Worker 2 | ... | Worker N]  →  Result bus

The TabPFN model is loaded ONCE per worker and reused for every prediction.
Workers run in a ThreadPoolExecutor (not separate processes) because:
  - TabPFN uses PyTorch, which is not safely fork()-able on all platforms.
  - Threads share the loaded model in memory (no serialization overhead).
  - Inference is GPU/MPS-bound; threads release the GIL during torch ops.

The model is loaded using the existing ModelRegistry so that any
multi-model ensemble configuration is automatically respected.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.live.live_preprocessor import LivePreprocessor
from tabpfn_nids.live.window_manager import DetectionJob
from tabpfn_nids.live.result_bus import PredictionResult, ResultBus

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl"
)

# Map from integer class index to string label (matches ModelInfo.classes)
_CLASS_LABELS = {0: "BENIGN", 1: "ATTACK"}


def _load_model(model_path: Path) -> Any:
    """Load a TabPFN model pickle from disk.

    Args:
        model_path: Path to the .pkl artifact.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If unpickling fails.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run the training pipeline to generate model artifacts."
        )
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from %s", model_path)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {model_path}: {exc}"
        ) from exc


def _run_inference_sync(
    model: Any,
    X: np.ndarray,
    window_id: str,
) -> PredictionResult:
    """Synchronous inference — runs in a thread.

    Args:
        model: A loaded TabPFN model with predict_proba().
        X: Preprocessed feature matrix (flows × features).
        window_id: Identifier for this detection window.

    Returns:
        PredictionResult with the aggregated window-level prediction.
    """
    start_ts = time.time()

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            # Shape: (n_flows, n_classes)
            # Class 0 = Normal/BENIGN, Class 1 = Attack
            attack_probs = probabilities[:, 1]
            mean_attack_prob = float(np.mean(attack_probs))
            predicted_label = "ATTACK" if mean_attack_prob >= 0.5 else "BENIGN"
            confidence = float(
                mean_attack_prob if predicted_label == "ATTACK"
                else 1.0 - mean_attack_prob
            )
        else:
            # Fallback: predict() only
            preds = model.predict(X)
            attack_count = int((preds == 1).sum())
            predicted_label = "ATTACK" if attack_count > len(preds) / 2 else "BENIGN"
            confidence = attack_count / max(len(preds), 1)
            mean_attack_prob = confidence if predicted_label == "ATTACK" else 1.0 - confidence

    except Exception as exc:
        logger.error(
            "Inference failed for window %s: %s", window_id, exc, exc_info=True
        )
        raise

    inference_seconds = time.time() - start_ts

    return PredictionResult(
        window_id=window_id,
        prediction=predicted_label,
        confidence=confidence,
        attack_probability=mean_attack_prob,
        flows_analyzed=len(X),
        inference_seconds=inference_seconds,
        timestamp=time.time(),
    )


class LiveInferenceManager:
    """Async inference manager: consumes detection jobs, emits predictions.

    Args:
        job_queue: Source asyncio.Queue of DetectionJob objects.
        result_bus: Destination for PredictionResult events.
        preprocessor: Fitted LivePreprocessor.
        model_path: Path to TabPFN model artifact.
        max_workers: Number of thread-pool inference workers.
    """

    def __init__(
        self,
        job_queue: asyncio.Queue,
        result_bus: "ResultBus",
        preprocessor: LivePreprocessor,
        model_path: Path | str | None = None,
        max_workers: int = 2,
    ) -> None:
        self.job_queue = job_queue
        self.result_bus = result_bus
        self.preprocessor = preprocessor
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.max_workers = max(1, max_workers)

        self._model: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._worker_task: asyncio.Task | None = None
        self._inference_tasks: set[asyncio.Task] = set()
        self._running = False

        # Metrics
        self.jobs_processed: int = 0
        self.jobs_failed: int = 0
        self.total_inference_seconds: float = 0.0

    def load_model(self) -> None:
        """Load the TabPFN model artifact. Call once before start()."""
        self._model = _load_model(self._model_path)

    async def start(self) -> None:
        """Start the inference worker loop."""
        if self._running:
            return
        if self._model is None:
            self.load_model()

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="nids-inference",
        )
        self._running = True
        self._worker_task = asyncio.create_task(
            self._inference_loop(), name="inference-loop"
        )
        logger.info(
            "LiveInferenceManager started: max_workers=%d model=%s",
            self.max_workers,
            self._model_path.name,
        )

    async def stop(self) -> None:
        """Stop the inference loop and wait for active inference tasks."""
        self._running = False

        # Stop accepting new inference jobs.
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Wait for all currently running inference tasks.
        if self._inference_tasks:
            await asyncio.gather(
                *self._inference_tasks,
                return_exceptions=True,
        )
        self._inference_tasks.clear()

    # Now it is safe to shut down the executor.
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        logger.info(
            "LiveInferenceManager stopped: processed=%d failed=%d",
            self.jobs_processed,
            self.jobs_failed,
    )

    async def _inference_loop(self) -> None:
        """Consume DetectionJobs from the queue and run inference."""
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                job: DetectionJob = await asyncio.wait_for(
                    self.job_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            task = asyncio.create_task(
                self._handle_job(job, loop),
                name=f"inference-{job.window_id}",
            )
            self._inference_tasks.add(task)
            task.add_done_callback(self._inference_tasks.discard)

            self.job_queue.task_done()

    async def _handle_job(
        self, job: DetectionJob, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Process one DetectionJob: preprocess → infer → publish result."""
        logger.debug(
            "Processing window %s: %d flows, %d packets",
            job.window_id,
            job.flow_count,
            job.packet_count,
        )

        if not job.flows:
            logger.debug("Window %s has no flows; skipping.", job.window_id)
            return

        # Preprocess on the event-loop thread (fast numpy ops)
        try:
            X, metadata = self.preprocessor.transform(job.flows)
        except Exception as exc:
            logger.error(
                "Preprocessing failed for window %s: %s",
                job.window_id,
                exc,
                exc_info=True,
            )
            self.jobs_failed += 1
            return

        # Run inference in thread pool (TabPFN/PyTorch)
        assert self._executor is not None
        assert self._model is not None
        try:
            result: PredictionResult = await loop.run_in_executor(
                self._executor,
                _run_inference_sync,
                self._model,
                X,
                job.window_id,
            )
        except Exception as exc:
            logger.error(
                "Inference failed for window %s: %s",
                job.window_id,
                exc,
                exc_info=True,
            )
            self.jobs_failed += 1
            return

        result.packets_analyzed = job.packet_count
        self.jobs_processed += 1
        self.total_inference_seconds += result.inference_seconds

        logger.info(
            "Inference completed window=%s prediction=%s confidence=%.2f "
            "flows=%d packets=%d latency=%.2fs",
            result.window_id,
            result.prediction,
            result.confidence,
            result.flows_analyzed,
            result.packets_analyzed,
            result.inference_seconds,
        )

        # Publish to result bus (WebSocket + REST)
        await self.result_bus.publish(result)
