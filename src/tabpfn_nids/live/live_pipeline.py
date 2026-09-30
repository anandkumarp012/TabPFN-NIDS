"""Live pipeline orchestrator.

The LivePipeline is the single entry point for starting and stopping
the full live monitoring pipeline. It wires together:

    TSharkCapture → packet_queue → LiveFlowSession → window_manager
        → job_queue → LiveInferenceManager → result_bus → WebSocket

Lifecycle:
    1. start(interface, config) — validates TShark, starts all layers.
    2. (running) — continuous monitoring.
    3. stop() — graceful shutdown in reverse order.

Thread-safety: All public methods are async-safe and designed to be
called from FastAPI endpoints. The pipeline state is guarded by an
asyncio.Lock to prevent concurrent start/stop calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tabpfn_nids.live.config import LiveCaptureConfig
from tabpfn_nids.live.flow_session import LiveFlowSession
from tabpfn_nids.live.interface_manager import validate_interface, validate_tshark
from tabpfn_nids.live.live_inference import LiveInferenceManager
from tabpfn_nids.live.live_preprocessor import LivePreprocessor
from tabpfn_nids.live.result_bus import ResultBus
from tabpfn_nids.live.tshark_capture import TSharkCapture
from tabpfn_nids.live.window_manager import WindowManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Snapshot of the current pipeline state."""
    running: bool = False
    interface: str | None = None
    packets_captured: int = 0
    packets_processed: int = 0
    packets_dropped: int = 0
    active_flows: int = 0
    windows_processed: int = 0
    windows_dropped: int = 0
    detections: int = 0
    job_queue_size: int = 0
    inference_jobs_failed: int = 0
    start_time: float | None = None
    tshark_pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interface": self.interface,
            "packets_captured": self.packets_captured,
            "packets_processed": self.packets_processed,
            "packets_dropped": self.packets_dropped,
            "active_flows": self.active_flows,
            "windows_processed": self.windows_processed,
            "windows_dropped": self.windows_dropped,
            "detections": self.detections,
            "job_queue_size": self.job_queue_size,
            "inference_jobs_failed": self.inference_jobs_failed,
            "uptime_seconds": (
                round(time.time() - self.start_time, 1)
                if self.start_time else 0
            ),
            "tshark_pid": self.tshark_pid,
        }


class LivePipeline:
    """Full live monitoring pipeline orchestrator.

    Args:
        result_bus: Shared ResultBus for predictions.
        config: LiveCaptureConfig with all parameters.
    """

    def __init__(
        self,
        result_bus: ResultBus,
        config: LiveCaptureConfig | None = None,
    ) -> None:
        self.result_bus = result_bus
        self.config = config
        self._lock = asyncio.Lock()
        self._running = False

        # Pipeline components (None when stopped)
        self._capture: TSharkCapture | None = None
        self._flow_session: LiveFlowSession | None = None
        self._window_manager: WindowManager | None = None
        self._inference_manager: LiveInferenceManager | None = None
        self._preprocessor: LivePreprocessor | None = None
        self._packet_queue: asyncio.Queue | None = None
        self._job_queue: asyncio.Queue | None = None
        self._tick_task: asyncio.Task | None = None

    async def start(
        self,
        interface: str,
        config: LiveCaptureConfig | None = None,
    ) -> None:
        """Start the complete live pipeline.

        Args:
            interface: Network interface name or index.
            config: Runtime config override.

        Raises:
            RuntimeError: If already running.
            Various: If TShark/Npcap validation fails.
        """
        async with self._lock:
            if self._running:
                raise RuntimeError(
                    "Live pipeline is already running. "
                    "Call /capture/stop before starting a new session."
                )

            cfg = config or self.config or LiveCaptureConfig()

            # 1. Validate TShark and interface
            logger.info("Validating TShark...")
            validate_tshark(cfg.capture.tshark_path)
            logger.info("Validating interface '%s'...", interface)
            iface = validate_interface(interface, cfg.capture.tshark_path)
            resolved_interface = iface.name

            # 2. Load preprocessing artifacts (fail fast before capture starts)
            logger.info("Loading preprocessing artifacts...")
            self._preprocessor = LivePreprocessor(
                scaler_path=cfg.inference.scaler_path,
                feature_schema_path=cfg.inference.feature_schema_path,
            )
            self._preprocessor.load()

            # 3. Build queues
            self._packet_queue = asyncio.Queue(
                maxsize=cfg.capture.packet_queue_size
            )
            self._job_queue = asyncio.Queue(
                maxsize=cfg.inference.queue_size
            )

            # 4. Build pipeline components
            self._capture = TSharkCapture(
                interface=resolved_interface,
                tshark_path=cfg.capture.tshark_path,
                packet_queue=self._packet_queue,
                queue_size=cfg.capture.packet_queue_size,
                on_error=self._on_tshark_error,
            )

            self._flow_session = LiveFlowSession(
                packet_queue=self._packet_queue,
                flow_timeout=cfg.detection.flow_timeout_seconds * 2,
                idle_timeout=cfg.detection.flow_timeout_seconds,
                timeout_sweep_interval=1.0,
            )

            self._window_manager = WindowManager(
                window_size_seconds=cfg.detection.window_size_seconds,
                step_seconds=cfg.detection.step_seconds,
                job_queue=self._job_queue,
                max_job_queue_size=cfg.inference.queue_size,
            )

            self._inference_manager = LiveInferenceManager(
                job_queue=self._job_queue,
                result_bus=self.result_bus,
                preprocessor=self._preprocessor,
                model_path=cfg.inference.model_path,
                max_workers=cfg.inference.max_workers,
            )

            # 5. Start pipeline in correct order
            logger.info("Starting flow session...")
            await self._flow_session.start()

            logger.info("Starting inference manager...")
            await self._inference_manager.start()

            logger.info("Starting TShark capture (interface=%s)...", resolved_interface)
            await self._capture.start()

            # 6. Start the window tick task
            self._tick_task = asyncio.create_task(
                self._window_tick_loop(), name="window-tick"
            )

            self._running = True
            logger.info(
                "Live pipeline ACTIVE: interface=%s window=%.0fs step=%.0fs "
                "flow_timeout=%.0fs workers=%d",
                resolved_interface,
                cfg.detection.window_size_seconds,
                cfg.detection.step_seconds,
                cfg.detection.flow_timeout_seconds,
                cfg.inference.max_workers,
            )

    async def stop(self) -> PipelineStatus:
        """Gracefully stop the pipeline.

        Returns:
            Final PipelineStatus snapshot.
        """
        async with self._lock:
            if not self._running:
                return PipelineStatus(running=False)

            logger.info("Stopping live pipeline...")

            # Stop in reverse order
            # 1. Stop tick task
            if self._tick_task and not self._tick_task.done():
                self._tick_task.cancel()
                try:
                    await self._tick_task
                except asyncio.CancelledError:
                    pass

            # 2. Stop TShark (no new packets)
            capture_stats = None
            if self._capture:
                capture_stats = await self._capture.stop()

            # 3. Drain packet queue → flow session
            if self._flow_session:
                flushed = await self._flow_session.stop(flush=True)
                # Push flushed flows into window manager for a final window
                if flushed and self._window_manager:
                    self._window_manager.add_flows(flushed)

            # 4. Stop inference manager (finishes in-flight jobs)
            if self._inference_manager:
                await self._inference_manager.stop()

            status = self.get_status()
            status.running = False

            # Reset all components
            self._capture = None
            self._flow_session = None
            self._window_manager = None
            self._inference_manager = None
            self._preprocessor = None
            self._packet_queue = None
            self._job_queue = None
            self._tick_task = None
            self._running = False

            logger.info("Live pipeline stopped.")
            return status

    async def _window_tick_loop(self) -> None:
        """Periodically feed completed flows into the window manager."""
        while self._running:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

            if not self._flow_session or not self._window_manager:
                continue

            completed = await self._flow_session.get_completed_flows()
            if completed:
                self._window_manager.add_flows(completed)

            await self._window_manager.tick()

    async def _on_tshark_error(self, exc: Exception) -> None:
        """Called when TShark exits unexpectedly."""
        logger.error("TShark error: %s", exc)
        # Publish an error event to the result bus subscribers via status
        self._running = False

    def get_status(self) -> PipelineStatus:
        """Return a snapshot of the current pipeline state."""
        status = PipelineStatus(running=self._running)

        if self._capture:
            status.packets_captured = self._capture.stats.packets_captured
            status.packets_dropped = self._capture.stats.packets_dropped
            status.tshark_pid = self._capture.stats.tshark_pid
            status.start_time = self._capture.stats.start_time
            status.interface = self._capture.interface

        if self._flow_session:
            status.packets_processed = self._flow_session.packets_processed
            status.active_flows = self._flow_session.active_flow_count

        if self._window_manager:
            status.windows_processed = self._window_manager.windows_emitted
            status.windows_dropped = self._window_manager.windows_dropped
            status.job_queue_size = self._window_manager.job_queue.qsize()

        if self._inference_manager:
            status.inference_jobs_failed = self._inference_manager.jobs_failed

        status.detections = self.result_bus.attack_count

        return status

    @property
    def is_running(self) -> bool:
        """True if the pipeline is active."""
        return self._running
