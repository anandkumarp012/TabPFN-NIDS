"""Sliding-window detection for live traffic.

Implements configurable sliding windows over completed flows.

Window semantics:
    Window 1: 0s – window_size_seconds
    Window 2: step_seconds – (step_seconds + window_size_seconds)
    Window 3: 2*step_seconds – (2*step_seconds + window_size_seconds)
    ...

A flow is included in window W if:
    flow.end_time >= window_start  AND  flow.start_time < window_end

This means a flow that spans multiple windows appears in each window it
touches.  This is correct for our per-flow feature representation because
each FlowRecord is self-contained (it stores all its own statistics); we
are not double-counting events — we are asking "which active/completed
flows were part of the network during this window?".

Each window emits a DetectionJob containing the full list of FlowRecord
objects relevant to that window. The job is pushed into the inference
queue.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tabpfn_nids.flows.flow_builder import FlowRecord

logger = logging.getLogger(__name__)


@dataclass
class DetectionJob:
    """A batch of flows ready for feature extraction and inference."""

    window_id: str
    window_start: float      # epoch seconds
    window_end: float        # epoch seconds
    flows: list[FlowRecord]
    submitted_at: float = field(default_factory=time.time)

    @property
    def flow_count(self) -> int:
        return len(self.flows)

    @property
    def packet_count(self) -> int:
        return sum(f.total_packets for f in self.flows)


class WindowManager:
    """Aggregates completed flows into sliding detection windows.

    Args:
        window_size_seconds: Duration of each detection window.
        step_seconds: How far to advance between consecutive windows.
        job_queue: Target asyncio.Queue for DetectionJob objects.
        max_job_queue_size: Upper bound for the job queue (backpressure).
    """

    def __init__(
        self,
        window_size_seconds: float = 10.0,
        step_seconds: float = 5.0,
        job_queue: asyncio.Queue | None = None,
        max_job_queue_size: int = 100,
    ) -> None:
        self.window_size = window_size_seconds
        self.step = step_seconds
        self.job_queue: asyncio.Queue[DetectionJob] = (
            job_queue if job_queue is not None
            else asyncio.Queue(maxsize=max_job_queue_size)
        )

        # All flows seen so far (bounded by _prune below)
        self._all_flows: list[FlowRecord] = []
        self._window_counter: int = 0
        self._last_emit_time: float = 0.0
        self._start_time: float = time.time()

        # Stats
        self.windows_emitted: int = 0
        self.windows_dropped: int = 0

    def add_flows(self, flows: list[FlowRecord]) -> None:
        """Add newly completed flows to the window buffer.

        Args:
            flows: Completed FlowRecord objects from LiveFlowSession.
        """
        self._all_flows.extend(flows)

    async def tick(self, current_time: float | None = None) -> None:
        """Check whether the next window is ready and emit a DetectionJob.

        Should be called regularly (e.g., every second from the pipeline).

        Args:
            current_time: Override wall-clock time (for testing).
        """
        now = current_time if current_time is not None else time.time()

        # Compute the window that should be emitted now
        elapsed = now - self._start_time
        next_emit_at = self._last_emit_time + self.step

        if elapsed < self.window_size:
            # Not enough time has passed to fill the first window
            return

        if now < self._start_time + self._last_emit_time + self.step:
            # Not time for the next step yet
            pass

        # Determine which windows are overdue
        while True:
            window_start = self._start_time + self._window_counter * self.step
            window_end = window_start + self.window_size

            if window_end > now:
                break  # Window not yet closed

            self._window_counter += 1
            self._last_emit_time = (self._window_counter - 1) * self.step

            # Select flows relevant to this window
            window_flows = [
                f for f in self._all_flows
                if f.end_time >= window_start and f.start_time < window_end
            ]

            window_id = f"window-{self.windows_emitted:05d}"
            job = DetectionJob(
                window_id=window_id,
                window_start=window_start,
                window_end=window_end,
                flows=window_flows,
            )

            # Non-blocking put — drop with warning if queue is full
            try:
                self.job_queue.put_nowait(job)
                self.windows_emitted += 1
                logger.info(
                    "Window %s: flows=%d packets=%d (%.1f–%.1fs)",
                    window_id,
                    job.flow_count,
                    job.packet_count,
                    window_start - self._start_time,
                    window_end - self._start_time,
                )
            except asyncio.QueueFull:
                self.windows_dropped += 1
                logger.warning(
                    "Inference queue full: window %s dropped (dropped_total=%d). "
                    "Increase inference.queue_size or max_workers.",
                    window_id,
                    self.windows_dropped,
                )

        # Prune flows that are older than 2× window_size to bound memory
        cutoff = now - 2 * self.window_size
        before = len(self._all_flows)
        self._all_flows = [f for f in self._all_flows if f.end_time >= cutoff]
        pruned = before - len(self._all_flows)
        if pruned > 0:
            logger.debug("Pruned %d old flows from window buffer", pruned)

    def get_stats(self) -> dict[str, Any]:
        """Return window manager statistics."""
        return {
            "windows_emitted": self.windows_emitted,
            "windows_dropped": self.windows_dropped,
            "buffered_flows": len(self._all_flows),
            "job_queue_size": self.job_queue.qsize(),
        }
