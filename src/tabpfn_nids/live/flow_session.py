"""Live flow session: wraps FlowBuilder for continuous operation.

Reuses the existing FlowBuilder and FlowRecord without modification.
The LiveFlowSession acts as a thin adapter that:
  1. Consumes PacketRecord objects from the capture queue.
  2. Feeds them into FlowBuilder.
  3. Periodically checks for timeout-expired flows.
  4. Returns completed FlowRecord objects to the window manager.

The flow semantics (bidirectional canonical 5-tuple, FIN/RST teardown,
idle/flow timeouts) are IDENTICAL to the offline PCAP pipeline because
the same FlowBuilder is used.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from tabpfn_nids.flows.flow_builder import FlowBuilder, FlowRecord
from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)


class LiveFlowSession:
    """Continuously processes packets into bidirectional flows.

    Args:
        packet_queue: Source of PacketRecord objects from TShark.
        flow_timeout: Maximum total flow duration before forced close.
        idle_timeout: Maximum inactivity gap before flow close.
        timeout_sweep_interval: Seconds between timeout sweep checks.
        max_packets_per_flow: Safety cap per flow.
    """

    def __init__(
        self,
        packet_queue: asyncio.Queue,
        flow_timeout: float = 120.0,
        idle_timeout: float = 10.0,
        timeout_sweep_interval: float = 1.0,
        max_packets_per_flow: int = 1_000_000,
    ) -> None:
        self.packet_queue = packet_queue
        self.timeout_sweep_interval = timeout_sweep_interval

        self._builder = FlowBuilder(
            flow_timeout=flow_timeout,
            idle_timeout=idle_timeout,
            max_packets_per_flow=max_packets_per_flow,
        )
        self._completed_flows: asyncio.Queue[FlowRecord] = asyncio.Queue()
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._sweep_task: asyncio.Task | None = None
        self._packets_processed = 0

    async def start(self) -> None:
        """Start background packet processing and timeout sweep tasks."""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(
            self._process_packets(), name="flow-processor"
        )
        self._sweep_task = asyncio.create_task(
            self._timeout_sweep(), name="flow-timeout-sweep"
        )
        logger.info("LiveFlowSession started")

    async def stop(self, flush: bool = True) -> list[FlowRecord]:
        """Stop processing and optionally flush remaining active flows.

        Args:
            flush: If True, finalize all active flows before stopping.

        Returns:
            List of flows finalized during shutdown.
        """
        self._running = False

        for task in (self._process_task, self._sweep_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        flushed: list[FlowRecord] = []
        if flush:
            flushed = self._builder.flush_all()
            logger.info(
                "LiveFlowSession shutdown: flushed %d active flows", len(flushed)
            )

        return flushed

    async def get_completed_flows(self) -> list[FlowRecord]:
        """Drain all completed flows available right now (non-blocking).

        Returns:
            List of completed FlowRecord objects.
        """
        flows: list[FlowRecord] = []
        while not self._completed_flows.empty():
            try:
                flows.append(self._completed_flows.get_nowait())
            except asyncio.QueueEmpty:
                break
        return flows

    @property
    def active_flow_count(self) -> int:
        """Number of currently open flows in the builder."""
        return self._builder.active_flow_count

    @property
    def packets_processed(self) -> int:
        """Total packets consumed from the queue."""
        return self._packets_processed

    async def _process_packets(self) -> None:
        """Consume PacketRecord objects and feed them to FlowBuilder."""
        while self._running:
            try:
                # Short timeout so we can check _running regularly
                pkt: PacketRecord = await asyncio.wait_for(
                    self.packet_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._builder.add_packet(pkt)
            self._packets_processed += 1
            self.packet_queue.task_done()

            # Drain any flows closed by FIN/RST in this packet
            completed = self._builder.get_completed()
            for flow in completed:
                await self._completed_flows.put(flow)

    async def _timeout_sweep(self) -> None:
        """Periodically expire idle/long flows and push them downstream."""
        while self._running:
            try:
                await asyncio.sleep(self.timeout_sweep_interval)
            except asyncio.CancelledError:
                break

            # Force a timeout check by calling _check_timeouts directly
            # using the current wall-clock time
            self._builder._check_timeouts(time.time())
            completed = self._builder.get_completed()
            for flow in completed:
                await self._completed_flows.put(flow)

            if completed:
                logger.debug(
                    "Timeout sweep: %d flows expired, %d active",
                    len(completed),
                    self._builder.active_flow_count,
                )
