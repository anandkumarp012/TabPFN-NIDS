"""Async TShark live packet capture.

Spawns TShark as a subprocess, reads its structured ``-T fields`` output
asynchronously, parses each line into a PacketRecord, and pushes it into a
bounded asyncio.Queue.

The TShark process runs continuously in the background; packet capture is
never blocked by flow processing or model inference. A separate async task
reads the TShark stdout line-by-line.

Design rules enforced here:
- No blocking subprocess.run() for the capture process.
- Bounded queue prevents unlimited memory growth.
- Dropped packets are counted and logged; they are never silently discarded.
- TShark crashes are detected and re-reported via the error callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)

# TShark field list — identical to _extract_with_tshark in extractor.py.
# Using the same field ordering ensures parity between offline and live
# parsing logic.
_TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "ip.proto",
    "frame.len",
    "tcp.flags",
    "tcp.len",
    "udp.length",
]

_FIELD_COUNT = len(_TSHARK_FIELDS)
_SEPARATOR = "|"


def _build_tshark_cmd(
    interface: str,
    tshark_path: str = "tshark",
) -> list[str]:
    """Build the TShark capture command for live monitoring.

    Uses the same -T fields approach as the offline extractor so that
    parsing logic is shared.

    Args:
        interface: Network interface name or index.
        tshark_path: Path to TShark binary.

    Returns:
        Command list ready for asyncio.create_subprocess_exec.
    """
    cmd = [
        tshark_path,
        "-i", interface,
        "-l",                          # line-buffered output
        "-T", "fields",
        *[arg for f in _TSHARK_FIELDS for arg in ("-e", f)],
        "-E", f"separator={_SEPARATOR}",
        "-E", "occurrence=f",          # first occurrence of each field
        "-q",                          # suppress packet summary to stderr
    ]
    return cmd


def _parse_tshark_line(line: str) -> PacketRecord | None:
    """Parse one TShark -T fields output line into a PacketRecord.

    Uses the identical parsing logic as _extract_with_tshark() in
    extractor.py to guarantee live/offline feature parity.

    Args:
        line: A single output line from TShark.

    Returns:
        PacketRecord on success, None if the line cannot be parsed or
        has no IP addresses (non-IP traffic).
    """
    parts = line.rstrip("\n\r").split(_SEPARATOR)
    if len(parts) < _FIELD_COUNT:
        return None

    try:
        ts_str, src_ip, dst_ip = parts[0], parts[1], parts[2]

        if not src_ip or not dst_ip:
            return None

        ts = float(ts_str) if ts_str else 0.0
        length = int(parts[8]) if parts[8] else 0

        if parts[3]:  # tcp.srcport present → TCP
            src_port = int(parts[3])
            dst_port = int(parts[4]) if parts[4] else 0
            proto_num = 6
            proto_name = "tcp"
            tcp_flags = int(parts[9], 16) if parts[9] else 0
            payload_len = int(parts[10]) if parts[10] else 0
        elif parts[5]:  # udp.srcport present → UDP
            src_port = int(parts[5])
            dst_port = int(parts[6]) if parts[6] else 0
            proto_num = 17
            proto_name = "udp"
            tcp_flags = 0
            payload_len = int(parts[11]) if parts[11] else 0
        else:           # ICMP or other
            proto_num = int(parts[7]) if parts[7] else 0
            proto_name = "icmp" if proto_num == 1 else "other"
            src_port = 0
            dst_port = 0
            tcp_flags = 0
            payload_len = 0

        return PacketRecord(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto_num,
            protocol_name=proto_name,
            length=length,
            tcp_flags=tcp_flags,
            payload_length=payload_len,
        )

    except (ValueError, IndexError):
        return None


@dataclass
class CaptureStatistics:
    """Rolling counters for the live capture layer."""
    packets_captured: int = 0
    packets_dropped: int = 0
    parse_errors: int = 0
    start_time: float = field(default_factory=time.time)
    tshark_pid: int | None = None

    @property
    def packets_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.packets_captured / max(elapsed, 1e-6)


class TSharkCapture:
    """Manages an async TShark live capture session.

    Args:
        interface: Network interface name or index.
        tshark_path: Path to TShark binary.
        packet_queue: Bounded asyncio.Queue for PacketRecord objects.
        on_error: Optional async callback invoked when TShark exits unexpectedly.
    """

    def __init__(
        self,
        interface: str,
        tshark_path: str = "tshark",
        packet_queue: asyncio.Queue | None = None,
        queue_size: int = 10_000,
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self.interface = interface
        self.tshark_path = tshark_path
        self.packet_queue: asyncio.Queue[PacketRecord] = (
            packet_queue if packet_queue is not None
            else asyncio.Queue(maxsize=queue_size)
        )
        self.on_error = on_error
        self.stats = CaptureStatistics()

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start TShark and begin reading packets into the queue.

        Raises:
            RuntimeError: If already running.
            OSError: If TShark cannot be launched.
        """
        if self._running:
            raise RuntimeError("TShark capture is already running.")

        cmd = _build_tshark_cmd(self.interface, self.tshark_path)
        logger.info(
            "Starting TShark: interface=%s cmd=%s",
            self.interface,
            " ".join(cmd),
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OSError(
                f"TShark executable not found at '{self.tshark_path}'. "
                "Ensure Wireshark is installed and TShark is on PATH."
            ) from exc

        self.stats.tshark_pid = self._process.pid
        self.stats.start_time = time.time()
        self._running = True

        logger.info(
            "TShark process started pid=%d interface=%s",
            self._process.pid,
            self.interface,
        )

        # Start background reader tasks
        self._reader_task = asyncio.create_task(
            self._read_stdout(), name="tshark-reader"
        )
        asyncio.create_task(
            self._watch_stderr(), name="tshark-stderr-watcher"
        )

    async def stop(self) -> CaptureStatistics:
        """Gracefully stop TShark and the reader task.

        Returns:
            Final capture statistics.
        """
        logger.info("Stopping TShark capture...")
        self._running = False

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass

        logger.info(
            "TShark stopped: captured=%d dropped=%d errors=%d",
            self.stats.packets_captured,
            self.stats.packets_dropped,
            self.stats.parse_errors,
        )
        return self.stats

    async def _read_stdout(self) -> None:
        """Continuously read TShark stdout and push PacketRecords to queue."""
        assert self._process is not None
        assert self._process.stdout is not None

        log_drop_every = 1000  # log a warning every N dropped packets

        while self._running:
            try:
                line = await self._process.stdout.readline()
            except asyncio.CancelledError:
                break

            if not line:
                # EOF — TShark exited
                if self._running:
                    logger.error(
                        "TShark stdout closed unexpectedly (pid=%s, "
                        "returncode=%s). Capture stopped.",
                        self.stats.tshark_pid,
                        self._process.returncode,
                    )
                    self._running = False
                    if self.on_error:
                        asyncio.create_task(
                            self.on_error(
                                RuntimeError(
                                    "TShark exited unexpectedly. "
                                    "Check interface, Npcap, and permissions."
                                )
                            )
                        )
                break

            decoded = line.decode("utf-8", errors="replace")
            record = _parse_tshark_line(decoded)
            if record is None:
                self.stats.parse_errors += 1
                continue

            self.stats.packets_captured += 1

            # Non-blocking put; drop if full (queue backpressure)
            try:
                self.packet_queue.put_nowait(record)
            except asyncio.QueueFull:
                self.stats.packets_dropped += 1
                if self.stats.packets_dropped % log_drop_every == 1:
                    logger.warning(
                        "Packet queue full: dropped=%d (queue maxsize=%d). "
                        "Consider increasing packet_queue_size or reducing "
                        "inference latency.",
                        self.stats.packets_dropped,
                        self.packet_queue.maxsize,
                    )

    async def _watch_stderr(self) -> None:
        """Log TShark stderr for diagnostics (permissions, Npcap errors)."""
        assert self._process is not None
        assert self._process.stderr is not None
        while self._running:
            try:
                line = await self._process.stderr.readline()
            except asyncio.CancelledError:
                break
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                # Log at WARNING level so it surfaces without being DEBUG noise
                logger.warning("TShark stderr: %s", decoded)

    @property
    def is_running(self) -> bool:
        """True while the capture is active."""
        return self._running

    def get_queue_size(self) -> int:
        """Current number of packets waiting in the queue."""
        return self.packet_queue.qsize()
