"""TShark interface discovery and validation.

Provides utilities to list available network interfaces via
``tshark -D`` and validate that TShark/Npcap are present and
operational.  Designed for Windows (Npcap + TShark from Wireshark).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NetworkInterface:
    """A single network interface as reported by ``tshark -D``."""

    index: int           # 1-based index from tshark -D
    name: str            # Interface name (e.g. "Wi-Fi", "Ethernet")
    description: str     # Full display string from tshark -D


class TSharkNotFoundError(RuntimeError):
    """Raised when tshark binary cannot be found or executed."""
    pass


class NpcapNotAvailableError(RuntimeError):
    """Raised when tshark reports Npcap is not installed."""
    pass


class InterfaceNotFoundError(ValueError):
    """Raised when the requested interface is not visible to tshark."""
    pass


def validate_tshark(tshark_path: str = "tshark") -> str:
    """Verify that tshark is installed and executable.

    Args:
        tshark_path: Path to tshark binary (or just "tshark" to use PATH).

    Returns:
        The tshark version string.

    Raises:
        TSharkNotFoundError: If the binary is not found or not executable.
        NpcapNotAvailableError: If tshark cannot open capture interfaces.
    """
    # First: try to find the binary
    resolved = shutil.which(tshark_path)
    if resolved is None and not Path(tshark_path).is_file():
        raise TSharkNotFoundError(
            f"TShark executable was not found at '{tshark_path}'.\n"
            "Suggestion: Install Wireshark (which includes TShark and Npcap) "
            "from https://www.wireshark.org/download.html and ensure TShark "
            "is on your PATH, or set the tshark_path in live_capture.yaml."
        )

    # Second: check version
    try:
        result = subprocess.run(
            [tshark_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise TSharkNotFoundError(
                f"tshark --version returned code {result.returncode}.\n"
                f"stderr: {result.stderr.strip()}"
            )
        version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        logger.info("TShark validated: %s", version_line)
        return version_line
    except FileNotFoundError:
        raise TSharkNotFoundError(
            f"TShark executable was not found at '{tshark_path}'.\n"
            "Suggestion: Install Wireshark with TShark/Npcap or configure "
            "tshark_path in live_capture.yaml."
        )
    except subprocess.TimeoutExpired:
        raise TSharkNotFoundError(
            f"tshark --version timed out. Is '{tshark_path}' a valid executable?"
        )


def list_interfaces(tshark_path: str = "tshark") -> list[NetworkInterface]:
    """List available capture interfaces via ``tshark -D``.

    Args:
        tshark_path: Path to tshark binary.

    Returns:
        List of NetworkInterface objects.

    Raises:
        TSharkNotFoundError: If tshark is not found.
        NpcapNotAvailableError: If no interfaces are available (Npcap missing).
    """
    validate_tshark(tshark_path)

    try:
        result = subprocess.run(
            [tshark_path, "-D"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        raise TSharkNotFoundError(
            f"TShark executable not found at '{tshark_path}'."
        )
    except subprocess.TimeoutExpired:
        raise TSharkNotFoundError("tshark -D timed out.")

    stderr = result.stderr.strip()
    if "npcap" in stderr.lower() or "winpcap" in stderr.lower() or (
        result.returncode != 0 and not result.stdout.strip()
    ):
        raise NpcapNotAvailableError(
            "TShark reports no capture interfaces. Npcap is likely not "
            "installed or the current user lacks capture permissions.\n"
            "Suggestion: Install Npcap from https://npcap.com/ and restart, "
            "or run as administrator."
        )

    interfaces: list[NetworkInterface] = []
    # tshark -D output format: "1. \Device\NPF_{GUID} (Interface Name)"
    # or on some versions: "1. Interface Name"
    pattern = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)
    for match in pattern.finditer(result.stdout):
        idx = int(match.group(1))
        desc = match.group(2).strip()

        # Extract human-readable name from "(Name)" suffix if present
        name_match = re.search(r"\(([^)]+)\)\s*$", desc)
        name = name_match.group(1) if name_match else desc
        interfaces.append(NetworkInterface(index=idx, name=name, description=desc))

    if not interfaces:
        raise NpcapNotAvailableError(
            "No network interfaces found. Ensure Npcap is installed and "
            "run as administrator (or grant capture permissions)."
        )

    logger.info("Found %d capture interfaces", len(interfaces))
    return interfaces


def validate_interface(
    interface: str,
    tshark_path: str = "tshark",
) -> NetworkInterface:
    """Validate that the specified interface is available for capture.

    Accepts either the interface name (e.g. ``"Wi-Fi"``) or its 1-based
    index as a string (e.g. ``"1"``).

    Args:
        interface: Interface name or index string.
        tshark_path: Path to tshark binary.

    Returns:
        The matching NetworkInterface.

    Raises:
        InterfaceNotFoundError: If no matching interface is found.
    """
    available = list_interfaces(tshark_path)

    # Try numeric index first
    if interface.isdigit():
        idx = int(interface)
        for iface in available:
            if iface.index == idx:
                logger.info("Interface '%s' validated (index %d)", iface.name, idx)
                return iface
        raise InterfaceNotFoundError(
            f"Interface index {idx} not found. "
            f"Available: {[str(i.index) + '. ' + i.name for i in available]}"
        )

    # Try by name (case-insensitive substring match)
    lower = interface.lower()
    for iface in available:
        if lower == iface.name.lower() or lower in iface.description.lower():
            logger.info(
                "Interface '%s' validated → '%s'", interface, iface.description
            )
            return iface

    raise InterfaceNotFoundError(
        f"Interface '{interface}' not found among available interfaces: "
        f"{[i.name for i in available]}.\n"
        "Run 'tshark -D' to list available interfaces and use the exact name "
        "or index number."
    )
