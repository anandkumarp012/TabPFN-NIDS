"""Live capture configuration.

Provides LiveCaptureConfig dataclass and YAML loader. All live-mode
parameters live here; no other live module hard-codes values.

Environment variable overrides use the prefix ``NIDS_``:
    NIDS_CAPTURE_INTERFACE=Wi-Fi
    NIDS_CAPTURE_TSHARK_PATH=C:/Program Files/Wireshark/tshark.exe
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tabpfn_nids.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_LIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "live_capture.yaml"


# ---------------------------------------------------------------------------
# Sub-section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """TShark capture layer parameters."""
    # Network interface to capture on. None → must be selected via API.
    interface: str | None = None
    # Path to tshark binary; "tshark" searches PATH.
    tshark_path: str = "tshark"
    # Bounded packet queue size (prevents unbounded memory growth).
    packet_queue_size: int = 10_000
    # Seconds between cleanup sweeps inside TShark capture loop.
    capture_poll_interval: float = 0.05


@dataclass
class DetectionConfig:
    """Flow and sliding-window detection parameters."""
    # Sliding window size in seconds.
    window_size_seconds: float = 10.0
    # Sliding window step (overlap) in seconds.
    step_seconds: float = 5.0
    # Flow idle timeout in seconds (matches offline FlowBuilder default reduced
    # for live latency; overrides idle_timeout in FlowBuilder).
    flow_timeout_seconds: float = 10.0
    # Minimum packets in a flow to submit for inference.
    min_flow_packets: int = 2


@dataclass
class LiveInferenceConfig:
    """Async inference worker pool parameters."""
    # Maximum parallel TabPFN inference workers.
    max_workers: int = 2
    # Bounded inference queue size.
    queue_size: int = 100
    # Path to trained model artifact; None → use ModelRegistry discovery.
    model_path: str | None = None
    # Path to fitted scaler artifact; None → try default path.
    scaler_path: str | None = None
    # Path to feature schema JSON; None → try default path.
    feature_schema_path: str | None = None


@dataclass
class WebSocketConfig:
    """WebSocket broadcast configuration."""
    enabled: bool = True
    # Seconds between periodic status broadcasts.
    status_interval_seconds: float = 2.0


@dataclass
class LiveCaptureConfig:
    """Top-level container for all live capture configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    inference: LiveInferenceConfig = field(default_factory=LiveInferenceConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: LiveCaptureConfig) -> None:
    """Apply NIDS_ environment variable overrides in-place."""
    env_map: dict[str, tuple[Any, str, type]] = {
        "NIDS_CAPTURE_INTERFACE":       (cfg.capture,   "interface",        str),
        "NIDS_CAPTURE_TSHARK_PATH":     (cfg.capture,   "tshark_path",      str),
        "NIDS_CAPTURE_QUEUE_SIZE":      (cfg.capture,   "packet_queue_size", int),
        "NIDS_DETECTION_WINDOW_SIZE":   (cfg.detection, "window_size_seconds", float),
        "NIDS_DETECTION_STEP":          (cfg.detection, "step_seconds",     float),
        "NIDS_DETECTION_FLOW_TIMEOUT":  (cfg.detection, "flow_timeout_seconds", float),
        "NIDS_INFERENCE_MAX_WORKERS":   (cfg.inference, "max_workers",      int),
        "NIDS_INFERENCE_QUEUE_SIZE":    (cfg.inference, "queue_size",       int),
        "NIDS_INFERENCE_MODEL_PATH":    (cfg.inference, "model_path",       str),
        "NIDS_INFERENCE_SCALER_PATH":   (cfg.inference, "scaler_path",      str),
    }
    for env_key, (obj, attr, cast) in env_map.items():
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                setattr(obj, attr, cast(raw))
                logger.debug("Env override %s=%s", env_key, raw)
            except ValueError as exc:
                logger.warning("Invalid env var %s=%r: %s", env_key, raw, exc)


def _build_section(cls: type, data: dict[str, Any] | None):
    """Instantiate a dataclass from a YAML dict, using field defaults."""
    if data is None:
        return cls()
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**valid)


def load_live_config(path: Path | str | None = None) -> LiveCaptureConfig:
    """Load live capture YAML config, falling back to defaults.

    Args:
        path: Path to YAML file. Defaults to configs/live_capture.yaml.

    Returns:
        A fully typed LiveCaptureConfig.
    """
    config_path = Path(path) if path else _DEFAULT_LIVE_CONFIG_PATH

    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Loaded live capture config from %s", config_path)
    else:
        logger.info(
            "Live capture config not found at %s; using defaults.", config_path
        )
        raw = {}

    cfg = LiveCaptureConfig(
        capture=_build_section(CaptureConfig, raw.get("capture")),
        detection=_build_section(DetectionConfig, raw.get("detection")),
        inference=_build_section(LiveInferenceConfig, raw.get("inference")),
        websocket=_build_section(WebSocketConfig, raw.get("websocket")),
    )

    _apply_env_overrides(cfg)
    return cfg
