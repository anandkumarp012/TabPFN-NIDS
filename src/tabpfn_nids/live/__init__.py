"""Live network capture and real-time intrusion detection pipeline.

This subpackage extends the offline PCAP pipeline with a continuous
live monitoring mode. It reuses the existing flow builder, feature
pipeline, and inference manager without modification.

Architecture::

    TShark → PacketQueue → FlowSession → WindowManager
        → LivePreprocessor → LiveInference → ResultBus → WebSocket
"""

from __future__ import annotations

from tabpfn_nids.live.config import LiveCaptureConfig, load_live_config

__all__ = ["LiveCaptureConfig", "load_live_config"]
