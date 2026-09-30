"""Unit tests for the live capture pipeline components.

These tests use no real network traffic or TShark. They test:
- Config loading and environment variable overrides
- TShark line parser correctness (PacketRecord parity with offline parser)
- WindowManager sliding window logic
- LivePreprocessor feature schema validation
- ResultBus pub-sub behavior
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tabpfn_nids.live.config import (
    CaptureConfig,
    DetectionConfig,
    LiveCaptureConfig,
    LiveInferenceConfig,
    load_live_config,
)
from tabpfn_nids.live.result_bus import PredictionResult, ResultBus
from tabpfn_nids.live.tshark_capture import _parse_tshark_line
from tabpfn_nids.live.window_manager import DetectionJob, WindowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prediction() -> PredictionResult:
    return PredictionResult(
        window_id="window-00001",
        prediction="ATTACK",
        confidence=0.87,
        attack_probability=0.87,
        flows_analyzed=12,
        packets_analyzed=543,
        inference_seconds=0.12,
        timestamp=time.time(),
    )


@pytest.fixture
def sample_flow_record():
    """Create a minimal FlowRecord for window manager tests."""
    from tabpfn_nids.flows.flow_builder import FlowRecord
    now = time.time()
    return FlowRecord(
        flow_id="abc123",
        src_ip="192.168.1.1",
        dst_ip="10.0.0.1",
        src_port=12345,
        dst_port=80,
        protocol=6,
        protocol_name="tcp",
        start_time=now - 3.0,
        end_time=now,
        duration=3.0,
        fwd_packets=5,
        bwd_packets=3,
        fwd_bytes=500,
        bwd_bytes=1200,
        total_packets=8,
        total_bytes=1700,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestLiveCaptureConfig:
    def test_defaults(self):
        cfg = LiveCaptureConfig()
        assert cfg.capture.tshark_path == "tshark"
        assert cfg.detection.window_size_seconds == 10.0
        assert cfg.detection.step_seconds == 5.0
        assert cfg.inference.max_workers == 2
        assert cfg.websocket.enabled is True

    def test_load_from_nonexistent_path_uses_defaults(self, tmp_path):
        """load_live_config falls back to defaults if file is missing."""
        cfg = load_live_config(tmp_path / "nonexistent.yaml")
        assert cfg.capture.tshark_path == "tshark"
        assert cfg.detection.window_size_seconds == 10.0

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
capture:
  tshark_path: "/custom/tshark"
  packet_queue_size: 5000
detection:
  window_size_seconds: 20.0
  step_seconds: 10.0
inference:
  max_workers: 4
"""
        config_file = tmp_path / "live_capture.yaml"
        config_file.write_text(yaml_content)
        cfg = load_live_config(config_file)
        assert cfg.capture.tshark_path == "/custom/tshark"
        assert cfg.capture.packet_queue_size == 5000
        assert cfg.detection.window_size_seconds == 20.0
        assert cfg.inference.max_workers == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NIDS_CAPTURE_TSHARK_PATH", "/env/tshark")
        monkeypatch.setenv("NIDS_DETECTION_WINDOW_SIZE", "30.0")
        monkeypatch.setenv("NIDS_INFERENCE_MAX_WORKERS", "8")
        cfg = load_live_config(None)  # Will use defaults then apply env
        assert cfg.capture.tshark_path == "/env/tshark"
        assert cfg.detection.window_size_seconds == 30.0
        assert cfg.inference.max_workers == 8


# ---------------------------------------------------------------------------
# TShark line parser tests
# ---------------------------------------------------------------------------

class TestTSharkLineParser:
    """Test _parse_tshark_line against known-good field strings."""

    def _make_line(self, fields: dict) -> str:
        """Build a fake tshark -T fields output line."""
        all_fields = [
            "frame.time_epoch", "ip.src", "ip.dst",
            "tcp.srcport", "tcp.dstport",
            "udp.srcport", "udp.dstport",
            "ip.proto", "frame.len", "tcp.flags", "tcp.len", "udp.length",
        ]
        values = [str(fields.get(f, "")) for f in all_fields]
        return "|".join(values)

    def test_tcp_packet(self):
        line = self._make_line({
            "frame.time_epoch": "1700000000.123456",
            "ip.src": "192.168.1.1",
            "ip.dst": "10.0.0.1",
            "tcp.srcport": "54321",
            "tcp.dstport": "80",
            "ip.proto": "6",
            "frame.len": "1514",
            "tcp.flags": "0x0018",  # PSH + ACK
            "tcp.len": "1460",
        })
        pkt = _parse_tshark_line(line)
        assert pkt is not None
        assert pkt.src_ip == "192.168.1.1"
        assert pkt.dst_ip == "10.0.0.1"
        assert pkt.src_port == 54321
        assert pkt.dst_port == 80
        assert pkt.protocol == 6
        assert pkt.protocol_name == "tcp"
        assert pkt.length == 1514
        assert pkt.tcp_flags == 0x0018
        assert abs(pkt.timestamp - 1700000000.123456) < 1e-3

    def test_udp_packet(self):
        line = self._make_line({
            "frame.time_epoch": "1700000001.0",
            "ip.src": "8.8.8.8",
            "ip.dst": "192.168.1.5",
            "udp.srcport": "53",
            "udp.dstport": "12345",
            "ip.proto": "17",
            "frame.len": "80",
            "udp.length": "60",
        })
        pkt = _parse_tshark_line(line)
        assert pkt is not None
        assert pkt.protocol == 17
        assert pkt.protocol_name == "udp"
        assert pkt.src_port == 53
        assert pkt.dst_port == 12345
        assert pkt.tcp_flags == 0

    def test_icmp_packet(self):
        line = self._make_line({
            "frame.time_epoch": "1700000002.5",
            "ip.src": "192.168.1.1",
            "ip.dst": "8.8.8.8",
            "ip.proto": "1",
            "frame.len": "98",
        })
        pkt = _parse_tshark_line(line)
        assert pkt is not None
        assert pkt.protocol == 1
        assert pkt.protocol_name == "icmp"
        assert pkt.src_port == 0
        assert pkt.dst_port == 0

    def test_empty_ips_return_none(self):
        line = self._make_line({"frame.time_epoch": "123.0"})
        assert _parse_tshark_line(line) is None

    def test_short_line_returns_none(self):
        assert _parse_tshark_line("192.168.1.1|10.0.0.1") is None

    def test_malformed_timestamp_returns_none(self):
        line = self._make_line({
            "frame.time_epoch": "not_a_number",
            "ip.src": "1.1.1.1",
            "ip.dst": "2.2.2.2",
            "ip.proto": "6",
        })
        # Should not raise; invalid timestamp → float("not_a_number") → ValueError
        result = _parse_tshark_line(line)
        assert result is None


# ---------------------------------------------------------------------------
# WindowManager tests
# ---------------------------------------------------------------------------

class TestWindowManager:
    """Verify sliding window logic without real time passing."""

    @pytest.mark.asyncio
    async def test_empty_window_not_emitted_before_window_size(self):
        wm = WindowManager(window_size_seconds=10.0, step_seconds=5.0)
        # Only 3 seconds have passed — window not yet ready
        await wm.tick(current_time=wm._start_time + 3.0)
        assert wm.windows_emitted == 0

    @pytest.mark.asyncio
    async def test_first_window_emitted_after_window_size(self):
        wm = WindowManager(window_size_seconds=10.0, step_seconds=5.0)
        # Create a flow whose timestamps fall within the first window
        from tabpfn_nids.flows.flow_builder import FlowRecord
        flow = FlowRecord(
            flow_id="test-flow",
            src_ip="1.1.1.1",
            dst_ip="2.2.2.2",
            src_port=1234,
            dst_port=80,
            protocol=6,
            protocol_name="tcp",
            start_time=wm._start_time + 1.0,
            end_time=wm._start_time + 5.0,
            duration=4.0,
            fwd_packets=3,
            bwd_packets=2,
            total_packets=5,
            total_bytes=500,
        )
        wm.add_flows([flow])
        # Advance past window_size
        await wm.tick(current_time=wm._start_time + 11.0)
        assert wm.windows_emitted == 1
        job = wm.job_queue.get_nowait()
        assert job.flow_count == 1
        assert job.window_id == "window-00000"

    @pytest.mark.asyncio
    async def test_multiple_windows_emitted(self):
        wm = WindowManager(window_size_seconds=10.0, step_seconds=5.0)
        from tabpfn_nids.flows.flow_builder import FlowRecord
        flow = FlowRecord(
            flow_id="multi-flow", src_ip="1.1.1.1", dst_ip="2.2.2.2",
            src_port=1234, dst_port=80, protocol=6, protocol_name="tcp",
            start_time=wm._start_time + 1.0, end_time=wm._start_time + 8.0,
            duration=7.0, total_packets=5, total_bytes=500,
        )
        wm.add_flows([flow])
        # At t=21s: windows at [0,10], [5,15], [10,20] should be ready
        await wm.tick(current_time=wm._start_time + 21.0)
        assert wm.windows_emitted == 3

    @pytest.mark.asyncio
    async def test_drop_when_queue_full(self, sample_flow_record):
        """Verify windows are dropped (not silently lost) when queue is full."""
        wm = WindowManager(
            window_size_seconds=10.0, step_seconds=5.0, max_job_queue_size=1
        )
        wm.add_flows([sample_flow_record])
        # Advance far enough for multiple windows
        await wm.tick(current_time=wm._start_time + 31.0)
        assert wm.windows_dropped > 0

    @pytest.mark.asyncio
    async def test_old_flows_pruned(self):
        """Very old flows should be removed from the buffer."""
        wm = WindowManager(window_size_seconds=10.0, step_seconds=5.0)
        from tabpfn_nids.flows.flow_builder import FlowRecord
        old_flow = FlowRecord(
            flow_id="old-flow", src_ip="1.1.1.1", dst_ip="2.2.2.2",
            src_port=1234, dst_port=80, protocol=6, protocol_name="tcp",
            start_time=wm._start_time - 100.0, end_time=wm._start_time - 50.0,
            duration=50.0, total_packets=5, total_bytes=500,
        )
        wm.add_flows([old_flow])
        await wm.tick(current_time=wm._start_time + 31.0)
        assert len(wm._all_flows) == 0

    def test_detection_job_dict(self, sample_flow_record):
        job = DetectionJob(
            window_id="test-00001",
            window_start=1000.0,
            window_end=1010.0,
            flows=[sample_flow_record],
        )
        assert job.flow_count == 1
        assert job.packet_count == sample_flow_record.total_packets


# ---------------------------------------------------------------------------
# ResultBus tests
# ---------------------------------------------------------------------------

class TestResultBus:
    @pytest.mark.asyncio
    async def test_publish_and_receive(self, sample_prediction):
        bus = ResultBus()
        q = bus.subscribe(maxsize=10)
        await bus.publish(sample_prediction)
        result = q.get_nowait()
        assert result.window_id == sample_prediction.window_id
        assert result.prediction == "ATTACK"

    @pytest.mark.asyncio
    async def test_attack_counter_increments(self, sample_prediction):
        bus = ResultBus()
        _ = bus.subscribe()
        assert bus.attack_count == 0
        await bus.publish(sample_prediction)
        assert bus.attack_count == 1

    @pytest.mark.asyncio
    async def test_benign_does_not_increment_attack_count(self):
        bus = ResultBus()
        _ = bus.subscribe()
        benign = PredictionResult(
            window_id="w1", prediction="BENIGN", confidence=0.9,
            attack_probability=0.1,
        )
        await bus.publish(benign)
        assert bus.attack_count == 0
        assert bus.total_count == 1

    @pytest.mark.asyncio
    async def test_history_bounded(self, sample_prediction):
        bus = ResultBus()
        for i in range(150):
            p = PredictionResult(
                window_id=f"w{i}", prediction="BENIGN",
                confidence=0.9, attack_probability=0.1,
            )
            await bus.publish(p)
        assert len(bus.get_history()) == 100  # _HISTORY_SIZE

    @pytest.mark.asyncio
    async def test_unsubscribe(self, sample_prediction):
        bus = ResultBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        await bus.publish(sample_prediction)
        assert bus.subscriber_count == 0

    def test_prediction_result_to_dict(self, sample_prediction):
        d = sample_prediction.to_dict()
        assert "window_id" in d
        assert "prediction" in d
        assert "attack_probability" in d
        assert isinstance(d["confidence"], float)


# ---------------------------------------------------------------------------
# LivePreprocessor tests (artifact-free)
# ---------------------------------------------------------------------------

class TestLivePreprocessor:
    """Test LivePreprocessor with mock artifacts."""

    @pytest.fixture
    def mock_artifacts(self, tmp_path):
        """Create minimal fake artifacts."""
        from sklearn.preprocessing import RobustScaler
        import pickle

        # Feature names (subset for testing)
        feature_names = [
            "duration", "total_packets", "total_bytes",
            "packets_per_second", "bytes_per_second",
            "fwd_packets", "bwd_packets",
        ]
        schema = {"feature_names": feature_names}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))

        # Fit a scaler on random data with matching feature count
        rng = np.random.default_rng(42)
        scaler = RobustScaler()
        scaler.fit(rng.random((50, len(feature_names))))
        scaler_path = tmp_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        return schema_path, scaler_path, feature_names

    def test_load_missing_schema_raises(self, tmp_path):
        from tabpfn_nids.live.live_preprocessor import ArtifactLoadError, LivePreprocessor
        pp = LivePreprocessor(
            scaler_path=tmp_path / "nonexistent.pkl",
            feature_schema_path=tmp_path / "nonexistent.json",
        )
        with pytest.raises(ArtifactLoadError, match="Feature schema"):
            pp.load()

    def test_load_missing_scaler_raises(self, tmp_path, mock_artifacts):
        from tabpfn_nids.live.live_preprocessor import ArtifactLoadError, LivePreprocessor
        schema_path, _, _ = mock_artifacts
        pp = LivePreprocessor(
            scaler_path=tmp_path / "nonexistent.pkl",
            feature_schema_path=schema_path,
        )
        with pytest.raises(ArtifactLoadError, match="scaler"):
            pp.load()

    def test_transform_without_load_raises(self, tmp_path, mock_artifacts):
        from tabpfn_nids.live.live_preprocessor import LivePreprocessor
        schema_path, scaler_path, _ = mock_artifacts
        pp = LivePreprocessor(scaler_path=scaler_path, feature_schema_path=schema_path)
        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            pp.transform([])

    def test_transform_empty_flows_raises(self, mock_artifacts):
        from tabpfn_nids.live.live_preprocessor import LivePreprocessor
        schema_path, scaler_path, _ = mock_artifacts
        pp = LivePreprocessor(scaler_path=scaler_path, feature_schema_path=schema_path)
        pp.load()
        with pytest.raises(ValueError, match="No flows"):
            pp.transform([])
