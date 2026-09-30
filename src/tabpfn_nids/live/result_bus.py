"""Result bus: prediction event publication and subscription.

All completed inference results are published here. WebSocket handlers
and REST endpoints subscribe to receive updates without polling.

Design:
- asyncio.Queue per subscriber (no single-point-of-failure broadcast).
- Recent results are buffered for late-joining subscribers.
- Attack counter is incremented here (single authoritative counter).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_HISTORY_SIZE = 100   # Number of recent predictions to retain in memory


@dataclass
class PredictionResult:
    """A single window-level detection result.

    Fields not provided by the model (e.g., packets_analyzed) are set
    by the inference manager after the model call completes.
    """
    window_id: str
    prediction: str              # "BENIGN" or "ATTACK"
    confidence: float            # 0.0 – 1.0 (P(predicted class))
    attack_probability: float    # Always P(ATTACK); useful for dashboards
    flows_analyzed: int = 0
    packets_analyzed: int = 0
    inference_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict for JSON / WebSocket transmission."""
        from datetime import datetime, timezone
        ts_iso = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()
        return {
            "window_id": self.window_id,
            "timestamp": ts_iso,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "attack_probability": round(self.attack_probability, 4),
            "flows_analyzed": self.flows_analyzed,
            "packets_analyzed": self.packets_analyzed,
            "inference_seconds": round(self.inference_seconds, 3),
        }


class ResultBus:
    """Pub-sub event bus for PredictionResult objects.

    Subscribers receive a shallow copy of every result published after
    they register. The bus retains the last _HISTORY_SIZE results so
    newly connected WebSocket clients can catch up.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[PredictionResult]] = []
        self._history: list[PredictionResult] = []
        self._attack_count: int = 0
        self._total_count: int = 0
        self._lock = asyncio.Lock()

    async def publish(self, result: PredictionResult) -> None:
        """Publish a prediction result to all subscribers.

        Args:
            result: The completed PredictionResult.
        """
        async with self._lock:
            self._total_count += 1
            if result.prediction == "ATTACK":
                self._attack_count += 1

            # Buffer for history
            self._history.append(result)
            if len(self._history) > _HISTORY_SIZE:
                self._history = self._history[-_HISTORY_SIZE:]

        # Broadcast to all subscribers (non-blocking; slow subscribers
        # will have their queues fill up and start dropping old items)
        dead: list[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                # Subscriber is too slow — drop oldest item and try again
                try:
                    q.get_nowait()
                    q.put_nowait(result)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:
                dead.append(q)

        for q in dead:
            self._subscribers.remove(q)

    def subscribe(self, maxsize: int = 50) -> asyncio.Queue[PredictionResult]:
        """Register a new subscriber queue.

        Args:
            maxsize: Maximum buffered results for this subscriber.

        Returns:
            asyncio.Queue that receives future results.
        """
        q: asyncio.Queue[PredictionResult] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def get_history(self) -> list[PredictionResult]:
        """Return recent prediction history (newest last)."""
        return list(self._history)

    @property
    def attack_count(self) -> int:
        """Total attack detections since capture started."""
        return self._attack_count

    @property
    def total_count(self) -> int:
        """Total windows processed since capture started."""
        return self._total_count

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)
