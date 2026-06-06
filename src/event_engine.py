"""
event_engine.py — Signal Merger and Trigger Dispatcher
=======================================================

Public API
----------
- ``SignalFrame`` — dataclass, one per frame
- ``EventEngine(cfg, fps)``
    - ``.update(signal_frame) -> list[dict]``
    - ``.reset()``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ["SignalFrame", "EventEngine"]

logger = logging.getLogger(__name__)

ALL_EVENT_TYPES = ["phone", "food", "danger", "drowsy", "distracted", "eyes_off"]


@dataclass
class SignalFrame:
    """Aggregated signals for one video frame.

    phone, food, danger  : max YOLO confidence in driver ROI (0-1, 0=not detected)
    drowsy_prob          : DriverStateNet softmax for class 1 (Drowsy)
    distracted_prob      : DriverStateNet softmax for class 2 (Distracted)
    alert_prob           : DriverStateNet softmax for class 0 (Alert)
    eyes_off_road_pct    : value of eyes_off_road_pct feature (0-1)
    timestamp_s          : seconds from session start
    """
    phone: float
    food: float
    danger: float
    drowsy_prob: float
    distracted_prob: float
    alert_prob: float
    eyes_off_road_pct: float
    timestamp_s: float


class EventEngine:
    """Sustain + cooldown + re-trigger logic.

    Parameters
    ----------
    cfg : dict — the ``events:`` section from yolo_config.yaml
    fps : float — video frame rate (used for logging only)
    """

    def __init__(self, cfg: Dict[str, Any], fps: float = 30.0) -> None:
        self._fps = fps
        s = cfg.get("sustain_seconds", {})
        self._sustain: Dict[str, float] = {evt: s.get(evt, 1.5) for evt in ALL_EVENT_TYPES}
        self._cooldown_s: float = cfg.get("cooldown_seconds", 10.0)
        self._retrigger_s: float = cfg.get("re_trigger_interval_seconds", 30.0)
        self._model_thresh: float = cfg.get("model_confidence_thresh", 0.55)
        self._eyes_thresh: float = cfg.get("eyes_off_road_thresh", 0.60)

        self._sustain_start: Dict[str, float] = {e: -1.0 for e in ALL_EVENT_TYPES}
        self._last_trigger_ts: Dict[str, float] = {e: -9999.0 for e in ALL_EVENT_TYPES}
        self._trigger_count: Dict[str, int] = {e: 0 for e in ALL_EVENT_TYPES}

    def update(self, frame: SignalFrame) -> List[Dict[str, Any]]:
        """Process one frame. Returns list of trigger dicts (may be empty).

        Each trigger dict has keys: event_type (str), confidence (float),
        timestamp_s (float).
        """
        active = self._active_signals(frame)
        triggers = []

        for evt in ALL_EVENT_TYPES:
            conf = active.get(evt, 0.0)

            if conf <= 0.0:
                self._sustain_start[evt] = -1.0
                continue

            if self._sustain_start[evt] < 0.0:
                self._sustain_start[evt] = frame.timestamp_s

            sustained = frame.timestamp_s - self._sustain_start[evt]
            if sustained < self._sustain[evt]:
                continue

            time_since = frame.timestamp_s - self._last_trigger_ts[evt]
            if self._trigger_count[evt] > 0:
                if time_since < self._cooldown_s:
                    continue
                if time_since < self._retrigger_s:
                    continue

            self._last_trigger_ts[evt] = frame.timestamp_s
            self._trigger_count[evt] += 1
            triggers.append({
                "event_type": evt,
                "confidence": float(conf),
                "timestamp_s": float(frame.timestamp_s),
            })
            logger.info("TRIGGER: %s conf=%.2f t=%.1fs #%d",
                        evt, conf, frame.timestamp_s, self._trigger_count[evt])

        return triggers

    def reset(self) -> None:
        """Reset all counters and timers."""
        self._sustain_start = {e: -1.0 for e in ALL_EVENT_TYPES}
        self._last_trigger_ts = {e: -9999.0 for e in ALL_EVENT_TYPES}
        self._trigger_count = {e: 0 for e in ALL_EVENT_TYPES}

    def _active_signals(self, frame: SignalFrame) -> Dict[str, float]:
        return {
            "phone":      frame.phone,
            "food":       frame.food,
            "danger":     frame.danger,
            "drowsy":     frame.drowsy_prob if frame.drowsy_prob >= self._model_thresh else 0.0,
            "distracted": frame.distracted_prob if frame.distracted_prob >= self._model_thresh else 0.0,
            "eyes_off":   frame.eyes_off_road_pct if frame.eyes_off_road_pct >= self._eyes_thresh else 0.0,
        }
