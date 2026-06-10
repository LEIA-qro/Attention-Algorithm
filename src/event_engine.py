"""Merge per-frame signals and dispatch event triggers with sustain/cooldown/re-trigger logic."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ALL_EVENT_TYPES = ["phone", "food", "danger", "drowsy", "distracted", "eyes_off"]


@dataclass
class SignalFrame:
    """Aggregated detector and model signals for one video frame."""
    phone: float
    food: float
    danger: float
    drowsy_prob: float
    distracted_prob: float
    alert_prob: float
    eyes_off_road_pct: float
    timestamp_s: float


class EventEngine:
    """Sustain, cooldown, and re-trigger gating over the merged signals."""

    def __init__(self, cfg: Dict[str, Any], fps: float = 30.0) -> None:
        # cfg is the events: section of yolo_config.yaml; fps is kept for logging only.
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
        """Process one frame and return any fired triggers as dicts."""
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
