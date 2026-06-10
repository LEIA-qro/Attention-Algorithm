"""feature extractor variant: absolute gaze plus face-loss handling."""

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from src.features import FeatureExtractor

logger = logging.getLogger(__name__)


class CustomFeatureExtractor(FeatureExtractor):
    """feature extractor using absolute gaze and default temporal values on face loss"""

    def __init__(
        self,
        model_path: str | Path,
        fps: float = 29.76,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_path, fps, cfg)
        self._last_result = None
        orig_detect = self._landmarker.detect_for_video

        def wrapped_detect(image, timestamp_ms, *args, **kwargs):
            res = orig_detect(image, timestamp_ms, *args, **kwargs)
            self._last_result = res
            return res

        self._landmarker.detect_for_video = wrapped_detect

    def _compute_eyes_off_road(self, current_s: float) -> float:
        """fraction of recent gaze samples past the off-road angle, using absolute gaze"""
        cutoff = current_s - self._gaze_offroad_window_s
        total = 0
        off_road = 0
        for y, p, h_y, h_p, t in zip(
            self._gaze_yaw_history,
            self._gaze_pitch_history,
            self._head_yaw_history,
            self._head_pitch_history,
            self._gaze_ts,
        ):
            if t >= cutoff:
                total += 1
                angle = math.sqrt((h_y + y) ** 2 + (h_p + p) ** 2)
                if angle > self._gaze_offroad_thresh:
                    off_road += 1
        if total == 0:
            return 0.0
        return off_road / total

    def extract(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> Dict[str, float]:
        """extract features, falling back to defaults when the face is lost"""
        feats = super().extract(frame, timestamp_ms)
        
        if self._last_result and not self._last_result.face_landmarks:
            timestamp_s = timestamp_ms / 1000.0
            
            # keep temporal windows filled on face loss
            self._ear_history.append(1.0)
            self._ear_ts.append(timestamp_s)
            self._eye_closed_flags.append(False)
            self._eye_closed_ts.append(timestamp_s)
            self._gaze_yaw_history.append(90.0)
            self._gaze_pitch_history.append(0.0)
            self._gaze_ts.append(timestamp_s)
            self._head_yaw_history.append(90.0)
            self._head_pitch_history.append(45.0)
            self._head_roll_history.append(0.0)
            self._head_ts.append(timestamp_s)
            
            # normal ear so the tracker logs no false blink
            self._blink_tracker.update(0.30, timestamp_s, self.fps)
            
            # reset mouth-open counter
            self._mouth_open_frames = 0
            
            # override per-frame feats
            feats["ear_left"] = 1.0
            feats["ear_right"] = 1.0
            feats["ear_avg"] = 1.0
            feats["mar"] = 0.0
            
            feats["yaw"] = 90.0
            feats["pitch"] = 45.0
            feats["roll"] = 0.0
            feats["gaze_yaw"] = 90.0
            feats["gaze_pitch"] = 0.0
            
            # temporal features
            feats["perclos"] = self._compute_perclos(timestamp_s)
            feats["blink_rate"] = self._blink_tracker.blink_rate(timestamp_s, window_s=60.0)
            feats["blink_duration_avg"] = self._blink_tracker.avg_duration(timestamp_s, window_s=60.0)
            feats["gaze_stability"] = self._compute_gaze_stability(timestamp_s)
            feats["head_pose_stability"] = self._compute_head_stability(timestamp_s)
            feats["ear_velocity"] = self._compute_ear_velocity()
            feats["head_nod_count"] = self._compute_head_nod_count(45.0, timestamp_s)
            feats["mouth_open_duration"] = 0.0
            feats["eyes_off_road_pct"] = self._compute_eyes_off_road(timestamp_s)
            
        return feats

