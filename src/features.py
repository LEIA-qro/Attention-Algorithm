"""
features.py — Per-Frame + Temporal Feature Extraction for DMS
==============================================================

The :class:`FeatureExtractor` wraps MediaPipe Face Landmarker v2
(478 landmarks, blendshapes, and 3D facial transformation matrix) to
produce an 18-dimensional feature vector per frame:

===  ========================  =====================================
 #   Feature                   Description
===  ========================  =====================================
 0   ear_left                  Left-eye EAR (calibration-normalised)
 1   ear_right                 Right-eye EAR (calibration-normalised)
 2   ear_avg                   Mean of left & right EAR
 3   mar                       Mouth Aspect Ratio
 4   perclos                   P80 PERCLOS over 60 s window
 5   blink_rate                Blinks per minute (60 s window)
 6   blink_duration_avg        Mean blink length in seconds (60 s)
 7   yaw                       Head yaw   (degrees, from 4×4 matrix)
 8   pitch                     Head pitch (degrees)
 9   roll                      Head roll  (degrees)
10   gaze_yaw                  Horizontal gaze angle (degrees)
11   gaze_pitch                Vertical gaze angle (degrees)
12   gaze_stability            Gaze-angle std dev over 1 s
13   head_pose_stability       Head-pose std dev over 1 s
14   ear_velocity              Δ(EAR) / Δt  (frames⁻¹)
15   head_nod_count            Pitch dips >15° in last 10 s
16   mouth_open_duration       Consecutive frames with MAR > thresh
17   eyes_off_road_pct         % time gaze >30° from centre (5 s)
===  ========================  =====================================

Public API
----------
- ``FeatureExtractor(model_path, fps, cfg)``
- ``FeatureExtractor.extract(frame, timestamp_ms) -> dict[str, float]``
- ``FeatureExtractor.reset()``
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

__all__ = ["FeatureExtractor"]

logger = logging.getLogger(__name__)

# MediaPipe type aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmark indices (478-point model)
# EAR landmarks — p1..p6 per the standard Eye Aspect Ratio formula
LEFT_EYE_IDX: List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX: List[int] = [33, 160, 158, 133, 153, 144]

# MAR landmarks — outer lip contour for more precise Mouth Aspect Ratio
UPPER_LIP_IDX: List[int] = [82, 13, 312]
LOWER_LIP_IDX: List[int] = [87, 14, 317]
LEFT_MOUTH_IDX: int = 78
RIGHT_MOUTH_IDX: int = 308

# Iris centres
LEFT_IRIS_CENTER: int = 468
RIGHT_IRIS_CENTER: int = 473

# Eye corners (for gaze computation)
LEFT_EYE_INNER: int = 362
LEFT_EYE_OUTER: int = 263
RIGHT_EYE_INNER: int = 133
RIGHT_EYE_OUTER: int = 33

# Feature names in canonical order
FEATURE_NAMES: List[str] = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "perclos",
    "blink_rate",
    "blink_duration_avg",
    "yaw",
    "pitch",
    "roll",
    "gaze_yaw",
    "gaze_pitch",
    "gaze_stability",
    "head_pose_stability",
    "ear_velocity",
    "head_nod_count",
    "mouth_open_duration",
    "eyes_off_road_pct",
]

assert len(FEATURE_NAMES) == 18, "Expected 18 features"


# Helper geometry

def _lm_to_np(landmark) -> np.ndarray:
    """Convert a single MediaPipe NormalizedLandmark to (x, y, z)."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def _compute_ear(landmarks, eye_indices: List[int]) -> float:
    """Eye Aspect Ratio (Soukupová & Čech 2016).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    landmarks : list of NormalizedLandmark
        Full 478-landmark list from MediaPipe.
    eye_indices : list of int
        Six indices [p1, p2, p3, p4, p5, p6].

    Returns
    -------
    float
        Eye Aspect Ratio ∈ [0, ~0.5].
    """
    pts = [_lm_to_np(landmarks[i]) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    numerator = _euclidean(p2, p6) + _euclidean(p3, p5)
    denominator = 2.0 * _euclidean(p1, p4)
    if denominator < 1e-8:
        return 0.0
    return numerator / denominator


def _compute_mar(landmarks) -> float:
    """Mouth Aspect Ratio using outer lip landmarks.

    MAR = (Σ ||upper_i - lower_i||) / (3 * ||left_corner - right_corner||)
    """
    upper = [_lm_to_np(landmarks[i]) for i in UPPER_LIP_IDX]
    lower = [_lm_to_np(landmarks[i]) for i in LOWER_LIP_IDX]
    left_corner = _lm_to_np(landmarks[LEFT_MOUTH_IDX])
    right_corner = _lm_to_np(landmarks[RIGHT_MOUTH_IDX])

    vert_sum = sum(_euclidean(u, l) for u, l in zip(upper, lower))
    horiz = _euclidean(left_corner, right_corner)
    if horiz < 1e-8:
        return 0.0
    return vert_sum / (3.0 * horiz)


def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to Euler angles (yaw, pitch, roll).

    Uses the ZYX (Tait–Bryan) convention.  Returns degrees.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def _compute_gaze_angles(
    landmarks,
) -> Tuple[float, float]:
    """Estimate gaze direction from iris position relative to eye corners.

    For each eye, the iris centre is projected onto the axis formed by the
    inner and outer corners.  The ratio gives a horizontal displacement;
    vertical displacement is computed perpendicular to that axis.  The two
    eyes are averaged and converted to approximate degrees.

    Returns
    -------
    gaze_yaw, gaze_pitch : float
        Gaze deviation in degrees from the straight-ahead direction.
    """
    # Left eye
    l_iris = _lm_to_np(landmarks[LEFT_IRIS_CENTER])[:2]
    l_inner = _lm_to_np(landmarks[LEFT_EYE_INNER])[:2]
    l_outer = _lm_to_np(landmarks[LEFT_EYE_OUTER])[:2]

    # Right eye
    r_iris = _lm_to_np(landmarks[RIGHT_IRIS_CENTER])[:2]
    r_inner = _lm_to_np(landmarks[RIGHT_EYE_INNER])[:2]
    r_outer = _lm_to_np(landmarks[RIGHT_EYE_OUTER])[:2]

    def _eye_gaze(iris: np.ndarray, inner: np.ndarray, outer: np.ndarray):
        eye_vec = outer - inner
        eye_len = np.linalg.norm(eye_vec)
        if eye_len < 1e-8:
            return 0.0, 0.0
        eye_dir = eye_vec / eye_len
        iris_rel = iris - inner
        # Horizontal ratio: 0 = inner corner, 1 = outer corner
        # Project iris onto the eye axis, normalise by eye width
        horiz_ratio = float(np.dot(iris_rel, eye_dir)) / eye_len
        # Vertical ratio (perpendicular component)
        perp = np.array([-eye_dir[1], eye_dir[0]])
        vert_ratio = float(np.dot(iris_rel, perp)) / eye_len
        return horiz_ratio, vert_ratio

    l_h, l_v = _eye_gaze(l_iris, l_inner, l_outer)
    r_h, r_v = _eye_gaze(r_iris, r_inner, r_outer)

    # Average and convert to approximate degrees
    # Typical eye opening spans ~30° of visual field, so we scale the
    # normalised displacement accordingly (tunable).
    scale_h = 60.0  # degrees per full eye-width displacement
    scale_v = 40.0

    avg_h = (l_h + r_h) / 2.0
    avg_v = (l_v + r_v) / 2.0

    # Centre: iris at midpoint of inner–outer → ratio ~ 0.5
    gaze_yaw = float(np.clip((avg_h - 0.5) * scale_h, -90.0, 90.0))
    gaze_pitch = float(np.clip(avg_v * scale_v, -90.0, 90.0))

    return float(gaze_yaw), float(gaze_pitch)


# Calibration data container

@dataclass
class _EARCalibration:
    """Stores EAR baseline collected during the first N frames."""

    target_frames: int = 894  # ~30 s at 29.76 fps
    collected: List[float] = field(default_factory=list)
    baseline: Optional[float] = None
    is_calibrated: bool = False

    def update(self, ear_avg: float) -> None:
        """Add a frame's EAR to calibration buffer."""
        if self.is_calibrated:
            return
        self.collected.append(ear_avg)
        if len(self.collected) >= self.target_frames:
            self.baseline = float(np.percentile(self.collected, 50))
            self.is_calibrated = True
            logger.info(
                "EAR calibration complete: baseline=%.4f from %d frames",
                self.baseline,
                len(self.collected),
            )

    def normalise(self, ear: float) -> float:
        """Normalise EAR relative to calibrated baseline."""
        if not self.is_calibrated or self.baseline is None or self.baseline < 1e-6:
            return ear
        return ear / self.baseline

    def reset(self) -> None:
        self.collected.clear()
        self.baseline = None
        self.is_calibrated = False


# Blink state machine

@dataclass
class _BlinkTracker:
    """Detects blinks from EAR time-series and tracks durations."""

    threshold: float = 0.21
    min_frames: int = 2
    max_frames: int = 15

    # Internal state
    _in_blink: bool = field(default=False, init=False)
    _blink_frame_count: int = field(default=0, init=False)
    _blink_durations: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5000), init=False
    )
    _blink_timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5000), init=False
    )

    def update(self, ear_avg: float, timestamp_s: float, fps: float) -> None:
        """Process one frame of EAR."""
        if ear_avg < self.threshold:
            if not self._in_blink:
                self._in_blink = True
                self._blink_frame_count = 1
            else:
                self._blink_frame_count += 1
        else:
            if self._in_blink:
                # Blink just ended
                if self.min_frames <= self._blink_frame_count <= self.max_frames:
                    duration_s = self._blink_frame_count / fps
                    self._blink_durations.append(duration_s)
                    self._blink_timestamps.append(timestamp_s)
                self._in_blink = False
                self._blink_frame_count = 0

    def blink_rate(self, current_time_s: float, window_s: float = 60.0) -> float:
        """Blinks per minute within the last ``window_s`` seconds."""
        cutoff = current_time_s - window_s
        count = sum(1 for t in self._blink_timestamps if t >= cutoff)
        # Scale to per minute
        effective_window = min(window_s, current_time_s) if current_time_s > 0 else 1.0
        if effective_window < 1e-3:
            return 0.0
        return count * 60.0 / effective_window

    def avg_duration(self, current_time_s: float, window_s: float = 60.0) -> float:
        """Mean blink duration (seconds) within the last ``window_s``."""
        cutoff = current_time_s - window_s
        recent = [
            d
            for d, t in zip(self._blink_durations, self._blink_timestamps)
            if t >= cutoff
        ]
        if not recent:
            return 0.0
        return float(np.mean(recent))

    def reset(self) -> None:
        self._in_blink = False
        self._blink_frame_count = 0
        self._blink_durations.clear()
        self._blink_timestamps.clear()


# Main Feature Extractor

class FeatureExtractor:
    """Extracts an 18-D feature vector from each video frame using
    MediaPipe Face Landmarker v2.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``.task`` model file for MediaPipe Face Landmarker.
    fps : float
        Video frame rate (default 29.76 for DMD).
    cfg : dict, optional
        Override dict whose keys mirror the ``features`` section of
        ``config.yaml``.  Missing keys fall back to built-in defaults.

    Example
    -------
    >>> ext = FeatureExtractor("models/face_landmarker_v2.task")
    >>> cap = cv2.VideoCapture("video.mp4")
    >>> ok, frame = cap.read()
    >>> feats = ext.extract(frame, timestamp_ms=0)
    >>> assert len(feats) == 18
    """

    def __init__(
        self,
        model_path: str | Path,
        fps: float = 29.76,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fps = fps
        self._cfg = cfg or {}
        self._frame_count: int = 0

        # Unpack config with defaults
        ear_cfg = self._cfg.get("ear", {})
        mar_cfg = self._cfg.get("mar", {})
        perclos_cfg = self._cfg.get("perclos", {})
        blink_cfg = self._cfg.get("blink", {})
        gaze_cfg = self._cfg.get("gaze", {})
        head_cfg = self._cfg.get("head_pose", {})
        smooth_cfg = self._cfg.get("smoothing", {})

        self._ear_threshold: float = ear_cfg.get("blink_threshold", 0.21)
        self._mar_threshold: float = mar_cfg.get("yawn_threshold", 0.6)
        self._perclos_window_s: float = perclos_cfg.get("window_seconds", 60.0)
        self._gaze_stability_window_s: float = gaze_cfg.get(
            "stability_window_seconds", 1.0
        )
        self._gaze_offroad_thresh: float = gaze_cfg.get(
            "off_road_threshold_deg", 45.0
        )
        self._gaze_offroad_window_s: float = gaze_cfg.get(
            "off_road_window_seconds", 5.0
        )
        self._head_stability_window_s: float = head_cfg.get(
            "stability_window_seconds", 1.0
        )
        self._nod_thresh: float = head_cfg.get("nod_threshold_deg", 15.0)
        self._nod_window_s: float = head_cfg.get("nod_window_seconds", 10.0)
        self._ear_vel_window: int = smooth_cfg.get("ear_velocity_window", 3)

        # Rolling buffers (maxlen = max needed window in frames) ─
        max_window_frames = int(
            max(
                self._perclos_window_s,
                self._gaze_offroad_window_s,
                self._nod_window_s,
                60.0,  # blink rate window
            )
            * self.fps
            + 10
        )
        self._ear_history: Deque[float] = deque(maxlen=max_window_frames)
        self._ear_ts: Deque[float] = deque(maxlen=max_window_frames)
        self._gaze_yaw_history: Deque[float] = deque(maxlen=max_window_frames)
        self._gaze_pitch_history: Deque[float] = deque(maxlen=max_window_frames)
        self._gaze_ts: Deque[float] = deque(maxlen=max_window_frames)
        self._head_yaw_history: Deque[float] = deque(maxlen=max_window_frames)
        self._head_pitch_history: Deque[float] = deque(maxlen=max_window_frames)
        self._head_roll_history: Deque[float] = deque(maxlen=max_window_frames)
        self._head_ts: Deque[float] = deque(maxlen=max_window_frames)
        self._eye_closed_flags: Deque[bool] = deque(maxlen=max_window_frames)
        self._eye_closed_ts: Deque[float] = deque(maxlen=max_window_frames)

        # Mouth open duration counter
        self._mouth_open_frames: int = 0

        # Head nod detector state
        self._pitch_prev: Optional[float] = None
        self._in_nod: bool = False
        self._nod_timestamps: Deque[float] = deque(maxlen=500)

        # EAR calibration
        calib_frames = ear_cfg.get("calibration_frames", 894)
        self._calibration = _EARCalibration(target_frames=calib_frames)

        # Blink tracker
        self._blink_tracker = _BlinkTracker(
            threshold=self._ear_threshold,
            min_frames=blink_cfg.get("min_duration_frames", 2),
            max_frames=blink_cfg.get("max_duration_frames", 15),
        )

        # Initialise MediaPipe FaceLandmarker
        model_path = str(Path(model_path).resolve())
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        logger.info(
            "FeatureExtractor initialised: model=%s, fps=%.2f", model_path, fps
        )

    # Public API

    def extract(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> Dict[str, float]:
        """Extract the 18-D feature vector from one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image (H, W, 3).
        timestamp_ms : int
            Frame timestamp in milliseconds (must be monotonically
            increasing across successive calls).

        Returns
        -------
        dict[str, float]
            Dictionary with all 18 features keyed by ``FEATURE_NAMES``.
            If no face is detected, all values are ``0.0`` (except head
            pose and gaze which remain at their last known value).
        """
        timestamp_s = timestamp_ms / 1000.0
        self._frame_count += 1

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result: FaceLandmarkerResult = self._landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        # Default feature dict (all zeros)
        feats = {name: 0.0 for name in FEATURE_NAMES}

        if not result.face_landmarks:
            logger.debug("No face detected at t=%.3f s", timestamp_s)
            # If the face is completely lost, the driver is almost certainly looking completely away.
            # We output extreme yaw and pitch so the model immediately triggers Distracted.
            # CRITICAL: We must simulate OPEN eyes, otherwise the model thinks EAR=0.0 (eyes closed)
            # and will strongly predict Drowsy instead!
            # EAR must be 1.0 (calibrated normal), not 0.3!
            feats["yaw"] = 90.0
            feats["pitch"] = 45.0
            feats["eyes_off_road_pct"] = 1.0
            feats["gaze_yaw"] = 90.0
            feats["ear_avg"] = 1.0
            feats["ear_left"] = 1.0
            feats["ear_right"] = 1.0
            feats["perclos"] = 0.0
            return feats

        landmarks = result.face_landmarks[0]

        # Per-frame geometric features
        ear_left_raw = _compute_ear(landmarks, LEFT_EYE_IDX)
        ear_right_raw = _compute_ear(landmarks, RIGHT_EYE_IDX)
        ear_avg_raw = (ear_left_raw + ear_right_raw) / 2.0
        mar = _compute_mar(landmarks)

        # Calibration
        self._calibration.update(ear_avg_raw)
        ear_left = self._calibration.normalise(ear_left_raw)
        ear_right = self._calibration.normalise(ear_right_raw)
        ear_avg = self._calibration.normalise(ear_avg_raw)

        feats["ear_left"] = ear_left
        feats["ear_right"] = ear_right
        feats["ear_avg"] = ear_avg
        feats["mar"] = mar

        # Head pose from 4×4 transformation matrix
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if result.facial_transformation_matrixes:
            mat_4x4 = np.array(
                result.facial_transformation_matrixes[0],
                dtype=np.float64,
            ).reshape(4, 4)
            R = mat_4x4[:3, :3]
            yaw, pitch, roll = _rotation_matrix_to_euler(R)

        feats["yaw"] = yaw
        feats["pitch"] = pitch
        feats["roll"] = roll

        # Gaze direction from iris landmarks
        gaze_yaw, gaze_pitch = _compute_gaze_angles(landmarks)
        feats["gaze_yaw"] = gaze_yaw
        feats["gaze_pitch"] = gaze_pitch

        # Update rolling buffers
        self._ear_history.append(ear_avg)
        self._ear_ts.append(timestamp_s)

        eye_closed = ear_avg_raw < self._ear_threshold
        self._eye_closed_flags.append(eye_closed)
        self._eye_closed_ts.append(timestamp_s)

        self._gaze_yaw_history.append(gaze_yaw)
        self._gaze_pitch_history.append(gaze_pitch)
        self._gaze_ts.append(timestamp_s)

        self._head_yaw_history.append(yaw)
        self._head_pitch_history.append(pitch)
        self._head_roll_history.append(roll)
        self._head_ts.append(timestamp_s)

        self._blink_tracker.update(ear_avg_raw, timestamp_s, self.fps)

        # Temporal features

        # PERCLOS (P80): fraction of time eyes are closed over window
        feats["perclos"] = self._compute_perclos(timestamp_s)

        # Blink rate & average duration
        feats["blink_rate"] = self._blink_tracker.blink_rate(
            timestamp_s, window_s=60.0
        )
        feats["blink_duration_avg"] = self._blink_tracker.avg_duration(
            timestamp_s, window_s=60.0
        )

        # Gaze stability (std dev of gaze angle over 1 s)
        feats["gaze_stability"] = self._compute_gaze_stability(timestamp_s)

        # Head pose stability (std dev over 1 s)
        feats["head_pose_stability"] = self._compute_head_stability(timestamp_s)

        # EAR velocity (Δ EAR / Δ frames)
        feats["ear_velocity"] = self._compute_ear_velocity()

        # Head nod count (pitch dips in last 10 s)
        feats["head_nod_count"] = self._compute_head_nod_count(
            pitch, timestamp_s
        )

        # Mouth open duration (consecutive frames)
        if mar > self._mar_threshold:
            self._mouth_open_frames += 1
        else:
            self._mouth_open_frames = 0
        feats["mouth_open_duration"] = float(self._mouth_open_frames)

        # Eyes off road percentage (last 5 s)
        feats["eyes_off_road_pct"] = self._compute_eyes_off_road(timestamp_s)

        return feats

    def reset(self) -> None:
        """Reset all internal state (call between videos / subjects)."""
        self._frame_count = 0
        self._ear_history.clear()
        self._ear_ts.clear()
        self._eye_closed_flags.clear()
        self._eye_closed_ts.clear()
        self._gaze_yaw_history.clear()
        self._gaze_pitch_history.clear()
        self._gaze_ts.clear()
        self._head_yaw_history.clear()
        self._head_pitch_history.clear()
        self._head_roll_history.clear()
        self._head_ts.clear()
        self._mouth_open_frames = 0
        self._pitch_prev = None
        self._in_nod = False
        self._nod_timestamps.clear()
        self._calibration.reset()
        self._blink_tracker.reset()
        logger.info("FeatureExtractor state reset")

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("FeatureExtractor closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # Private temporal computation helpers

    def _compute_perclos(self, current_s: float) -> float:
        """PERCLOS P80: fraction of time eyes ≥80% closed in window."""
        cutoff = current_s - self._perclos_window_s
        closed_in_window = [
            c for c, t in zip(self._eye_closed_flags, self._eye_closed_ts)
            if t >= cutoff
        ]
        if not closed_in_window:
            return 0.0
        return sum(closed_in_window) / len(closed_in_window)

    def _compute_gaze_stability(self, current_s: float) -> float:
        """Std dev of gaze angle magnitude over a 1 s window."""
        cutoff = current_s - self._gaze_stability_window_s
        yaws = []
        pitches = []
        for y, p, t in zip(
            self._gaze_yaw_history,
            self._gaze_pitch_history,
            self._gaze_ts,
        ):
            if t >= cutoff:
                yaws.append(y)
                pitches.append(p)
        if len(yaws) < 2:
            return 0.0
        # Combine yaw and pitch into magnitude
        magnitudes = [math.sqrt(y ** 2 + p ** 2) for y, p in zip(yaws, pitches)]
        return float(np.std(magnitudes))

    def _compute_head_stability(self, current_s: float) -> float:
        """Std dev of head pose magnitude (yaw²+pitch²+roll²)^½ over 1 s."""
        cutoff = current_s - self._head_stability_window_s
        poses = []
        for y, p, r, t in zip(
            self._head_yaw_history,
            self._head_pitch_history,
            self._head_roll_history,
            self._head_ts,
        ):
            if t >= cutoff:
                poses.append(math.sqrt(y ** 2 + p ** 2 + r ** 2))
        if len(poses) < 2:
            return 0.0
        return float(np.std(poses))

    def _compute_ear_velocity(self) -> float:
        """Central-difference EAR velocity: Δ(ear) / Δ(frames)."""
        w = self._ear_vel_window
        if len(self._ear_history) < 2 * w + 1:
            return 0.0
        recent = list(self._ear_history)
        # Central difference over ±w frames
        current = recent[-1]
        past = recent[-(2 * w + 1)]
        return (current - past) / (2.0 * w)

    def _compute_head_nod_count(
        self, pitch: float, current_s: float
    ) -> float:
        """Count pitch dips exceeding threshold in last N seconds.

        A "nod" is detected when pitch drops below –threshold and then
        returns above it.
        """
        # Detect transitions
        if self._pitch_prev is not None:
            was_above = self._pitch_prev > -self._nod_thresh
            is_below = pitch <= -self._nod_thresh
            if was_above and is_below:
                self._in_nod = True
            elif self._in_nod and pitch > -self._nod_thresh:
                # Nod completed (went below and came back up)
                self._nod_timestamps.append(current_s)
                self._in_nod = False
        self._pitch_prev = pitch

        # Count nods within window
        cutoff = current_s - self._nod_window_s
        count = sum(1 for t in self._nod_timestamps if t >= cutoff)
        return float(count)

    def _compute_eyes_off_road(self, current_s: float) -> float:
        """Percentage of time gaze is >threshold degrees from centre."""
        cutoff = current_s - self._gaze_offroad_window_s
        total = 0
        off_road = 0
        for y, p, t in zip(
            self._gaze_yaw_history,
            self._gaze_pitch_history,
            self._gaze_ts,
        ):
            if t >= cutoff:
                total += 1
                angle = math.sqrt(y ** 2 + p ** 2)
                if angle > self._gaze_offroad_thresh:
                    off_road += 1
        if total == 0:
            return 0.0
        return off_road / total

    @staticmethod
    def feature_names() -> List[str]:
        """Return the canonical ordered list of 18 feature names."""
        return list(FEATURE_NAMES)
