#!/usr/bin/env python3

"""Real-time driver monitoring demo: webcam -> features -> ONNX DriverStateNet, with an overlay HUD.

Usage:
    python scripts/inference_demo.py --config config/config.yaml
    python scripts/inference_demo.py --onnx models/driver_state_net.onnx
    python scripts/inference_demo.py --source 0
    python scripts/inference_demo.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger("dms.inference_demo")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from src.preprocessing import preprocess_frame  # noqa: E402

# Constants

LABEL_NAMES = ["Alert", "Drowsy", "Distracted"]
LABEL_COLOURS = {
    "Alert":      (0, 200, 0),     # Green (BGR)
    "Drowsy":     (0, 200, 255),   # Yellow/Orange
    "Distracted": (0, 0, 220),     # Red
}
LABEL_COLOURS_HEX = {
    "Alert":      "#00C800",
    "Drowsy":     "#FFC800",
    "Distracted": "#DC0000",
}

# Impairment proxy thresholds (from feature_config.json defaults)
IMPAIRMENT_DEFAULTS = {
    "gaze_stability_threshold": 8.0,
    "blink_duration_threshold_s": 0.4,
    "head_nod_threshold": 3,
}


# Helpers

def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_feature_config(path: Path) -> Dict[str, Any]:
    """Load feature_config.json produced by export_onnx.py."""
    if not path.exists():
        logger.warning("feature_config.json not found at %s, using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _try_init_audio() -> Optional[Any]:
    """Try to initialise pygame for audio alerts. Returns mixer or None."""
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        logger.info("Audio alerts enabled (pygame)")
        return pygame.mixer
    except Exception as exc:
        logger.warning("Audio alerts disabled: %s", exc)
        return None


def _generate_alert_sound(mixer: Any) -> Optional[Any]:
    """Generate a simple warning beep as a pygame Sound object."""
    try:
        import pygame
        sample_rate = 44100
        duration_s = 0.3
        freq_hz = 880  # A5
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        wave = (np.sin(2 * np.pi * freq_hz * t) * 32767 * 0.5).astype(np.int16)
        sound = pygame.mixer.Sound(buffer=wave.tobytes())
        return sound
    except Exception as exc:
        logger.warning("Could not generate alert sound: %s", exc)
        return None


# ONNX Session

class OnnxInferenceSession:
    """Thin wrapper around ONNX Runtime for DriverStateNet inference."""

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        use = [p for p in providers if p in available]
        logger.info("ONNX providers: %s", use)

        self.session = ort.InferenceSession(str(onnx_path), providers=use)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("ONNX model loaded: %s", onnx_path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run the ONNX model on a (batch, seq_len, num_features) array and return class logits."""
        logits = self.session.run(
            [self.output_name],
            {self.input_name: features.astype(np.float32)},
        )[0]
        return logits

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        logits = self.predict(features)
        # Stable softmax
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)


# Overlay Drawing

class OverlayRenderer:
    """Draws all HUD elements onto the video frame."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        class_names: List[str] = LABEL_NAMES,
    ) -> None:
        self.w = frame_width
        self.h = frame_height
        self.class_names = class_names
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN

    def draw_state_badge(
        self, frame: np.ndarray, state: str, confidence: float
    ) -> None:
        """Draw a colour-coded state badge in the top-left corner."""
        colour = LABEL_COLOURS.get(state, (200, 200, 200))
        # Semi-transparent background
        overlay = frame.copy()
        badge_w, badge_h = 280, 70
        cv2.rectangle(overlay, (10, 10), (10 + badge_w, 10 + badge_h), colour, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + badge_w, 10 + badge_h), colour, 2)
        text = f"{state.upper()}"
        cv2.putText(frame, text, (25, 55), self.font, 1.2, (255, 255, 255), 2)
        conf_text = f"{confidence:.0%}"
        cv2.putText(frame, conf_text, (200, 55), self.font, 0.8, (255, 255, 255), 2)

    def draw_confidence_bars(
        self, frame: np.ndarray, probs: np.ndarray
    ) -> None:
        """Draw horizontal probability bars for each class."""
        bar_x = self.w - 250
        bar_w = 200
        bar_h = 20
        y_start = 20

        for i, (name, prob) in enumerate(zip(self.class_names, probs)):
            y = y_start + i * (bar_h + 10)
            colour = LABEL_COLOURS.get(name, (200, 200, 200))

            cv2.putText(
                frame, f"{name}:", (bar_x - 5, y + 15),
                self.font_small, 1.1, (220, 220, 220), 1,
            )

            cv2.rectangle(
                frame, (bar_x + 80, y), (bar_x + 80 + bar_w, y + bar_h),
                (60, 60, 60), -1,
            )
            fill_w = int(bar_w * prob)
            cv2.rectangle(
                frame, (bar_x + 80, y), (bar_x + 80 + fill_w, y + bar_h),
                colour, -1,
            )
            cv2.putText(
                frame, f"{prob:.0%}", (bar_x + 80 + bar_w + 5, y + 15),
                self.font_small, 1.0, (200, 200, 200), 1,
            )

    def draw_feature_gauges(
        self, frame: np.ndarray, features: Dict[str, float]
    ) -> None:
        """Draw key feature values as compact gauges on the left side."""
        gauge_features = [
            ("EAR", "ear_avg", 0.0, 0.5),
            ("MAR", "mar", 0.0, 1.0),
            ("PERCLOS", "perclos", 0.0, 1.0),
            ("Yaw", "yaw", -90.0, 90.0),
            ("Pitch", "pitch", -90.0, 90.0),
            ("Gaze Stab.", "gaze_stability", 0.0, 15.0),
            ("Blink/min", "blink_rate", 0.0, 40.0),
            ("Off-Road %", "eyes_off_road_pct", 0.0, 1.0),
        ]

        x_start = 10
        y_start = 100
        bar_w = 120
        bar_h = 14
        gap = 22

        for i, (label, key, vmin, vmax) in enumerate(gauge_features):
            y = y_start + i * gap
            val = features.get(key, 0.0)

            if vmax - vmin > 1e-6:
                norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = 0.0

            display_val = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
            cv2.putText(
                frame, f"{label}: {display_val}",
                (x_start, y + 11), self.font_plain_if_available, 1.0,
                (180, 180, 180), 1,
            )

            bx = x_start + 110
            cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), (50, 50, 50), -1)

            # Fill colour signals danger level; direction depends on the metric.
            if key in ("perclos", "gaze_stability", "eyes_off_road_pct"):
                # Higher is worse.
                if norm > 0.7:
                    fill_col = (0, 0, 220)
                elif norm > 0.4:
                    fill_col = (0, 180, 255)
                else:
                    fill_col = (0, 200, 0)
            elif key in ("ear_avg",):
                # Lower is worse.
                if norm < 0.3:
                    fill_col = (0, 0, 220)
                elif norm < 0.5:
                    fill_col = (0, 180, 255)
                else:
                    fill_col = (0, 200, 0)
            else:
                fill_col = (200, 150, 50)

            fill_w = int(bar_w * norm)
            cv2.rectangle(frame, (bx, y), (bx + fill_w, y + bar_h), fill_col, -1)

    @property
    def font_plain_if_available(self):
        return cv2.FONT_HERSHEY_PLAIN

    def draw_explainability(
        self, frame: np.ndarray, explanations: List[str]
    ) -> None:
        """Draw explainability text lines at the bottom."""
        y_start = self.h - 20 - len(explanations) * 22
        for i, text in enumerate(explanations):
            y = y_start + i * 22
            # Draw a dark offset copy first so text stays readable over any background.
            cv2.putText(frame, text, (12, y + 1), self.font_small, 1.1, (0, 0, 0), 2)
            cv2.putText(frame, text, (10, y), self.font_small, 1.1, (255, 255, 200), 1)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter in top-right."""
        text = f"FPS: {fps:.0f}"
        cv2.putText(
            frame, text, (self.w - 130, self.h - 15),
            self.font, 0.6, (150, 150, 150), 1,
        )

    def draw_impairment_warning(self, frame: np.ndarray) -> None:
        """Draw a flashing impairment warning banner."""
        # Flash the banner by toggling visibility a few times per second.
        if int(time.time() * 3) % 2 == 0:
            overlay = frame.copy()
            banner_h = 50
            y_top = self.h // 2 - banner_h // 2
            cv2.rectangle(
                overlay, (0, y_top), (self.w, y_top + banner_h),
                (0, 0, 180), -1,
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(
                frame, "!! POSSIBLE IMPAIRMENT DETECTED !!",
                (self.w // 2 - 260, y_top + 35),
                self.font, 0.9, (255, 255, 255), 2,
            )

    def draw_help(self, frame: np.ndarray) -> None:
        """Draw controls help overlay."""
        lines = [
            "Controls:",
            "  q - Quit",
            "  c - Recalibrate EAR",
            "  r - Reset all state",
            "  m - Toggle mute",
            "  h - Toggle this help",
        ]
        overlay = frame.copy()
        bx, by = self.w // 2 - 120, self.h // 2 - 80
        bw, bh = 240, 160
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (bx + 15, by + 25 + i * 22),
                self.font_small, 1.1, (220, 220, 220), 1,
            )

    def draw_recording_dot(self, frame: np.ndarray) -> None:
        """Draw a small recording indicator dot."""
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (self.w - 20, 20), 8, (0, 0, 255), -1)


# Explainability Engine

def generate_explanations(
    features: Dict[str, float],
    state: str,
    fps: float,
) -> List[str]:
    """Generate human-readable explanation strings from features."""
    explanations: List[str] = []

    ear = features.get("ear_avg", 0.3)
    mar = features.get("mar", 0.0)
    perclos = features.get("perclos", 0.0)
    yaw = features.get("yaw", 0.0)
    pitch = features.get("pitch", 0.0)
    gaze_stability = features.get("gaze_stability", 0.0)
    blink_rate = features.get("blink_rate", 0.0)
    blink_dur = features.get("blink_duration_avg", 0.0)
    mouth_frames = features.get("mouth_open_duration", 0.0)
    head_nods = features.get("head_nod_count", 0.0)

    # Eye closure
    if ear < 0.21:
        explanations.append(f"Eyes closed (EAR={ear:.2f})")

    # PERCLOS
    if perclos > 0.4:
        explanations.append(f"High PERCLOS: {perclos:.0%} eyes closed in 60s")
    elif perclos > 0.15:
        explanations.append(f"Elevated PERCLOS: {perclos:.0%}")

    # Yawning
    if mouth_frames > 0 and mar > 0.6:
        mouth_sec = mouth_frames / max(fps, 1.0)
        explanations.append(f"Yawning: mouth open {mouth_sec:.1f}s (MAR={mar:.2f})")

    # Head turn (distraction)
    if abs(yaw) > 30:
        direction = "right" if yaw > 0 else "left"
        explanations.append(f"Head turned {abs(yaw):.0f} deg {direction}")

    # Head nod (drowsiness)
    if head_nods >= 2:
        explanations.append(f"Head nodding detected ({int(head_nods)} nods in 10s)")

    # Head pitch down
    if pitch < -20:
        explanations.append(f"Head drooping (pitch={pitch:.0f} deg)")

    # Gaze off road
    # if eyes_off > 0.5:
    #     explanations.append(f"Eyes off road {eyes_off:.0%} of last 5s")

    # Blink rate
    if blink_rate > 25:
        explanations.append(f"High blink rate: {blink_rate:.0f}/min")
    elif blink_rate < 5 and blink_rate > 0:
        explanations.append(f"Low blink rate: {blink_rate:.0f}/min (staring)")

    # Slow blinks
    if blink_dur > 0.3:
        explanations.append(f"Slow blinks: avg {blink_dur * 1000:.0f}ms")

    # Gaze instability
    if gaze_stability > 8.0:
        explanations.append(f"Gaze unstable (sigma={gaze_stability:.1f} deg)")

    # Limit to top 4 most important
    return explanations[:4]


# Impairment Proxy

def check_impairment_proxy(
    features: Dict[str, float],
    state: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    """True when at least 3 of 4 impairment signals (drowsy, unstable gaze, abnormal blinks, head bobbing) fire."""
    if thresholds is None:
        thresholds = IMPAIRMENT_DEFAULTS

    is_drowsy = state == "Drowsy"
    gaze_unstable = features.get("gaze_stability", 0.0) > thresholds.get(
        "gaze_stability_threshold", 8.0
    )
    abnormal_blinks = features.get("blink_duration_avg", 0.0) > thresholds.get(
        "blink_duration_threshold_s", 0.4
    )
    head_bobbing = features.get("head_nod_count", 0.0) > thresholds.get(
        "head_nod_threshold", 3
    )

    signals = sum([is_drowsy, gaze_unstable, abnormal_blinks, head_bobbing])
    return signals >= 3


# Main Demo Loop

class InferenceDemo:
    """Real-time driver monitoring demo with webcam."""

    def __init__(
        self,
        onnx_path: str | Path,
        mediapipe_model_path: str | Path,
        source: int | str = 0,
        seq_len: int = 90,
        classify_every: int = 5,
        alert_duration_s: float = 2.0,
        fps_hint: float = 30.0,
        cfg: Optional[Dict[str, Any]] = None,
        mute: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.classify_every = classify_every
        self.alert_duration_s = alert_duration_s
        self.fps_hint = fps_hint
        self.mute = mute
        self._cfg = cfg or {}

        # ONNX model
        self.model = OnnxInferenceSession(onnx_path)

        # Feature extractor
        feature_cfg = self._cfg.get("features", {})
        self.extractor = FeatureExtractor(
            model_path=mediapipe_model_path,
            fps=fps_hint,
            cfg=feature_cfg,
        )

        # Video capture
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        if isinstance(source, int) and sys.platform == "win32":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps_hint = actual_fps
        logger.info(
            "Video source opened: %dx%d @ %.1f fps",
            self.frame_w, self.frame_h, self.fps_hint,
        )

        # Rolling feature buffer
        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
        self._frame_count: int = 0

        # State tracking
        self._current_state: str = "Alert"
        self._current_probs: np.ndarray = np.array([1.0, 0.0, 0.0])
        self._current_features: Dict[str, float] = {}
        self._state_history: Deque[Tuple[str, float]] = deque(maxlen=300)

        # Alert sustained state
        self._alert_state_start: Optional[float] = None
        self._alert_active: bool = False
        self._impairment_detected: bool = False
        
        # Heuristic state
        self._distracted_frames: int = 0
        self._baseline_yaw: float = 0.0
        self._baseline_pitch: float = 0.0

        # Overlay
        self.renderer = OverlayRenderer(self.frame_w, self.frame_h)

        # Audio
        self._mixer = _try_init_audio() if not mute else None
        self._alert_sound = (
            _generate_alert_sound(self._mixer) if self._mixer else None
        )
        self._last_audio_time: float = 0.0

        # FPS tracking
        self._fps_times: Deque[float] = deque(maxlen=60)
        self._show_help: bool = False

        # Preprocessing config
        prep_cfg = self._cfg.get("preprocessing", {})
        clahe_cfg = prep_cfg.get("clahe", {})
        gamma_cfg = prep_cfg.get("gamma_correction", {})
        self._clahe_clip = clahe_cfg.get("clip_limit", 2.0)
        self._clahe_tile = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        self._brightness_thresh = clahe_cfg.get("brightness_threshold", 100.0)
        self._gamma = gamma_cfg.get("gamma", 1.2)
        self._adaptive_gamma = gamma_cfg.get("adaptive", True)
        self._apply_gamma = gamma_cfg.get("enabled", True)

        # Normalisation stats (from training, if available)
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

    def set_normalisation_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Set feature normalisation stats from training (z-score)."""
        self._norm_mean = mean
        self._norm_std = std
        # Avoid division by zero
        self._norm_std[self._norm_std < 1e-8] = 1.0
        logger.info("Normalisation stats loaded: %d features", len(mean))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE and gamma preprocessing."""
        return preprocess_frame(
            frame,
            clahe_clip=self._clahe_clip,
            tile_size=self._clahe_tile,
            brightness_threshold=self._brightness_thresh,
            gamma=self._gamma,
            adaptive_gamma=self._adaptive_gamma,
            apply_gamma=self._apply_gamma,
        )

    def _extract_and_buffer(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Dict[str, float]:
        """Extract features from one frame and add to rolling buffer."""
        features = self.extractor.extract(frame, timestamp_ms)

        # Convert to numpy vector in canonical order
        feat_vec = np.array(
            [features[name] for name in FEATURE_NAMES],
            dtype=np.float32,
        )
        self._feature_buffer.append(feat_vec)
        return features

    def _classify(self) -> Tuple[str, np.ndarray]:
        """Run classification on the current feature buffer."""
        if len(self._feature_buffer) < self.seq_len:
            return self._current_state, self._current_probs

        # Stack buffer into (1, seq_len, 18)
        buffer_arr = np.stack(list(self._feature_buffer), axis=0)

        # Apply normalisation if available
        if self._norm_mean is not None:
            buffer_arr = (buffer_arr - self._norm_mean) / self._norm_std

        batch = buffer_arr[np.newaxis, ...]  # (1, seq_len, 18)

        probs = self.model.predict_proba(batch)[0]  # (3,)
        pred_idx = int(np.argmax(probs))
        state = LABEL_NAMES[pred_idx]

        # Webcam placement can differ from training data, so force Distracted on large head angles.
        if state == "Alert" and self._current_features:
            raw = self._current_features
            # features.py names are swapped vs physical axes: raw yaw is physical roll, raw roll is physical pitch.
            phys_yaw = raw.get("pitch", 0.0) - self._baseline_yaw
            phys_pitch = raw.get("roll", 0.0) - self._baseline_pitch
            gaze_pitch = raw.get("gaze_pitch", 0.0)
            ear = raw.get("ear_avg", 1.0)

            heuristic_distracted = False

            # Looking sharply to the side.
            if abs(phys_yaw) > 30.0:
                heuristic_distracted = True
            # Looking down at a phone: head pitched, gaze angled down, or eyes near-closed while facing forward.
            elif abs(phys_pitch) > 25.0 or abs(gaze_pitch) > 40.0 or (ear < 0.35 and abs(phys_yaw) < 15.0):
                heuristic_distracted = True

            if heuristic_distracted:
                self._distracted_frames += self.classify_every
            else:
                self._distracted_frames = max(0, self._distracted_frames - self.classify_every)

            # Require ~1.5s sustained (45 frames @30fps) so quick shoulder checks don't flash Distracted.
            if self._distracted_frames > 45:
                state = "Distracted"
                probs = np.array([0.05, 0.05, 0.90], dtype=np.float32)

        return state, probs

    def _update_alert_state(self, state: str, timestamp: float) -> None:
        """Track sustained non-alert states for audio alerts."""
        self._state_history.append((state, timestamp))

        if state != "Alert":
            if self._alert_state_start is None:
                self._alert_state_start = timestamp
            elif (timestamp - self._alert_state_start) >= self.alert_duration_s:
                self._alert_active = True
        else:
            self._alert_state_start = None
            self._alert_active = False

        # Impairment proxy
        self._impairment_detected = check_impairment_proxy(
            self._current_features, state
        )

    def _play_alert(self, timestamp: float) -> None:
        """Play audio alert if conditions met and not muted."""
        if self.mute or self._alert_sound is None:
            return
        if not self._alert_active:
            return
        # Don't spam: at most one alert every 2 seconds.
        if timestamp - self._last_audio_time < 2.0:
            return
        try:
            self._alert_sound.play()
            self._last_audio_time = timestamp
        except Exception:
            pass

    def run(self) -> None:
        """Main demo loop."""
        logger.info("Starting inference demo - press 'q' to quit, 'h' for help")

        window_name = "DMS - Driver Monitoring System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(self.frame_w, 1280), min(self.frame_h, 720))

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self._frame_count < 10:
                        logger.debug("Waiting for webcam to warm up...")
                        time.sleep(0.1)
                        self._frame_count += 1
                        continue
                    logger.info("Video source ended or no frame available")
                    break

                self._frame_count += 1
                timestamp_ms = int(self._frame_count * (1000.0 / self.fps_hint))
                timestamp_s = timestamp_ms / 1000.0

                # Preprocess
                preprocessed = self._preprocess(frame)

                # Extract features
                self._current_features = self._extract_and_buffer(
                    preprocessed, timestamp_ms
                )

                # Classify (every N frames)
                if self._frame_count % self.classify_every == 0:
                    self._current_state, self._current_probs = self._classify()

                # Alert logic
                self._update_alert_state(self._current_state, timestamp_s)
                self._play_alert(timestamp_s)

                # Draw overlay
                display = frame.copy()

                self.renderer.draw_state_badge(
                    display,
                    self._current_state,
                    float(self._current_probs.max()),
                )
                self.renderer.draw_confidence_bars(display, self._current_probs)
                self.renderer.draw_feature_gauges(display, self._current_features)

                explanations = generate_explanations(
                    self._current_features,
                    self._current_state,
                    self.fps_hint,
                )
                self.renderer.draw_explainability(display, explanations)

                if self._impairment_detected:
                    self.renderer.draw_impairment_warning(display)

                # FPS
                self._fps_times.append(time.perf_counter())
                if len(self._fps_times) > 1:
                    dt = self._fps_times[-1] - self._fps_times[0]
                    fps = (len(self._fps_times) - 1) / dt if dt > 0 else 0
                else:
                    fps = 0
                self.renderer.draw_fps(display, fps)

                self.renderer.draw_recording_dot(display)

                if self._show_help:
                    self.renderer.draw_help(display)

                # Buffer fill indicator
                fill_pct = len(self._feature_buffer) / self.seq_len
                if fill_pct < 1.0:
                    cv2.putText(
                        display,
                        f"Buffering... {fill_pct:.0%}",
                        (self.frame_w // 2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    )

                cv2.imshow(window_name, display)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' - exiting")
                    break
                elif key == ord("c"):
                    logger.info("Recalibrating EAR and Baseline Pose...")
                    self.extractor._calibration.reset()
                    if self._current_features:
                        self._baseline_yaw = self._current_features.get("pitch", 0.0)
                        self._baseline_pitch = self._current_features.get("roll", 0.0)
                elif key == ord("r"):
                    logger.info("Resetting all state...")
                    self.extractor.reset()
                    self._feature_buffer.clear()
                    self._current_state = "Alert"
                    self._current_probs = np.array([1.0, 0.0, 0.0])
                    self._alert_state_start = None
                    self._alert_active = False
                    self._impairment_detected = False
                elif key == ord("m"):
                    self.mute = not self.mute
                    logger.info("Audio %s", "muted" if self.mute else "unmuted")
                elif key == ord("h"):
                    self._show_help = not self._show_help

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - exiting")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            logger.info(
                "Demo ended: %d frames processed", self._frame_count
            )


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time Driver Monitoring System demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--onnx", default=None,
        help="Path to ONNX model (overrides config)",
    )
    parser.add_argument(
        "--mediapipe-model", default=None,
        help="Path to MediaPipe .task file (overrides config)",
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: camera index (0, 1, ...) or path to video file",
    )
    parser.add_argument(
        "--seq-len", type=int, default=90,
        help="Sliding window length (default: 90)",
    )
    parser.add_argument(
        "--classify-every", type=int, default=5,
        help="Run classification every N frames (default: 5)",
    )
    parser.add_argument(
        "--alert-duration", type=float, default=2.0,
        help="Seconds of sustained non-alert before audio alert (default: 2.0)",
    )
    parser.add_argument(
        "--mute", action="store_true",
        help="Disable audio alerts",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--norm-stats", default=None,
        help="Path to normalisation stats JSON (mean/std from training)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    # Load config
    cfg = _load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    # Resolve paths
    project_root = Path(paths_cfg.get("project_root", _PROJECT_ROOT))
    models_dir = project_root / "models"

    onnx_path = (
        Path(args.onnx) if args.onnx
        else models_dir / "driver_state_net.onnx"
    )
    mp_model_path = (
        Path(args.mediapipe_model) if args.mediapipe_model
        else Path(paths_cfg.get(
            "mediapipe_model",
            str(models_dir / "face_landmarker_v2_with_blendshapes.task"),
        ))
    )

    # Check files exist
    if not onnx_path.exists():
        logger.error("ONNX model not found: %s", onnx_path)
        logger.error("Run scripts/export_onnx.py first to export the model.")
        sys.exit(1)
    if not mp_model_path.exists():
        logger.error("MediaPipe model not found: %s", mp_model_path)
        logger.error(
            "Download face_landmarker_v2_with_blendshapes.task from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        sys.exit(1)

    # Resolve source
    source = args.source
    if source.isdigit():
        source = int(source)

    # Create and run demo
    demo = InferenceDemo(
        onnx_path=onnx_path,
        mediapipe_model_path=mp_model_path,
        source=source,
        seq_len=args.seq_len,
        classify_every=args.classify_every,
        alert_duration_s=args.alert_duration,
        cfg=cfg,
        mute=args.mute,
    )

    # Load normalisation stats if available
    if args.norm_stats:
        norm_path = Path(args.norm_stats)
        if norm_path.exists():
            with open(norm_path, "r") as f:
                stats = json.load(f)
            demo.set_normalisation_stats(
                np.array(stats["mean"], dtype=np.float32),
                np.array(stats["std"], dtype=np.float32),
            )
    else:
        # Try to load from feature_config.json
        fc_path = models_dir / "feature_config.json"
        fc = _load_feature_config(fc_path)
        norm_stats = fc.get("normalisation", {})
        if "mean" in norm_stats and "std" in norm_stats:
            demo.set_normalisation_stats(
                np.array(norm_stats["mean"], dtype=np.float32),
                np.array(norm_stats["std"], dtype=np.float32),
            )

    logger.info("=" * 60)
    logger.info("DMS INFERENCE DEMO")
    logger.info("  ONNX model:    %s", onnx_path)
    logger.info("  MediaPipe:     %s", mp_model_path)
    logger.info("  Source:        %s", args.source)
    logger.info("  Seq len:       %d", args.seq_len)
    logger.info("  Classify/N:    %d", args.classify_every)
    logger.info("  Alert delay:   %.1f s", args.alert_duration)
    logger.info("  Audio:         %s", "muted" if args.mute else "enabled")
    logger.info("=" * 60)

    demo.run()


if __name__ == "__main__":
    main()
