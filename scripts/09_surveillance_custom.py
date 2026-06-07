#!/usr/bin/env python3
"""
09_surveillance_custom.py — Custom Driver Surveillance Pipeline with HUD overlays and deduplicated clip saving
========================================================================================================

Full 4-layer pipeline with custom tracker (selfie mode support), custom feature extractor (robust face loss & absolute gaze),
and custom clip writer (contiguous frame deduplication and HUD frame overlays on saved video clips).

Usage
-----
    python scripts/09_surveillance_custom.py --source 0
    python scripts/09_surveillance_custom.py --source path/to/video.mp4 --no-display --selfie
    python scripts/09_surveillance_custom.py --help

Controls (display on)
----------------------
    q   Quit
    r   Reset all state
    h   Toggle help overlay

Output
------
    output/clips/   — MP4 clips
    output/logs/    — events_<timestamp>.csv and .json
"""

from __future__ import annotations

import os
# Optimize CPU utilization by limiting threads for highly parallelised libraries
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(2)

import numpy as np
import yaml
import torch
torch.set_num_threads(2)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.detection import detect_objects_in_roi, detect_persons, load_yolo, pad_box
from src.event_engine import EventEngine, SignalFrame
from src.custom_clip_writer import CustomClipWriter
from src.event_logger import EventLogger
from src.features import FEATURE_NAMES
from src.custom_features import CustomFeatureExtractor
from src.custom_detection import CustomDriverTracker
from src.preprocessing import preprocess_frame
from src.hailo_backend import detect_persons_hailo, detect_objects_in_roi_hailo

logger = logging.getLogger("dms.surveillance_custom")

LABEL_NAMES = ["Alert", "Drowsy", "Distracted"]
LABEL_COLOURS = {"Alert": (0, 200, 0), "Drowsy": (0, 200, 255), "Distracted": (0, 0, 220)}
EVENT_COLOURS = {"phone": (0, 180, 255), "food": (0, 255, 180), "danger": (0, 0, 255),
                 "drowsy": (0, 200, 255), "distracted": (0, 0, 220), "eyes_off": (180, 0, 255)}


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class OnnxInferenceSession:
    """Thin ONNX Runtime wrapper for DriverStateNet.

    Input:  "features"  shape (1, seq_len, 18) float32
    Output: "logits"    shape (1, 3)            float32
    """
    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        # Force CPUExecutionProvider for ONNX Runtime to avoid cuDNN version conflicts with PyTorch/YOLOv8
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("ONNX loaded: %s  providers=%s", onnx_path, providers)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """features: (1, seq_len, 18) float32 → softmax probs (1, 3)."""
        logits = self.session.run(
            [self.output_name], {self.input_name: features.astype(np.float32)}
        )[0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)


def _draw_hud(
    frame: np.ndarray,
    driver_box: Optional[np.ndarray],
    state: str,
    probs: np.ndarray,
    object_scores: Dict[str, float],
    recent_events: List[str],
    fps: float,
    buffering_pct: float,
    show_help: bool,
    detected_objects: List[Dict[str, Any]] = None,
    feats: Dict[str, float] = None,
    full_hud: bool = True,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    small = cv2.FONT_HERSHEY_PLAIN

    if driver_box is not None:
        x1, y1, x2, y2 = [int(v) for v in driver_box]
        col = LABEL_COLOURS.get(state, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, "DRIVER", (x1, max(y1 - 8, 0)), small, 1.2, col, 1)

    # Highlight all detected YOLO objects in driver's ROI
    if detected_objects:
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = [int(v) for v in obj["box"]]
            label = obj["label"]
            conf = obj["conf"]
            color = EVENT_COLOURS.get(label, (0, 255, 255))
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(out, f"{label.upper()} {conf:.0%}", (ox1, max(oy1 - 5, 0)),
                        small, 1.2, color, 1)

    if full_hud:
        col = LABEL_COLOURS.get(state, (200, 200, 200))
        ov = out.copy()
        cv2.rectangle(ov, (10, 10), (290, 70), col, -1)
        cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
        cv2.rectangle(out, (10, 10), (290, 70), col, 2)
        cv2.putText(out, state.upper(), (25, 55), font, 1.2, (255, 255, 255), 2)
        cv2.putText(out, f"{float(probs.max()):.0%}", (210, 55), font, 0.9, (255, 255, 255), 2)

        y_obj = 85
        for evt, conf in object_scores.items():
            if conf > 0.0:
                cv2.putText(out, f"{evt}: {conf:.0%}", (15, y_obj), small, 1.2,
                            EVENT_COLOURS.get(evt, (200, 200, 200)), 1)
                y_obj += 18

        bar_x, bar_w = w - 250, 180
        for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
            y = 20 + i * 30
            c = LABEL_COLOURS.get(name, (200, 200, 200))
            cv2.putText(out, f"{name}:", (bar_x - 5, y + 14), small, 1.1, (220, 220, 220), 1)
            cv2.rectangle(out, (bar_x + 80, y), (bar_x + 80 + bar_w, y + 18), (60, 60, 60), -1)
            cv2.rectangle(out, (bar_x + 80, y), (bar_x + 80 + int(bar_w * prob), y + 18), c, -1)
            cv2.putText(out, f"{prob:.0%}", (bar_x + 80 + bar_w + 5, y + 14), small, 1.0, (200, 200, 200), 1)

        for i, evt in enumerate(recent_events):
            cv2.putText(out, f"SAVED: {evt.upper()}", (10, h - 30 - i * 25),
                        font, 0.65, EVENT_COLOURS.get(evt, (200, 200, 200)), 2)

        cv2.putText(out, f"FPS:{fps:.0f}", (w - 110, h - 10), small, 1.1, (140, 140, 140), 1)
        
        if feats:
            debug_txt = f"YAW:{feats.get('pitch', 0):.0f} PITCH:{feats.get('roll', 0):.0f} EAR:{feats.get('ear_avg', 0):.2f} MAR:{feats.get('mar', 0):.2f}"
            cv2.putText(out, debug_txt, (10, h - 10), small, 1.0, (0, 255, 255), 1)

        if show_help:
            lines = ["Controls:", "  q - Quit", "  r - Reset state", "  h - Toggle help"]
            ov2 = out.copy()
            bx, by, bw, bh = w // 2 - 130, h // 2 - 60, 260, 110
            cv2.rectangle(ov2, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
            cv2.addWeighted(ov2, 0.85, out, 0.15, 0, out)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)
            for j, line in enumerate(lines):
                cv2.putText(out, line, (bx + 12, by + 25 + j * 22), small, 1.1, (220, 220, 220), 1)

    return out


class SurveillancePipeline:
    """Wires all custom components for the 4-layer surveillance pipeline."""

    def __init__(
        self,
        onnx_path: Path,
        mediapipe_model_path: Path,
        yolo_cfg: Dict[str, Any],
        events_cfg: Dict[str, Any],
        clips_cfg: Dict[str, Any],
        logging_cfg: Dict[str, Any],
        dms_cfg: Dict[str, Any],
        source=None,
        seq_len: int = 90,
        display: bool = True,
        project_root: Path = _PROJECT_ROOT,
        backend: str = "cpu",
        hef_path: Optional[Path] = None,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> None:
        self._display = display
        self._seq_len = seq_len
        self._yolo_cfg = yolo_cfg
        self._yolo = None
        try:
            self._backend = backend
            if backend == "hailo":
                if hef_path is None or not hef_path.exists():
                    raise FileNotFoundError(
                        f"--backend hailo requires a valid --hef path. Got: {hef_path}"
                    )
                from src.hailo_backend import load_hailo_yolo
                self._yolo = load_hailo_yolo(str(hef_path))
                logger.info("Using Hailo-8 backend: %s", hef_path)
            else:
                self._yolo = load_yolo(yolo_cfg.get("model_path", "yolov8n.pt"))
                logger.info("Using CPU backend (ultralytics)")

            self._onnx = OnnxInferenceSession(onnx_path)
            feature_cfg = dms_cfg.get("features", {})
            fps_hint = feature_cfg.get("fps", 30.0)
            
            # Override calibration frames for real-time surveillance (2 seconds)
            if "ear" not in feature_cfg:
                feature_cfg["ear"] = {}
            feature_cfg["ear"]["calibration_frames"] = int(2.0 * fps_hint)
            
            self._extractor = CustomFeatureExtractor(
                model_path=str(mediapipe_model_path), fps=fps_hint, cfg=feature_cfg
            )

            # ---------- Camera setup (optional) ----------
            self._cap = None
            self._source = source
            if source is not None:
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                self._source = source
                self._cap = (
                    cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    if isinstance(source, int) and sys.platform == "win32"
                    else cv2.VideoCapture(source)
                )
                if not self._cap.isOpened():
                    raise RuntimeError(f"Cannot open source: {source}")
                self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._fps: float = actual_fps if actual_fps > 0 else fps_hint
            else:
                # External-capture mode: caller provides frames via process_frame()
                self._frame_w = frame_w
                self._frame_h = frame_h
                self._fps: float = fps_hint
                logger.info("External-capture mode: %dx%d", frame_w, frame_h)

            self._tracker = CustomDriverTracker(yolo_cfg, self._frame_w, self._frame_h)
            self._engine = EventEngine(events_cfg, fps=self._fps)

            clip_dir = project_root / clips_cfg.get("output_dir", "output/clips")
            log_dir = project_root / logging_cfg.get("output_dir", "output/logs")
            self._clip_writer = CustomClipWriter(
                output_dir=clip_dir, fps=self._fps,
                pre_buffer_seconds=clips_cfg.get("pre_buffer_seconds", 5.0),
                post_buffer_seconds=clips_cfg.get("post_buffer_seconds", 5.0),
                fourcc=clips_cfg.get("fourcc", "mp4v"),
            )
            self._event_logger = EventLogger(
                output_dir=log_dir, base_name=logging_cfg.get("base_name", "events")
            )
        except Exception:
            if hasattr(self, "_extractor") and self._extractor is not None:
                try:
                    self._extractor.close()
                except Exception as e:
                    logger.warning("Error closing extractor during pipeline initialization abort: %s", e)
            if hasattr(self, "_clip_writer") and self._clip_writer is not None:
                try:
                    self._clip_writer.close()
                except Exception as e:
                    logger.warning("Error closing clip writer during pipeline initialization abort: %s", e)
            if hasattr(self, "_yolo") and self._yolo is not None and hasattr(self._yolo, "close"):
                try:
                    self._yolo.close()
                except Exception as e:
                    logger.warning("Error closing model during pipeline initialization abort: %s", e)
            if hasattr(self, "_cap") and self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning("Error releasing video capture during pipeline initialization abort: %s", e)
            raise

        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._current_state: str = "Alert"
        self._current_probs: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._current_object_scores: Dict[str, float] = {}
        self._current_detected_objects: List[Dict[str, Any]] = []
        self._extractor_skip = feature_cfg.get("skip_frames", 3)
        self._current_feats: Dict[str, float] = {}
        self._current_driver_box: Optional[np.ndarray] = None
        # key=event_type → {"pre_frames": list, "frames": [], "remaining": int, "trigger": dict}
        self._collecting_post: Dict[str, Dict[str, Any]] = {}
        self._fps_times: Deque[float] = deque(maxlen=60)
        self._recent_saved: Deque[Tuple[str, float]] = deque(maxlen=5)
        self._was_heuristic_distracted: bool = False
        self._was_heuristic_drowsy: bool = False
        self._eyes_closed_frames: int = 0
        self._driver_missing_frames: int = 0
        self._show_help: bool = False
        
        # Heuristic state overrides
        self._baseline_yaw: float = 0.0
        self._baseline_pitch: float = 0.0
        self._distracted_frames: int = 0

    def set_normalisation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._norm_mean = mean
        std[std < 1e-8] = 1.0
        self._norm_std = std

    def process_frame(self, raw_frame: np.ndarray, timestamp_ms: int = None) -> dict:
        """Run the full AI pipeline on a single frame and return telemetry.

        This is the core method used by external capture mode (web dashboard).
        The caller owns the camera; this method only does inference and state
        updates.  It is NOT thread-safe — call it from a single thread.

        Returns a telemetry dict (same schema as run_generator).
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        timestamp_s = timestamp_ms / 1000.0

        self._frame_count += 1

        self._fps_times.append(time.perf_counter())
        current_fps = 0.0
        if len(self._fps_times) > 1:
            dt = self._fps_times[-1] - self._fps_times[0]
            current_fps = (len(self._fps_times) - 1) / dt if dt > 0 else 0
        if current_fps <= 0.0:
            current_fps = self._fps

        # Stage 1: person detection + driver isolation
        redetect_every = self._yolo_cfg.get("redetect_every_n_frames", 30)
        should_run_yolo_person = (
            self._tracker._tracked_box is None
            or (self._frame_count % redetect_every == 0)
        )

        if should_run_yolo_person:
            if self._backend == "hailo":
                person_boxes = detect_persons_hailo(
                    self._yolo, raw_frame,
                    conf_thresh=self._yolo_cfg.get("person_conf_thresh", 0.50),
                    person_class_id=self._yolo_cfg.get("person_class_id", 0),
                    imgsz=self._yolo_cfg.get("person_imgsz", 256),
                )
            else:
                person_boxes = detect_persons(
                    self._yolo, raw_frame,
                    conf_thresh=self._yolo_cfg.get("person_conf_thresh", 0.50),
                    person_class_id=self._yolo_cfg.get("person_class_id", 0),
                    device=self._yolo_cfg.get("device", None),
                    imgsz=self._yolo_cfg.get("person_imgsz", self._yolo_cfg.get("imgsz", 320)),
                )
            driver_box = self._tracker.update(person_boxes, self._frame_count)
        else:
            driver_box = self._tracker._tracked_box

        self._current_driver_box = driver_box

        # Stage 2: MediaPipe + DriverStateNet
        if driver_box is None:
            self._driver_missing_frames += 1
            if self._driver_missing_frames > 15:
                self._current_object_scores = {}
                self._current_detected_objects = []
                self._current_state = "Alert"
                self._current_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self._feature_buffer.clear()
                self._current_feats = {}
        else:
            self._driver_missing_frames = 0

        probs = self._current_probs
        feats = self._current_feats
        if driver_box is not None:
            padded = pad_box(driver_box,
                             self._yolo_cfg.get("roi_padding_px", 30),
                             self._frame_w, self._frame_h)
            x1, y1, x2, y2 = [int(v) for v in padded]
            roi = raw_frame[y1:y2, x1:x2]
            if roi.size > 0:
                preprocessed = preprocess_frame(roi)
                
                if self._frame_count % self._extractor_skip == 0 or not self._current_feats:
                    feats = self._extractor.extract(preprocessed, timestamp_ms)
                    self._current_feats = feats
                else:
                    feats = self._current_feats
                    
                if feats and "ear_avg" in feats:
                    feats["gaze_yaw"] = -88.45
                    feats["eyes_off_road_pct"] = 0.98
                    
                    feat_vec = np.array(
                        [feats.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32
                    )
                    
                    insert_count = max(1, int(round(self._fps / max(1.0, current_fps))))
                    for _ in range(insert_count):
                        self._feature_buffer.append(feat_vec)
                        
                    if (len(self._feature_buffer) >= self._seq_len
                            and self._frame_count % 10 == 0):
                        buf = np.stack(list(self._feature_buffer), axis=0)
                        if self._norm_mean is not None:
                            buf = (buf - self._norm_mean) / self._norm_std
                        probs = self._onnx.predict_proba(buf[np.newaxis])[0]
                        self._current_state = LABEL_NAMES[int(np.argmax(probs))]
                        self._current_probs = probs

        # Stage 3: object detection in driver ROI (every 10 frames to save CPU)
        if self._frame_count % 10 == 0:
            if self._backend == "hailo":
                self._current_object_scores, self._current_detected_objects = (
                    detect_objects_in_roi_hailo(self._yolo, raw_frame, driver_box, self._yolo_cfg)
                )
            else:
                self._current_object_scores, self._current_detected_objects = (
                    detect_objects_in_roi(self._yolo, raw_frame, driver_box, self._yolo_cfg)
                )
        object_scores = self._current_object_scores
        detected_objects = self._current_detected_objects

        # --- Heuristic Overrides ---
        if driver_box is not None and feats:
            phys_yaw = feats.get("pitch", 0.0) - self._baseline_yaw
            phys_pitch = feats.get("roll", 0.0) - self._baseline_pitch
            gaze_pitch = feats.get("gaze_pitch", 0.0)
            ear = feats.get("ear_avg", 1.0)
            
            head_distracted = False
            if abs(phys_yaw) > 25.0:
                head_distracted = True
            elif abs(phys_pitch) > 20.0 or abs(gaze_pitch) > 35.0:
                head_distracted = True
                
            trigger_frames = int(0.3 * current_fps)
            cap_frames = int(1.0 * current_fps)
                
            if head_distracted:
                self._distracted_frames = min(cap_frames, self._distracted_frames + 1)
            else:
                self._distracted_frames = max(0, self._distracted_frames - 3)
                
            eyes_closed_or_off = (feats.get("ear_avg", 1.0) < 0.26)
            if eyes_closed_or_off:
                self._eyes_closed_frames = min(cap_frames, self._eyes_closed_frames + 1)
            else:
                self._eyes_closed_frames = max(0, self._eyes_closed_frames - 3)
                
            is_distracted_heuristic = (
                self._distracted_frames > trigger_frames or 
                object_scores.get("phone", 0.0) > 0.45 or 
                object_scores.get("danger", 0.0) > 0.45 or
                self._eyes_closed_frames > 2.0 * current_fps
            )
            
            is_drowsy_heuristic = (
                feats.get("perclos", 0.0) > 0.4 or 
                feats.get("mar", 0.0) > 0.45 or
                self._eyes_closed_frames > 0.2 * current_fps
            )
            
            is_alert_heuristic = (
                not head_distracted and
                not eyes_closed_or_off and
                feats.get("mar", 0.0) < 0.3 and
                abs(phys_yaw) < 20.0 and
                abs(phys_pitch) < 20.0
            )
                
            if is_distracted_heuristic:
                self._current_state = "Distracted"
                self._current_probs = np.array([0.05, 0.05, 0.90], dtype=np.float32)
                self._was_heuristic_distracted = True
            elif self._was_heuristic_distracted:
                self._was_heuristic_distracted = False
                if len(self._feature_buffer) > 0:
                    self._feature_buffer.extend([self._feature_buffer[-1]] * self._seq_len)
                self._current_state = "Alert"
                self._current_probs = np.array([0.90, 0.05, 0.05], dtype=np.float32)
                
            elif is_drowsy_heuristic:
                self._current_state = "Drowsy"
                self._current_probs = np.array([0.05, 0.90, 0.05], dtype=np.float32)
                self._was_heuristic_drowsy = True
            elif self._was_heuristic_drowsy:
                self._was_heuristic_drowsy = False
                if len(self._feature_buffer) > 0:
                    self._feature_buffer.extend([self._feature_buffer[-1]] * self._seq_len)
                self._current_state = "Alert"
                self._current_probs = np.array([0.90, 0.05, 0.05], dtype=np.float32)
                
            elif is_alert_heuristic:
                self._current_state = "Alert"
                self._current_probs = np.array([0.90, 0.05, 0.05], dtype=np.float32)

        # Stage 4: event engine
        signal = SignalFrame(
            phone=object_scores.get("phone", 0.0),
            food=object_scores.get("food", 0.0),
            danger=object_scores.get("danger", 0.0),
            drowsy_prob=float(probs[1]) if probs is not None else 0.0,
            distracted_prob=float(probs[2]),
            alert_prob=float(probs[0]),
            eyes_off_road_pct=feats.get("eyes_off_road_pct", 0.0) if feats else 0.0,
            timestamp_s=timestamp_s,
        )
        triggers = self._engine.update(signal)

        # Build a lightweight HUD frame for clip recording
        hud_frame = _draw_hud(
            frame=raw_frame,
            driver_box=driver_box,
            state=self._current_state,
            probs=self._current_probs,
            object_scores=object_scores,
            recent_events=[evt for evt, t in self._recent_saved if timestamp_s - t < 3.0],
            fps=current_fps,
            buffering_pct=len(self._feature_buffer) / self._seq_len,
            show_help=False,
            detected_objects=detected_objects,
            feats=feats,
            full_hud=True,
        )

        # Push to clip ring buffer
        self._clip_writer.push_frame(hud_frame, timestamp_s)
        self._handle_post_collection(hud_frame, timestamp_s)

        post_needed = int(self._clip_writer._post_s * self._fps)
        for trigger in triggers:
            evt = trigger["event_type"]
            if evt not in self._collecting_post:
                # Add to recent events instantly so UI reflects it's recording
                self._recent_saved.append((evt, timestamp_s))
                
                # Contiguous Frame Deduplication: snapshot the current ring buffer frames
                pre_frames = [f.copy() for f, _ in self._clip_writer._ring_buffer]
                self._collecting_post[evt] = {
                    "pre_frames": pre_frames,
                    "frames": [],
                    "remaining": post_needed,
                    "trigger": trigger
                }

        # Build telemetry dict
        # --- UI Display Boxes ---
        face_box = None
        if self._extractor._last_result and self._extractor._last_result.face_landmarks:
            try:
                lms = self._extractor._last_result.face_landmarks[0]
                h, w = raw_frame.shape[:2]
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                
                # Add 15% padding
                px, py = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
                face_box = [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]
            except IndexError:
                pass

        telemetry = {
            "state": self._current_state,
            "probs": {
                "alert": float(self._current_probs[0]),
                "drowsy": float(self._current_probs[1]),
                "distracted": float(self._current_probs[2]),
            },
            "fps": round(current_fps, 1),
            "objects": {k: round(v, 2) for k, v in object_scores.items() if v > 0},
            "feats": {
                "yaw": round(feats.get("pitch", 0), 1) if feats else 0,
                "pitch": round(feats.get("roll", 0), 1) if feats else 0,
                "ear": round(feats.get("ear_avg", 0), 2) if feats else 0,
                "mar": round(feats.get("mar", 0), 2) if feats else 0,
                "perclos": round(feats.get("perclos", 0), 2) if feats else 0,
            },
            "events": [evt for evt, t in self._recent_saved if timestamp_s - t < 3.0],
            "triggers": [t["event_type"] for t in triggers],
            "frame_count": self._frame_count,
            "driver_box": [int(v) for v in driver_box] if driver_box is not None else None,
            "face_box": [int(v) for v in face_box] if face_box is not None else None,
            "detected_objects": detected_objects,
            "frame_width": int(raw_frame.shape[1]),
            "frame_height": int(raw_frame.shape[0]),
        }

        return telemetry

    def run_generator(self):
        """Yield (raw_frame, hud_frame, telemetry_dict) per frame.

        Uses the internal VideoCapture.  For external-capture mode, use
        process_frame() directly instead.
        """
        if self._cap is None:
            raise RuntimeError("run_generator() requires a video source. "
                               "Use process_frame() for external-capture mode.")
        consecutive_failures = 0
        while True:
            ret, raw_frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                is_camera = True
                if isinstance(self._source, str) and os.path.exists(self._source):
                    is_camera = False
                max_failures = 50 if is_camera else 1
                if consecutive_failures >= max_failures:
                    if is_camera:
                        raise RuntimeError("Failed to read from camera: 50 consecutive frame failures.")
                    else:
                        logger.info("Video source ended.")
                        break
                else:
                    time.sleep(0.1)
                    continue

            consecutive_failures = 0
            telemetry = self.process_frame(raw_frame)

            hud_frame = _draw_hud(
                frame=raw_frame,
                driver_box=self._current_driver_box,
                state=self._current_state,
                probs=self._current_probs,
                object_scores=self._current_object_scores,
                recent_events=[evt for evt, _ in self._recent_saved],
                fps=telemetry["fps"],
                buffering_pct=len(self._feature_buffer) / self._seq_len,
                show_help=self._show_help,
                detected_objects=self._current_detected_objects,
                feats=self._current_feats,
                full_hud=self._display,
            )

            yield raw_frame, hud_frame, telemetry

    def run(self) -> None:
        """Run pipeline with OpenCV display (backward-compatible entry point)."""
        logger.info("Surveillance started — q=quit  r=reset  h=help")
        window = "DMS Surveillance"
        if self._display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(self._frame_w, 1280), min(self._frame_h, 720))

        try:
            for raw_frame, hud_frame, telemetry in self.run_generator():
                # Display
                if self._display:
                    cv2.imshow(window, hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        self.reset()
                    elif key == ord("c"):
                        self.calibrate()
                    elif key == ord("h"):
                        self._show_help = not self._show_help

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt.")
        finally:
            if hasattr(self, "_yolo") and self._yolo is not None and hasattr(self._yolo, "close"):
                try:
                    self._yolo.close()
                except Exception as e:
                    logger.warning("Error closing model: %s", e)
            if hasattr(self, "_cap") and self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning("Error releasing video capture: %s", e)
            if hasattr(self, "_extractor") and self._extractor is not None:
                try:
                    self._extractor.close()
                except Exception as e:
                    logger.warning("Error closing extractor: %s", e)
            if hasattr(self, "_clip_writer") and self._clip_writer is not None:
                try:
                    self._clip_writer.close()
                except Exception as e:
                    logger.warning("Error closing clip writer: %s", e)
            if hasattr(self, "_event_logger") and self._event_logger is not None:
                try:
                    self._event_logger.close()
                except Exception as e:
                    logger.warning("Error closing event logger: %s", e)
            if self._display:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    logger.warning("Error destroying windows: %s", e)
            logger.info("Done. %d frames processed.", self._frame_count)

    def _handle_post_collection(self, frame: np.ndarray, timestamp_s: float) -> None:
        completed = []
        for evt, state in self._collecting_post.items():
            state["frames"].append(frame.copy())
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                completed.append(evt)

        for evt in completed:
            state = self._collecting_post.pop(evt)
            trigger = state["trigger"]
            clip_path = self._clip_writer.save_clip_async(
                event_type=trigger["event_type"],
                confidence=trigger["confidence"],
                trigger_timestamp_s=trigger["timestamp_s"],
                post_frames=state["frames"],
                pre_frames=state["pre_frames"],
            )
            self._event_logger.log(
                event_type=trigger["event_type"],
                confidence=trigger["confidence"],
                trigger_timestamp_s=trigger["timestamp_s"],
                clip_path=clip_path,
                driver_box=self._current_driver_box,
            )

    def reset(self) -> None:
        logger.info("Resetting state.")
        self._tracker.reset()
        self._extractor.reset()
        self._engine.reset()
        self._clip_writer.reset()
        self._feature_buffer.clear()
        self._current_state = "Alert"
        self._current_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._current_detected_objects.clear()
        self._current_feats.clear()
        self._collecting_post.clear()

    def calibrate(self) -> None:
        logger.info("Recalibrating baseline pose and EAR...")
        if self._current_feats:
            self._baseline_yaw = self._current_feats.get("pitch", 0.0)
            self._baseline_pitch = self._current_feats.get("roll", 0.0)
        # Also reset the feature extractor so it recalibrates the EAR baseline for glasses
        self._extractor.reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Driver Surveillance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--yolo-config", default="config/yolo_config.yaml")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model")
    parser.add_argument("--mediapipe-model", default=None)
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--selfie", action="store_true", help="Set camera to selfie/mirrored mode")
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--backend", default="cpu", choices=["cpu", "hailo"],
        help="Inference backend: 'cpu' uses ultralytics, 'hailo' uses Hailo-8 HEF (Pi only)",
    )
    parser.add_argument(
        "--hef", default=None,
        help="Path to compiled .hef model file (required when --backend hailo)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)
    dms_cfg = _load_yaml(args.config)
    yolo_full = _load_yaml(args.yolo_config)

    yolo_cfg = yolo_full.get("yolo", {})
    events_cfg = yolo_full.get("events", {})
    clips_cfg = yolo_full.get("clips", {})
    logging_cfg = yolo_full.get("logging_out", {})
    paths_cfg = dms_cfg.get("paths", {})
    models_dir = _PROJECT_ROOT / "models"

    selfie = args.selfie or yolo_full.get("yolo", {}).get("selfie", False)
    yolo_cfg["selfie"] = selfie
    if "features" not in dms_cfg:
        dms_cfg["features"] = {}
    dms_cfg["features"]["selfie"] = selfie

    onnx_path = Path(args.onnx) if args.onnx else models_dir / "driver_state_net.onnx"
    mp_model = (
        Path(args.mediapipe_model) if args.mediapipe_model
        else Path(paths_cfg.get("mediapipe_model",
                                str(models_dir / "face_landmarker_v2_with_blendshapes.task")))
    )

    for p, label in [(onnx_path, "ONNX"), (mp_model, "MediaPipe")]:
        if not p.exists():
            logger.error("%s model not found: %s", label, p)
            sys.exit(1)

    source = int(args.source) if args.source.isdigit() else args.source

    hef_path = Path(args.hef) if args.hef else None

    pipeline = SurveillancePipeline(
        onnx_path=onnx_path, mediapipe_model_path=mp_model,
        yolo_cfg=yolo_cfg, events_cfg=events_cfg,
        clips_cfg=clips_cfg, logging_cfg=logging_cfg,
        dms_cfg=dms_cfg, source=source,
        seq_len=args.seq_len, display=not args.no_display,
        project_root=_PROJECT_ROOT,
        backend=args.backend,
        hef_path=hef_path,
    )

    fc_path = models_dir / "feature_config.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        norm = fc.get("normalisation", {})
        if "mean" in norm and "std" in norm:
            pipeline.set_normalisation_stats(
                np.array(norm["mean"], dtype=np.float32),
                np.array(norm["std"], dtype=np.float32),
            )

    logger.info("=" * 60)
    logger.info("DMS SURVEILLANCE (CUSTOM)")
    logger.info("  Source:    %s", args.source)
    logger.info("  ONNX:      %s", onnx_path)
    logger.info("  Display:   %s", not args.no_display)
    logger.info("  Selfie:    %s", selfie)
    logger.info("  Backend:   %s", args.backend)
    if args.backend == "hailo":
        logger.info("  HEF:       %s", hef_path)
    logger.info("  Clips →    %s", _PROJECT_ROOT / clips_cfg.get("output_dir", "output/clips"))
    logger.info("  Logs  →    %s", _PROJECT_ROOT / logging_cfg.get("output_dir", "output/logs"))
    logger.info("=" * 60)

    pipeline.run()


if __name__ == "__main__":
    main()
