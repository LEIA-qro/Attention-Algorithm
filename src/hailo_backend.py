"""
hailo_backend.py — Hailo-8 YOLO inference shim
================================================

Replaces ultralytics YOLO calls with hailo_platform SDK calls.
Only used when --backend hailo is passed to 08_surveillance.py.

Requirements
------------
- hailo_platform Python SDK installed (Pi only)
- A compiled .hef file (see plan Task 6 prerequisites)

Public API (mirrors detection.py)
----------------------------------
- load_hailo_yolo(hef_path: str) -> HailoYOLO
- detect_persons_hailo(model, frame, conf_thresh, person_class_id, imgsz)
      -> list[tuple[ndarray, float]]
- detect_objects_in_roi_hailo(model, frame, driver_box, cfg)
      -> tuple[dict[str, float], list[dict]]
- _preprocess_for_hailo(frame, imgsz) -> ndarray   (exposed for testing)
- _parse_hailo_detections(raw, conf_thresh, target_classes)
      -> list[tuple[ndarray, float, int]]           (exposed for testing)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from hailo_platform import (
        ConfigureParams,
        FormatType,
        HEF,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
    _HAILO_AVAILABLE = True
except (ImportError, ModuleNotFoundError, TypeError):
    _HAILO_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "load_hailo_yolo",
    "detect_persons_hailo",
    "detect_objects_in_roi_hailo",
    "_preprocess_for_hailo",
    "_parse_hailo_detections",
]


class HailoYOLO:
    """Thin wrapper around a loaded Hailo HEF network group."""

    def __init__(self, hef_path: str) -> None:
        if not _HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform is not installed. "
                "Install the Hailo SDK (Raspberry Pi with Hailo AI Kit only). "
                "On a dev machine use --backend cpu instead."
            )
        self._active_context = None
        self._infer_pipeline = None
        self._hef = HEF(hef_path)
        self._target = VDevice()
        configure_params = ConfigureParams.init_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_groups = self._target.configure(self._hef, configure_params)
        self._network_group = self._network_groups[0]
        self._network_group_params = self._network_group.create_params()
        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        input_info = self._hef.get_input_vstream_infos()
        self._input_name: str = input_info[0].name
        logger.info("Hailo HEF loaded: %s  input=%s", hef_path, self._input_name)

    def __enter__(self) -> HailoYOLO:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def start(self) -> None:
        """Start the persistent network activation and inference pipeline context."""
        if self._active_context is not None:
            return

        # Activate the network group context
        active_context = self._network_group.activate(self._network_group_params)
        active_context.__enter__()
        self._active_context = active_context

        try:
            # Start the InferVStreams context
            infer_pipeline = InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params,
            )
            infer_pipeline.__enter__()
            self._infer_pipeline = infer_pipeline
        except Exception:
            # Clean up active_context if InferVStreams fails
            try:
                active_context.__exit__(None, None, None)
            except Exception as e:
                logger.error("Error exiting network activation context during fallback: %s", e)
            self._active_context = None
            raise

    def infer(self, preprocessed: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference. preprocessed: uint8 (1, H, W, 3). Returns output dict."""
        self.start()
        input_data = {self._input_name: preprocessed}
        self._infer_pipeline.infer(input_data)
        return self._infer_pipeline.get_output()

    def close(self) -> None:
        """Release InferVStreams, deactivate network, and release VDevice target."""
        if self._infer_pipeline is not None:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception as e:
                logger.error("Error exiting InferVStreams: %s", e)
            self._infer_pipeline = None

        if self._active_context is not None:
            try:
                self._active_context.__exit__(None, None, None)
            except Exception as e:
                logger.error("Error exiting network activation context: %s", e)
            self._active_context = None

        if hasattr(self, "_target") and self._target is not None:
            try:
                if hasattr(self._target, "release"):
                    self._target.release()
            except Exception as e:
                logger.error("Error releasing VDevice target: %s", e)
            self._target = None


def load_hailo_yolo(hef_path: str) -> HailoYOLO:
    """Load a compiled Hailo HEF model. Raises RuntimeError if SDK unavailable."""
    return HailoYOLO(hef_path)


def _preprocess_for_hailo(frame: Optional[np.ndarray], imgsz: int = 320) -> Optional[np.ndarray]:
    """Resize frame to (imgsz, imgsz) and convert to RGB, return uint8 batch (1, H, W, 3)."""
    if frame is None or frame.size == 0:
        return None
    resized = cv2.resize(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb[np.newaxis].astype(np.uint8)


def _parse_hailo_detections(
    raw_output: Dict[str, np.ndarray],
    conf_thresh: float,
    target_classes: Set[int],
) -> List[Tuple[np.ndarray, float, int]]:
    """Parse Hailo YOLOv8n output into (box_xyxy_norm, conf, cls_id) tuples.

    Hailo YOLOv8n outputs a single tensor of shape (1, N, 6) where each row is
    [x1_norm, y1_norm, x2_norm, y2_norm, conf, cls_id] in [0, 1] coordinates.
    Only rows with conf >= conf_thresh and cls_id in target_classes are returned.
    """
    detections: List[Tuple[np.ndarray, float, int]] = []
    if not raw_output:
        return detections

    tensor = next(iter(raw_output.values()))
    if tensor is None or tensor.size == 0:
        return detections

    preds = tensor.reshape(-1, tensor.shape[-1])  # (N, 6)
    if preds.shape[1] < 6:
        return detections

    # NumPy vectorized filtering
    conf_mask = preds[:, 4] >= conf_thresh
    if target_classes:
        class_mask = np.isin(preds[:, 5].astype(int), list(target_classes))
        mask = conf_mask & class_mask
    else:
        mask = conf_mask

    filtered_preds = preds[mask]
    for row in filtered_preds:
        box = row[:4].astype(np.float32)
        conf = float(row[4])
        cls_id = int(row[5])
        detections.append((box, conf, cls_id))
    return detections


def detect_persons_hailo(
    model: HailoYOLO,
    frame: Optional[np.ndarray],
    conf_thresh: float = 0.50,
    person_class_id: int = 0,
    imgsz: int = 256,
) -> List[Tuple[np.ndarray, float]]:
    """Hailo equivalent of detect_persons(). Returns list of (box_xyxy_px, conf)."""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    preprocessed = _preprocess_for_hailo(frame, imgsz=imgsz)
    if preprocessed is None:
        return []
    raw = model.infer(preprocessed)
    raw_dets = _parse_hailo_detections(raw, conf_thresh=conf_thresh,
                                        target_classes={person_class_id})
    persons = []
    for box_norm, conf, _ in raw_dets:
        box_px = np.array([
            box_norm[0] * w,
            box_norm[1] * h,
            box_norm[2] * w,
            box_norm[3] * h,
        ], dtype=np.float32)
        persons.append((box_px, conf))
    return persons


def detect_objects_in_roi_hailo(
    model: HailoYOLO,
    frame: Optional[np.ndarray],
    driver_box: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Hailo equivalent of detect_objects_in_roi(). Returns (scores, objects)."""
    from src.detection import pad_box

    conf_thresh: float = cfg.get("object_conf_thresh", 0.45)
    padding: int = cfg.get("roi_padding_px", 30)
    imgsz: int = cfg.get("imgsz", 320)
    class_to_event: Dict[int, str] = {
        int(k): v for k, v in cfg.get("class_to_event", {}).items()
    }
    object_classes: Dict[int, str] = {
        int(k): v for k, v in cfg.get("object_classes", {}).items()
    }

    event_scores: Dict[str, float] = {v: 0.0 for v in set(class_to_event.values())}
    detected_objects: List[Dict[str, Any]] = []
    if not event_scores:
        return event_scores, detected_objects

    if frame is None or frame.size == 0:
        return event_scores, detected_objects

    fh, fw = frame.shape[:2]
    padded = pad_box(driver_box, padding, fw, fh)
    x1, y1, x2, y2 = int(padded[0]), int(padded[1]), int(padded[2]), int(padded[3])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return event_scores, detected_objects

    roi_h, roi_w = roi.shape[:2]
    preprocessed = _preprocess_for_hailo(roi, imgsz=imgsz)
    if preprocessed is None:
        return event_scores, detected_objects
    raw = model.infer(preprocessed)
    raw_dets = _parse_hailo_detections(raw, conf_thresh=conf_thresh,
                                        target_classes=set(class_to_event.keys()))

    for box_norm, conf, cls_id in raw_dets:
        evt = class_to_event.get(cls_id)
        if evt is None:
            continue
        event_scores[evt] = max(event_scores[evt], conf)
        full_box = np.array([
            box_norm[0] * roi_w + x1,
            box_norm[1] * roi_h + y1,
            box_norm[2] * roi_w + x1,
            box_norm[3] * roi_h + y1,
        ], dtype=np.float32)
        label = object_classes.get(cls_id, evt)
        detected_objects.append({"label": label, "conf": conf, "box": full_box})

    return event_scores, detected_objects
