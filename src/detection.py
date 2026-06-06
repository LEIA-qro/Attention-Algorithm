"""
detection.py — YOLOv8n Wrapper and Driver Tracker
==================================================

Public API
----------
- ``load_yolo(model_path) -> YOLO``
- ``detect_persons(model, frame, conf_thresh, person_class_id) -> list[tuple[ndarray, float]]``
- ``score_driver_candidate(box, frame_w, frame_h, drive_side, weights) -> float``
- ``iou(boxA, boxB) -> float``
- ``pad_box(box, padding, frame_w, frame_h) -> ndarray``
- ``DriverTracker(cfg, frame_w, frame_h)``
    - ``.update(person_boxes, frame_count) -> ndarray | None``
    - ``.reset()``
- ``detect_objects_in_roi(model, frame, driver_box, cfg) -> Tuple[Dict, List]``

All bounding boxes are ``np.ndarray`` shape ``(4,)`` with values
``[x1, y1, x2, y2]`` in pixel coordinates (float32).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "load_yolo",
    "detect_persons",
    "score_driver_candidate",
    "iou",
    "pad_box",
    "DriverTracker",
    "detect_objects_in_roi",
]

logger = logging.getLogger(__name__)


def load_yolo(model_path: str):
    """Load a YOLOv8 model. Downloads weights on first call if given a
    model name like ``"yolov8n.pt"``.

    Returns
    -------
    ultralytics.YOLO
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    logger.info("YOLOv8 model loaded: %s", model_path)
    return model


def detect_persons(
    model,
    frame: np.ndarray,
    conf_thresh: float = 0.50,
    person_class_id: int = 0,
    device: str | None = None,
    imgsz: int = 320,
) -> List[Tuple[np.ndarray, float]]:
    """Run YOLO and return all person detections above ``conf_thresh``.

    Returns
    -------
    list of (box, conf)
        box is np.ndarray([x1, y1, x2, y2], dtype=float32).
    """
    results = model(frame, device=device, verbose=False, imgsz=imgsz)
    persons: List[Tuple[np.ndarray, float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box_data in result.boxes:
            if int(box_data.cls[0]) == person_class_id:
                conf = float(box_data.conf[0])
                if conf >= conf_thresh:
                    xyxy = box_data.xyxy[0].cpu().numpy().astype(np.float32)
                    persons.append((xyxy, conf))
    return persons


def score_driver_candidate(
    box: np.ndarray,
    frame_w: int,
    frame_h: int,
    drive_side: str,
    weights: Dict[str, float],
) -> float:
    """Heuristic score for how likely a bounding box is the driver.

    Sub-scores
    ----------
    horizontal_position : 1.0 at the expected drive side, 0.0 at opposite.
    box_area            : box area / frame area (larger = closer = driver).
    vertical_anchor     : 1.0 if box centre is in upper 60% of frame.

    Returns
    -------
    float in [0, 1]
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = float(frame_w * frame_h)

    cx_norm = cx / float(frame_w)
    horiz_score = (1.0 - cx_norm) if drive_side == "left" else cx_norm

    area_score = float(np.clip(box_area / frame_area, 0.0, 1.0))

    cy_norm = cy / float(frame_h)
    vert_score = float(np.clip(1.0 - cy_norm / 0.6, 0.0, 1.0))

    return float(
        weights.get("horizontal_position", 0.35) * horiz_score
        + weights.get("box_area", 0.50) * area_score
        + weights.get("vertical_anchor", 0.15) * vert_score
    )


def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Intersection over Union of two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0.0 else 0.0


def pad_box(
    box: np.ndarray,
    padding: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Expand box by ``padding`` pixels on all sides, clamped to frame."""
    return np.array([
        max(0.0, box[0] - padding),
        max(0.0, box[1] - padding),
        min(float(frame_w), box[2] + padding),
        min(float(frame_h), box[3] + padding),
    ], dtype=np.float32)


class DriverTracker:
    """Tracks the driver's bounding box across frames using IoU matching.

    Parameters
    ----------
    cfg : dict
        The ``yolo:`` section from ``yolo_config.yaml``.
    frame_w, frame_h : int
        Full frame dimensions.
    """

    def __init__(self, cfg: Dict[str, Any], frame_w: int, frame_h: int) -> None:
        self._drive_side: str = cfg.get("drive_side", "left")
        self._weights: Dict[str, float] = cfg.get(
            "driver_score_weights",
            {"horizontal_position": 0.35, "box_area": 0.50, "vertical_anchor": 0.15},
        )
        self._iou_thresh: float = cfg.get("iou_track_thresh", 0.40)
        self._redetect_every: int = cfg.get("redetect_every_n_frames", 30)
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._tracked_box: Optional[np.ndarray] = None

    def update(
        self,
        person_boxes: List[Tuple[np.ndarray, float]],
        frame_count: int,
    ) -> Optional[np.ndarray]:
        """Update tracker with new person detections.

        Parameters
        ----------
        person_boxes : list of (box, conf) — output of detect_persons()
        frame_count : int — monotonically increasing from 1

        Returns
        -------
        np.ndarray shape (4,) or None
        """
        if not person_boxes:
            self._tracked_box = None
            return None

        force_rescore = (
            self._tracked_box is None
            or (frame_count % self._redetect_every == 0)
        )

        if not force_rescore and self._tracked_box is not None:
            for box, _conf in person_boxes:
                if iou(self._tracked_box, box) >= self._iou_thresh:
                    self._tracked_box = box
                    return self._tracked_box

        # Full re-score
        best_score = -1.0
        best_box: Optional[np.ndarray] = None
        for box, _conf in person_boxes:
            s = score_driver_candidate(
                box=box, frame_w=self._frame_w, frame_h=self._frame_h,
                drive_side=self._drive_side, weights=self._weights,
            )
            if s > best_score:
                best_score = s
                best_box = box

        self._tracked_box = best_box
        return self._tracked_box

    def reset(self) -> None:
        """Clear all tracking state."""
        self._tracked_box = None


def detect_objects_in_roi(
    model,
    frame: np.ndarray,
    driver_box: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run YOLO object detection inside the driver's ROI.

    Crops driver_box (+ roi_padding_px) from frame, runs YOLO, and returns
    the maximum confidence per event category defined in class_to_event,
    along with a list of detected objects (their labels, confidences,
    and bounding boxes mapped to the full frame).

    Parameters
    ----------
    model : ultralytics.YOLO
    frame : np.ndarray — full BGR uint8 frame
    driver_box : np.ndarray shape (4,) — [x1, y1, x2, y2]
    cfg : dict — the ``yolo:`` section from yolo_config.yaml

    Returns
    -------
    tuple of (event_scores, detected_objects)
        event_scores: dict[str, float]
        detected_objects: list of dicts with keys: "label", "conf", "box"
    """
    conf_thresh: float = cfg.get("object_conf_thresh", 0.45)
    padding: int = cfg.get("roi_padding_px", 30)
    device = cfg.get("device", None)
    imgsz = cfg.get("imgsz", 320)
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

    h, w = frame.shape[:2]
    # We still calculate padded to potentially filter later, but we run YOLO on full frame
    if driver_box is None:
        padded = [0, 0, w, h]
    else:
        padded = pad_box(driver_box, padding, w, h)
    
    # Bypass ROI cropping to ensure objects like phones held wide are never clipped out
    roi = frame
    x1, y1 = 0, 0
    if roi.size == 0:
        return event_scores, detected_objects

    # Pass conf=conf_thresh to YOLO to override its internal default of 0.25
    results = model(roi, device=device, verbose=False, imgsz=imgsz, conf=conf_thresh)
    for result in results:
        if result.boxes is None:
            continue
        for box_data in result.boxes:
            cls_id = int(box_data.cls[0])
            conf = float(box_data.conf[0])
            if cls_id in class_to_event and conf >= conf_thresh:
                evt = class_to_event[cls_id]
                event_scores[evt] = max(event_scores[evt], conf)

                r_xyxy = box_data.xyxy[0].cpu().numpy().astype(np.float32)
                full_xyxy = np.array([
                    r_xyxy[0] + x1,
                    r_xyxy[1] + y1,
                    r_xyxy[2] + x1,
                    r_xyxy[3] + y1
                ], dtype=np.float32)

                label = object_classes.get(cls_id, class_to_event.get(cls_id, str(cls_id)))
                detected_objects.append({
                    "label": label,
                    "conf": round(float(conf), 2),
                    "box": [float(v) for v in full_xyxy]
                })

    return event_scores, detected_objects
