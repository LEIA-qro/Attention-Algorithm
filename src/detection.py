"""YOLOv8n person detection and IoU-based driver tracking.

Boxes are np.ndarray shape (4,) [x1, y1, x2, y2] in pixel coordinates (float32).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_yolo(model_path: str):
    """Load a YOLOv8 model, downloading weights on first use if given a name like "yolov8n.pt"."""
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
    """Run YOLO and return (box, conf) for every person detection above conf_thresh."""
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
    """Score 0-1 for how likely a box is the driver, weighting horizontal position, area, and vertical anchor."""
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
    """Tracks the driver's box across frames by IoU matching, re-scoring candidates periodically."""

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
        """Update the tracked box from the latest person detections; returns the driver box or None."""
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
    """Run YOLO in the driver's ROI; returns max confidence per event plus detected objects in full-frame coords."""
    conf_thresh: float = cfg.get("object_conf_thresh", 0.45)
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

    # Run YOLO on the full frame so wide-held objects like phones are never clipped by an ROI crop.
    roi = frame
    x1, y1 = 0, 0
    if roi.size == 0:
        return event_scores, detected_objects

    # Override YOLO's internal default conf of 0.25.
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
