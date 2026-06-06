"""
custom_detection.py — Custom Driver Tracker and Selfie Mode logic
==================================================================

Public API
----------
- ``custom_score_driver_candidate(box, frame_w, frame_h, drive_side, weights, selfie) -> float``
- ``CustomDriverTracker(cfg, frame_w, frame_h)``
    - ``.update(person_boxes, frame_count) -> ndarray | None``
    - ``.reset()``
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.detection import DriverTracker, score_driver_candidate, iou

__all__ = [
    "custom_score_driver_candidate",
    "CustomDriverTracker",
]


def custom_score_driver_candidate(
    box: np.ndarray,
    frame_w: int,
    frame_h: int,
    drive_side: str,
    weights: Dict[str, float],
    selfie: bool = False,
) -> float:
    """Heuristic score for how likely a bounding box is the driver.
    Behaves exactly like score_driver_candidate, but inverts drive_side
    when selfie is True.
    """
    effective_drive_side = drive_side
    if selfie:
        effective_drive_side = "right" if drive_side == "left" else "left"

    return score_driver_candidate(
        box=box,
        frame_w=frame_w,
        frame_h=frame_h,
        drive_side=effective_drive_side,
        weights=weights,
    )


class CustomDriverTracker(DriverTracker):
    """Tracks the driver's bounding box across frames, supporting selfie inversion."""

    def __init__(self, cfg: Dict[str, Any], frame_w: int, frame_h: int) -> None:
        super().__init__(cfg, frame_w, frame_h)
        self._selfie: bool = bool(cfg.get("selfie", False))

    def update(
        self,
        person_boxes: List[Tuple[np.ndarray, float]],
        frame_count: int,
    ) -> Optional[np.ndarray]:
        """Update tracker with new person detections.
        Overrides DriverTracker.update to temporarily toggle self._drive_side
        during the call to super().update() if selfie mode is enabled.
        """
        if self._selfie:
            original_side = self._drive_side
            self._drive_side = "right" if original_side == "left" else "left"
            try:
                return super().update(person_boxes, frame_count)
            finally:
                self._drive_side = original_side
        else:
            return super().update(person_boxes, frame_count)

