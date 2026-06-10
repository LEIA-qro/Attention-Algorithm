"""Driver tracker with selfie-mode (mirrored drive side) support."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.detection import DriverTracker


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
        """Track the driver across detections, mirroring drive side in selfie mode."""
        if self._selfie:
            original_side = self._drive_side
            self._drive_side = "right" if original_side == "left" else "left"
            try:
                return super().update(person_boxes, frame_count)
            finally:
                self._drive_side = original_side
        else:
            return super().update(person_boxes, frame_count)

