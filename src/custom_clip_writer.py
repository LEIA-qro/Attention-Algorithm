"""
custom_clip_writer.py — Custom Clip Writer with pre_frames support
==================================================================

Public API
----------
- ``CustomClipWriter(output_dir, fps, pre_buffer_seconds, post_buffer_seconds, fourcc)``
    - ``.save_clip_async(event_type, confidence, trigger_timestamp_s, post_frames, pre_frames=None) -> Path``
    - ``.save_clip(event_type, confidence, trigger_timestamp_s, post_frames, pre_frames=None) -> Path | None``
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.clip_writer import ClipWriter

__all__ = ["CustomClipWriter"]


class CustomClipWriter(ClipWriter):
    """Subclass of ClipWriter supporting custom pre_frames."""

    def _prepare_frames(
        self,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Prepare frames by copying and merging pre_frames and post_frames."""
        if pre_frames is None:
            pre_frames_copied = [f.copy() for f, _ in self._ring_buffer]
        else:
            pre_frames_copied = [f.copy() for f in pre_frames]

        return pre_frames_copied + [f.copy() for f in post_frames]

    def save_clip_async(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> Path:
        """Asynchronously write (pre_frames or ring-buffer frames) + post_frames to an MP4 file.

        Returns
        -------
        Path where the file will be written.
        """
        all_frames = self._prepare_frames(post_frames, pre_frames)
        out_path = self.get_clip_path(event_type, confidence, trigger_timestamp_s)
        self._queue.put((event_type, confidence, trigger_timestamp_s, all_frames))
        return out_path

    def save_clip(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> Optional[Path]:
        """Write (pre_frames or ring-buffer frames) + post_frames to an MP4 file.

        Returns
        -------
        Path of written file, or None if writing failed.
        """
        all_frames = self._prepare_frames(post_frames, pre_frames)
        return self._write_clip_sync(event_type, confidence, trigger_timestamp_s, all_frames)
