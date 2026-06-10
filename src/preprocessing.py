"""Frame preprocessing: CLAHE on the Lab L channel plus optional adaptive gamma.

CLAHE runs only when mean brightness is below threshold. All funcs take and
return BGR uint8 arrays.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "preprocess_frame",
    "apply_clahe",
    "adaptive_gamma_correction",
]

logger = logging.getLogger(__name__)

# clahe on the L channel of Lab

def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """clahe on the L channel of Lab."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got shape "
            f"{getattr(frame, 'shape', None)}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {frame.dtype}")

    # convert BGR to Lab
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    logger.debug(
        "CLAHE applied: clip=%.1f, tile=%s, L_mean %.1f -> %.1f",
        clip_limit,
        tile_grid_size,
        l_channel.mean(),
        l_enhanced.mean(),
    )
    return result


# adaptive gamma correction

def adaptive_gamma_correction(
    frame: np.ndarray,
    gamma: float = 1.2,
    adaptive: bool = True,
) -> np.ndarray:
    """gamma correction; when adaptive, picks gamma from mean brightness."""
    if frame is None or frame.ndim != 3:
        raise ValueError("Expected 3-channel image")

    if adaptive:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()

        # target mean is 127 (midpoint of 0-255)
        target_mean = 127.0

        if mean_brightness < 1.0:
            # nearly black frame, use a strong brightening gamma
            gamma_eff = 0.5
        else:
            gamma_eff = np.log(target_mean / 255.0) / np.log(
                mean_brightness / 255.0 + 1e-7
            )
            gamma_eff = float(np.clip(gamma_eff, 0.5, 2.5))

        logger.debug(
            "Adaptive gamma: mean_brightness=%.1f -> gamma_eff=%.3f",
            mean_brightness,
            gamma_eff,
        )
    else:
        gamma_eff = gamma

    # lut for fast application
    inv_gamma = 1.0 / gamma_eff
    lut = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    corrected = cv2.LUT(frame, lut)
    return corrected


# preprocessing entry point

def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    clahe_clip: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8),
    brightness_threshold: float = 100.0,
    gamma: float = 1.2,
    adaptive_gamma: bool = True,
    apply_gamma: bool = True,
) -> np.ndarray:
    """resize, conditional clahe (dark frames only), then gamma."""
    if frame is None:
        raise ValueError("Input frame is None")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got shape {frame.shape}"
        )

    processed = frame.copy()

    if target_size is not None:
        w, h = target_size
        if (processed.shape[1], processed.shape[0]) != (w, h):
            processed = cv2.resize(
                processed, (w, h), interpolation=cv2.INTER_LINEAR
            )
            logger.debug("Resized frame to %dx%d", w, h)

    # clahe only on dark frames
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = float(l_channel.mean())

    if mean_l < brightness_threshold:
        processed = apply_clahe(
            processed,
            clip_limit=clahe_clip,
            tile_grid_size=tile_size,
        )
        logger.debug(
            "CLAHE applied: mean_L=%.1f < threshold=%.1f",
            mean_l,
            brightness_threshold,
        )
    else:
        logger.debug(
            "CLAHE skipped: mean_L=%.1f >= threshold=%.1f",
            mean_l,
            brightness_threshold,
        )

    if apply_gamma:
        processed = adaptive_gamma_correction(
            processed, gamma=gamma, adaptive=adaptive_gamma
        )

    return processed
