"""Depigmentation segmentation.

Two backends:
  - "colour": classical baseline. Skin detection in YCrCb, then depigmented skin =
    skin pixels whose lightness (L* in CIELAB) is well above the pigmented-skin mode.
    Works on any machine, no weights. Expect it to be rough; it is the floor, not the goal.
  - "model": placeholder for a learned model (SegFormer / U-Net++ / fine-tuned SAM).
    Raise NotImplementedError until weights exist.

Both return (lesion_mask, skin_mask) as bool arrays.
"""
from __future__ import annotations
import numpy as np
import cv2

def skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """Broad skin detector in YCrCb; tuned to include dark skin (Fitzpatrick V–VI)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    m = cv2.inRange(ycrcb, lower, upper) > 0
    # depigmented patches are very pale and may fall outside the Cr range; add high-L, low-sat pixels
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pale = (hsv[..., 2] > 150) & (hsv[..., 1] < 90)
    m = m | pale
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    return m

def lesion_mask_colour(bgr: np.ndarray, skin: np.ndarray, delta_L: float = 18.0) -> np.ndarray:
    """Depigmented = skin pixels whose L* exceeds the pigmented-skin mode by delta_L."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) * (100.0 / 255.0)
    vals = L[skin]
    if vals.size == 0:
        return np.zeros_like(skin)
    hist, edges = np.histogram(vals, bins=50, range=(0, 100))
    mode_L = edges[int(np.argmax(hist))]
    lesion = skin & (L > mode_L + delta_L)
    lesion = cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return lesion

def segment(bgr: np.ndarray, backend: str = "colour") -> tuple[np.ndarray, np.ndarray]:
    if backend == "colour":
        skin = skin_mask_ycrcb(bgr)
        return lesion_mask_colour(bgr, skin), skin
    if backend == "model":
        raise NotImplementedError("learned segmentation backend not available yet — see docs/ROADMAP.md")
    raise ValueError(backend)
