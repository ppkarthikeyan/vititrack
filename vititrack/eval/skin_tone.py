"""Estimate skin tone from an image via the Individual Typology Angle (ITA).

ITA = arctan((L* - 50) / b*) in degrees, computed on pigmented (non-lesion) skin pixels.
Standard bins (Del Bino & Bernerd): >55 very light, 41–55 light, 28–41 intermediate,
10–28 tan, -30–10 brown, < -30 dark. We map them to Fitzpatrick-like groups I–VI.
Camera white balance makes this approximate — good enough for stratified reporting.
"""
from __future__ import annotations
import numpy as np
import cv2

ITA_BINS = [(55, "I"), (41, "II"), (28, "III"), (10, "IV"), (-30, "V"), (-999, "VI")]

def ita_degrees(bgr: np.ndarray, skin: np.ndarray, lesion: np.ndarray | None = None) -> float:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    b = lab[..., 2] - 128.0
    sel = skin.copy()
    if lesion is not None:
        sel &= ~lesion
    if sel.sum() < 100:
        return float("nan")
    Lm, bm = float(np.median(L[sel])), float(np.median(b[sel]))
    return float(np.degrees(np.arctan2(Lm - 50.0, bm if abs(bm) > 1e-3 else 1e-3)))

def fitzpatrick_group(ita: float) -> str:
    if np.isnan(ita):
        return "unknown"
    for thr, name in ITA_BINS:
        if ita > thr:
            return name
    return "VI"
