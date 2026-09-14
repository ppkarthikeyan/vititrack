"""Inference with the trained U-Net. Weights: models/unet_<encoder>.pt (gitignored; download or train)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch

_MODEL = None

def load(path: str | Path | None = None):
    global _MODEL
    if _MODEL is None:
        import segmentation_models_pytorch as smp
        path = Path(path) if path else next(Path("models").glob("unet_*.pt"))
        ck = torch.load(path, map_location="cpu")
        m = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(ck["state"]); m.eval()
        _MODEL = (m, ck["size"])
    return _MODEL

@torch.no_grad()
def segment_model(bgr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    model, size = load()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(r.transpose(2, 0, 1).astype(np.float32))[None]
    p = torch.sigmoid(model(x))[0, 0].numpy()
    return cv2.resize(p, (w, h)) > thr
