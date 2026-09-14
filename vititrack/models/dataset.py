"""Torch dataset over the Roboflow COCO exports, with a fixed 80/10/10 split by image hash."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from vititrack.eval.coco import iter_dataset

def source_key(name: str) -> str:
    """Group Roboflow augmented copies with their original.

    Roboflow names files '<original-stem>_jpg.rf.<hash>.jpg'; augmented copies share the stem
    but carry a different hash. Splitting on the stem keeps all copies of one photo in one split,
    which prevents train/test leakage.
    """
    stem = name.split(".rf.")[0]
    for suf in ("_jpg", "_jpeg", "_png", "_JPG", "_PNG"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem.lower()

def _bucket(name: str) -> str:
    h = int(hashlib.md5(source_key(name).encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")

def build_index(roots: list[Path], pseudo_dir: Path | None = None, holdout: str | None = None) -> list[tuple[Path, np.ndarray, str]]:
    """(image_path, mask, bucket) for every annotated image.

    pseudo_dir: if given, a mask at <pseudo_dir>/<dataset>/<stem>.png replaces the COCO mask (box→SAM pseudo-labels).
                Images from a dataset in pseudo_dir with no pseudo mask are skipped (their COCO mask is a box).
    holdout:    dataset name whose images ALL go to the 'test' bucket (e.g. 'roboflow_viti-main' — real traced masks).
    """
    items = []
    for root in roots:
        pdir = (Path(pseudo_dir) / root.name) if pseudo_dir else None
        keep = None
        if pdir is not None and (pdir / "keep.txt").exists():
            keep = set((pdir / "keep.txt").read_text().split())
        for _split, img_path, mask in iter_dataset(root):
            if pdir is not None and pdir.exists():
                pm = pdir / (img_path.stem + ".png")
                if not pm.exists() or (keep is not None and img_path.stem not in keep):
                    continue
                mask = cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE) > 127
            bucket = "test" if holdout and root.name == holdout else _bucket(img_path.name)
            items.append((img_path, mask, bucket))
    return items

class VitiligoSegDataset(Dataset):
    def __init__(self, items, split: str, size: int = 320, augment: bool = False):
        self.items = [it for it in items if it[2] == split]
        self.size, self.augment = size, augment
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, mask, _ = self.items[i]
        bgr = cv2.imread(str(img_path)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5: rgb, m = rgb[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5: rgb, m = rgb[::-1], m[::-1]
            # mild brightness/contrast jitter so the model doesn't learn one lighting setup
            a, b = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
            rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(m)).float().unsqueeze(0)
        return x, y
