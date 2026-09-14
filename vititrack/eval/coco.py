"""Load Roboflow COCO-segmentation exports as (image_path, lesion_mask) pairs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
import numpy as np
import cv2

LESION_CLASS_HINTS = ("vitiligo", "lesion", "patch", "depig", "white")

def _is_lesion_category(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in LESION_CLASS_HINTS) and "healthy" not in n and "trace" not in n

def iter_coco_split(split_dir: Path) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield (image_path, bool mask HxW) for every annotated image in a Roboflow split folder."""
    split_dir = Path(split_dir)
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return
    coco = json.loads(ann_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    lesion_cat_ids = {cid for cid, n in cats.items() if _is_lesion_category(n)}
    if not lesion_cat_ids:  # single-class datasets often name the class after the project
        lesion_cat_ids = {cid for cid, n in cats.items() if n.lower() not in ("background", "healthy", "trace")}
    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        if a["category_id"] in lesion_cat_ids:
            by_image.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        img_path = split_dir / im["file_name"]
        if not img_path.exists():
            continue
        h, w = im["height"], im["width"]
        mask = np.zeros((h, w), np.uint8)
        for a in by_image.get(im["id"], []):
            seg = a.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        yield img_path, mask.astype(bool)

def iter_dataset(root: Path) -> Iterator[tuple[str, Path, np.ndarray]]:
    for split in ("train", "valid", "test"):
        for img_path, mask in iter_coco_split(Path(root) / split):
            yield split, img_path, mask
