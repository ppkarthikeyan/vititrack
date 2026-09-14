"""Keep only pseudo-masks whose prompt boxes were small enough for SAM to localise.

Whole-image boxes give SAM nothing to localise → it returns all skin. We keep an image only if
every box covers < max_box_frac of the image and the union of boxes < max_total_frac.
Writes <pseudo_dir>/<dataset>/keep.txt (one image stem per line); build_index honours it.

Usage: python -m vititrack.models.pseudo_filter --dataset data/public/roboflow_vitiligo-amw4r
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/public/roboflow_vitiligo-amw4r")
    ap.add_argument("--pseudo", default="data/public/pseudo")
    ap.add_argument("--max-box-frac", type=float, default=0.35)
    ap.add_argument("--max-total-frac", type=float, default=0.50)
    a = ap.parse_args()
    ds = Path(a.dataset); pdir = Path(a.pseudo) / ds.name
    keep, total = [], 0
    for split in ("train", "valid", "test"):
        ann = ds / split / "_annotations.coco.json"
        if not ann.exists(): continue
        coco = json.loads(ann.read_text())
        by_img: dict[int, list] = {}
        for x in coco["annotations"]: by_img.setdefault(x["image_id"], []).append(x["bbox"])
        for im in coco["images"]:
            total += 1
            W = im["width"] * im["height"]; boxes = by_img.get(im["id"], [])
            if not boxes or not (pdir / (Path(im["file_name"]).stem + ".png")).exists(): continue
            fr = [w * h / W for _, _, w, h in boxes]
            if max(fr) < a.max_box_frac and sum(fr) < a.max_total_frac:
                keep.append(Path(im["file_name"]).stem)
    (pdir / "keep.txt").write_text("\n".join(keep))
    print(f"kept {len(keep)} / {total} images ({len(keep)/max(total,1)*100:.0f}%) → {pdir/'keep.txt'}")

if __name__ == "__main__":
    main()
