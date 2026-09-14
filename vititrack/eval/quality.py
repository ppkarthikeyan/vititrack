"""Automatic label-quality flags for COCO segmentation masks.

Flags (any → 'suspect'):
  boxy      : mask is essentially an axis-aligned rectangle (area / bbox_area > 0.92) — bounding boxes exported as polygons
  whole     : mask covers > 60 % of the image — whole-image labels
  empty     : no lesion annotation at all
  tiny      : mask covers < 0.05 % of the image — likely a stray click
Usage: python -m vititrack.eval.quality            → data/eval/quality_flags.csv + summary
Then training/eval can exclude suspects via --clean.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from .coco import iter_dataset
from vititrack.models.dataset import source_key, _bucket

def mask_flags(mask: np.ndarray) -> dict:
    h, w = mask.shape; frac = float(mask.mean())
    flags = {"boxy": False, "whole": frac > 0.60, "empty": frac == 0.0, "tiny": 0 < frac < 0.0005, "rectangularity": 0.0}
    if frac > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        # rectangularity of the largest component
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        rect = area / max(bw * bh, 1)
        flags["rectangularity"] = float(rect)
        flags["boxy"] = rect > 0.92 and area > 400
    flags["suspect"] = bool(flags["boxy"] or flags["whole"] or flags["empty"] or flags["tiny"])
    return flags

def build_flags(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for split, p, m in iter_dataset(root):
            f = mask_flags(m)
            rows.append({"dataset": root.name, "image": p.name, "source": source_key(p.name), "bucket": _bucket(p.name),
                         "gt_frac": float(m.mean()), **f})
    return pd.DataFrame(rows)

def load_flags(path: Path = Path("data/eval/quality_flags.csv")) -> set[str]:
    """Return the set of image filenames flagged suspect (empty set if no flags file)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df.loc[df.suspect, "image"])

if __name__ == "__main__":
    roots = sorted(Path("data/public").glob("roboflow_*"))
    df = build_flags(roots)
    out = Path("data/eval"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "quality_flags.csv", index=False)
    print(f"images: {len(df)}  suspect: {int(df.suspect.sum())} ({df.suspect.mean()*100:.1f}%)")
    print(df[["boxy", "whole", "empty", "tiny"]].sum().to_string())
    print("\nsuspect by bucket:\n" + df.groupby("bucket").suspect.agg(["size", "sum"]).to_string())
    print("\nunique source photos per bucket (clean):\n" + df[~df.suspect].groupby("bucket").source.nunique().to_string())
