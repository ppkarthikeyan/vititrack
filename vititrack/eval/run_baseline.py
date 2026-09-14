"""Evaluate the colour-baseline segmentation on public COCO datasets, stratified by skin tone.

Usage:  python -m vititrack.eval.run_baseline [data/public/roboflow_*  ...]
Writes: data/eval/baseline_per_image.csv and data/eval/baseline_summary.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from vititrack.scoring.segment import segment
from .coco import iter_dataset
from .skin_tone import ita_degrees, fitzpatrick_group

def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s

def evaluate(roots: list[Path], out_dir: Path = Path("data/eval"), backend: str = "colour") -> pd.DataFrame:
    rows = []
    for root in roots:
        for split, img_path, gt in iter_dataset(root):
            bgr = cv2.imread(str(img_path))
            if bgr is None or bgr.shape[:2] != gt.shape:
                continue
            pred, skin = segment(bgr, backend)
            ita = ita_degrees(bgr, skin, gt)
            rows.append({
                "dataset": root.name, "split": split, "image": img_path.name,
                "gt_frac": float(gt.mean()), "pred_frac": float(pred.mean()),
                "dice": dice(pred, gt),
                "precision": float((pred & gt).sum() / max(pred.sum(), 1)),
                "recall": float((pred & gt).sum() / max(gt.sum(), 1)),
                "ita": ita, "fitzpatrick_est": fitzpatrick_group(ita),
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baseline_per_image.csv", index=False)
    lines = [f"# Colour-baseline evaluation ({backend})", "",
             f"Images: {len(df)}  |  mean Dice: {df.dice.mean():.3f}  |  median Dice: {df.dice.median():.3f}", "",
             "## By estimated skin tone (ITA → Fitzpatrick-like)", "",
             df.groupby("fitzpatrick_est").agg(n=("dice", "size"), dice=("dice", "mean"),
                                               precision=("precision", "mean"), recall=("recall", "mean")).round(3).to_markdown(),
             "", "## By dataset", "",
             df.groupby("dataset").agg(n=("dice", "size"), dice=("dice", "mean")).round(3).to_markdown(), ""]
    (out_dir / "baseline_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return df

if __name__ == "__main__":
    roots = [Path(p) for p in sys.argv[1:]] or sorted(Path("data/public").glob("roboflow_*"))
    evaluate(roots)
