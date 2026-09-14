"""Show the worst-scoring test images: original | ground truth | prediction, plus a CSV.

Usage: python -m vititrack.eval.error_analysis [--n 20] [--csv data/eval/unet_mobilenet_v2_test_per_image.csv]
Writes data/eval/errors/<rank>_<dice>_<image>.jpg and data/eval/errors/worst.csv
Look at each panel and tag it in worst.csv: 'annotation' (GT looks wrong), 'model' (GT fine, prediction wrong),
'image' (blurry / not vitiligo / unusual), or 'ok'. That tagging decides what we fix next.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from vititrack.models.dataset import build_index
from vititrack.models.infer import segment_model

def overlay(bgr, mask, colour):
    out = bgr.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(colour)).astype(np.uint8)
    return out

def panel(bgr, gt, pred, title):
    h = 300; scale = h / bgr.shape[0]; w = int(bgr.shape[1] * scale)
    r = lambda im: cv2.resize(im, (w, h))
    a, b, c = r(bgr), r(overlay(bgr, gt, (0, 200, 0))), r(overlay(bgr, pred, (0, 0, 255)))
    strip = np.hstack([a, b, c])
    bar = np.full((28, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([bar, strip])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--csv", default="data/eval/unet_mobilenet_v2_test_per_image.csv")
    ap.add_argument("--out", default="data/eval/errors")
    a = ap.parse_args()
    df = pd.read_csv(a.csv).sort_values("dice").head(a.n)
    items = {p.name: (p, m) for p, m, s in build_index(sorted(Path("data/public").glob("roboflow_*"))) if s == "test"}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, r in enumerate(df.itertuples(), 1):
        p, gt = items[r.image]
        bgr = cv2.imread(str(p)); pred = segment_model(bgr)
        title = f"#{rank} dice={r.dice:.2f} tone={r.fitzpatrick_est} gt={gt.mean()*100:.1f}% pred={pred.mean()*100:.1f}%  [orig | GT green | pred red]"
        fn = f"{rank:02d}_{r.dice:.2f}_{p.stem[:30]}.jpg"
        cv2.imwrite(str(out / fn), panel(bgr, gt, pred, title))
        rows.append({"rank": rank, "image": r.image, "dice": round(r.dice, 3), "tone": r.fitzpatrick_est,
                     "gt_pct": round(gt.mean() * 100, 1), "pred_pct": round(pred.mean() * 100, 1), "file": fn, "tag": ""})
    pd.DataFrame(rows).to_csv(out / "worst.csv", index=False)
    print(f"wrote {len(rows)} panels to {out}/ — open them, then fill the 'tag' column in {out}/worst.csv")

if __name__ == "__main__":
    main()
