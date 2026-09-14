"""Train a U-Net (pretrained encoder) for vitiligo lesion segmentation.

Usage: python -m vititrack.models.train --epochs 15 --size 320 --encoder mobilenet_v2
Writes models/unet_<encoder>.pt and data/eval/model_summary.md (stratified by skin tone).
Runs on Apple MPS, CUDA, or CPU (slow).
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from .dataset import build_index, VitiligoSegDataset
from vititrack.eval.skin_tone import ita_degrees, fitzpatrick_group
from vititrack.scoring.segment import skin_mask_ycrcb
import cv2

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); s = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (s + eps)).mean()

@torch.no_grad()
def predict_mask(model, bgr, size, dev):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(r.transpose(2, 0, 1).astype(np.float32))[None].to(dev)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h)) > 0.5

def evaluate(model, items, size, dev, out_dir: Path, tag: str):
    rows = []
    for img_path, gt, _ in items:
        bgr = cv2.imread(str(img_path)); pred = predict_mask(model, bgr, size, dev)
        s = pred.sum() + gt.sum(); d = 1.0 if s == 0 else 2.0 * (pred & gt).sum() / s
        ita = ita_degrees(bgr, skin_mask_ycrcb(bgr), gt)
        rows.append({"image": img_path.name, "dice": d, "fitzpatrick_est": fitzpatrick_group(ita),
                     "precision": (pred & gt).sum() / max(pred.sum(), 1), "recall": (pred & gt).sum() / max(gt.sum(), 1)})
    df = pd.DataFrame(rows); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{tag}_per_image.csv", index=False)
    md = [f"# {tag}", "", f"Images: {len(df)} | mean Dice {df.dice.mean():.3f} | median {df.dice.median():.3f}", "",
          df.groupby("fitzpatrick_est").agg(n=("dice", "size"), dice=("dice", "mean"),
                                            precision=("precision", "mean"), recall=("recall", "mean")).round(3).to_markdown(), ""]
    (out_dir / f"{tag}_summary.md").write_text("\n".join(md)); print("\n".join(md))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pseudo", default=None, help="dir of SAM pseudo-masks, e.g. data/public/pseudo")
    ap.add_argument("--holdout", default=None, help="dataset name used ONLY for test, e.g. roboflow_viti-main")
    ap.add_argument("--clean", action="store_true", help="exclude images flagged suspect in data/eval/quality_flags.csv")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet encoder weights (offline smoke tests)")
    a = ap.parse_args()
    roots = [Path(p) for p in a.data] if a.data else sorted(Path("data/public").glob("roboflow_*"))
    items = build_index(roots, pseudo_dir=Path(a.pseudo) if a.pseudo else None, holdout=a.holdout)
    from .dataset import source_key
    if a.clean:
        from vititrack.eval.quality import load_flags
        bad = load_flags(); before = len(items)
        items = [it for it in items if it[0].name not in bad]
        print(f"--clean: dropped {before - len(items)} suspect images")
    n = {s: sum(1 for it in items if it[2] == s) for s in ("train", "val", "test")}
    src = {s: len({source_key(it[0].name) for it in items if it[2] == s}) for s in n}
    print("split sizes (images)", n, "| unique source photos", src)
    dev = device(); print("device", dev)
    tr = DataLoader(VitiligoSegDataset(items, "train", a.size, augment=True), batch_size=a.batch, shuffle=True, num_workers=0)
    va = DataLoader(VitiligoSegDataset(items, "val", a.size), batch_size=a.batch, num_workers=0)
    model = smp.Unet(a.encoder, encoder_weights=None if a.no_pretrained else "imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, out = -1.0, Path("models"); out.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            logits = model(x); loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        sched.step()
        model.eval(); ds = []
        with torch.no_grad():
            for x, y in va:
                p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
                inter = (p * y).sum((1, 2, 3)); s = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
                ds += ((2 * inter + 1) / (s + 1)).tolist()
        vd = float(np.mean(ds))
        print(f"epoch {ep+1}/{a.epochs} loss {tl/len(tr):.3f} val_dice {vd:.3f} ({time.time()-t0:.0f}s)")
        if vd > best:
            best = vd; torch.save({"encoder": a.encoder, "size": a.size, "state": model.state_dict()}, out / f"unet_{a.encoder}{a.tag}.pt")
    print("best val dice", round(best, 3))
    model.load_state_dict(torch.load(out / f"unet_{a.encoder}{a.tag}.pt", map_location=dev)["state"]); model.eval()
    test = [it for it in items if it[2] == "test"]
    seen, test_unique = set(), []
    for it in test:  # one copy per source photo so augmented duplicates don't over-weight the metric
        k = source_key(it[0].name)
        if k not in seen:
            seen.add(k); test_unique.append(it)
    print(f"test: {len(test)} images → {len(test_unique)} unique source photos")
    evaluate(model, test_unique, a.size, dev, Path("data/eval"), f"unet_{a.encoder}{a.tag}_test")

if __name__ == "__main__":
    main()
