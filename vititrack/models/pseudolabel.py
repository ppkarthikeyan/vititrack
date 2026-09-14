"""Turn bounding-box labels into pixel masks with box-prompted Segment Anything (SAM).

Why: `roboflow_vitiligo-amw4r` was annotated with boxes; Roboflow exported them as rectangular polygons.
Pixel Dice against rectangles is meaningless, but a box is a strong prompt for SAM, which returns the
white patch inside it well. We keep the SAM mask ∩ box as the pseudo-label.

Usage: python -m vititrack.models.pseudolabel --dataset data/public/roboflow_vitiligo-amw4r
Writes data/public/pseudo/<dataset-name>/<image>.png (0/255). ~1–2 s per image on Apple MPS.
Model: facebook/sam-vit-base (~375 MB, auto-downloaded from HF Hub on first run).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import torch

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_sam(name="facebook/sam-vit-base", dev=None):
    from transformers import SamModel, SamProcessor
    dev = dev or device()
    return SamModel.from_pretrained(name).to(dev).eval(), SamProcessor.from_pretrained(name), dev

@torch.no_grad()
def masks_for_boxes(model, proc, dev, rgb: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """boxes: [[x0,y0,x1,y1], ...] in pixels → union bool mask HxW."""
    from PIL import Image
    inputs = proc(Image.fromarray(rgb), input_boxes=[boxes], return_tensors="pt")
    inputs = {k: (v.float() if v.dtype == torch.float64 else v).to(dev) for k, v in inputs.items()}  # MPS has no float64
    out = model(**inputs, multimask_output=False)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                    inputs["reshaped_input_sizes"].cpu())[0]  # (n_boxes, 1, H, W)
    m = masks[:, 0].numpy().astype(bool)
    union = np.zeros(rgb.shape[:2], bool)
    for k, (x0, y0, x1, y1) in enumerate(boxes):   # clip each mask to its own box
        clip = np.zeros_like(union); clip[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
        union |= m[k] & clip
    return union

def run(dataset: Path, out_root: Path, limit: int | None = None, min_box_px: int = 16):
    model, proc, dev = load_sam(); print("device", dev)
    out_dir = out_root / dataset.name; out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for split in ("train", "valid", "test"):
        ann = dataset / split / "_annotations.coco.json"
        if not ann.exists(): continue
        coco = json.loads(ann.read_text())
        by_img: dict[int, list] = {}
        for a in coco["annotations"]: by_img.setdefault(a["image_id"], []).append(a["bbox"])
        for im in coco["images"]:
            if limit and done >= limit: break
            dst = out_dir / (Path(im["file_name"]).stem + ".png")
            if dst.exists(): done += 1; continue
            src = dataset / split / im["file_name"]
            boxes = [[x, y, x + w, y + h] for x, y, w, h in by_img.get(im["id"], []) if w >= min_box_px and h >= min_box_px]
            if not boxes or not src.exists(): skipped += 1; continue
            bgr = cv2.imread(str(src)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                mask = masks_for_boxes(model, proc, dev, rgb, boxes)
            except Exception as e:  # MPS occasionally fails on odd sizes → retry on CPU
                print("retry on cpu:", src.name, str(e)[:80])
                model_cpu = model.to("cpu"); mask = masks_for_boxes(model_cpu, proc, torch.device("cpu"), rgb, boxes); model.to(dev)
            cv2.imwrite(str(dst), (mask * 255).astype(np.uint8)); done += 1
            if done % 50 == 0: print(f"{done} masks written")
    print(f"done: {done} masks in {out_dir} (skipped {skipped} with no usable box)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/public/roboflow_vitiligo-amw4r")
    ap.add_argument("--out", default="data/public/pseudo")
    ap.add_argument("--limit", type=int, default=None, help="stop after N images (smoke test)")
    a = ap.parse_args()
    run(Path(a.dataset), Path(a.out), a.limit)
