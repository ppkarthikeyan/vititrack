# U-Net (MobileNetV2) trained on SAM pseudo-labels, tested on real traced masks — 2026-09-14  ← REFERENCE

**This is the first result in the project measured against real pixel ground truth.** Everything before it
(colour baseline 0.186, U-Net v1 0.930, v2 0.831) was scored against bounding boxes and is retracted — see `2026-09-14_label_audit.md`.

## Setup
- Train/val: `roboflow_vitiligo-amw4r` boxes → box-prompted SAM (`facebook/sam-vit-base`) → pixel masks;
  kept only images whose boxes covered < 35 % of the frame (`pseudo_filter`): 608 images = **99 unique source photos** (+ augmented copies), val 10.
- Test: **all of `roboflow_viti-main`** — 109 unique source photos with hand-traced polygons, never used in training, never pseudo-labelled.
- 320 px, 20 epochs, BCE + Dice, AdamW 3e-4 cosine, MacBook Air MPS (~1 min/epoch). Best val Dice 0.713 (val is pseudo-labelled; treat as noisy).

## Test (n = 109 real photos)
| fitzpatrick_est (ITA) |  n | dice | precision | recall |
|---|---:|---:|---:|---:|
| I   | 18 | 0.599 | 0.742 | 0.621 |
| II  |  7 | 0.529 | 0.637 | 0.552 |
| III | 10 | 0.574 | 0.688 | 0.539 |
| IV  | 19 | 0.667 | 0.750 | 0.681 |
| V   | 34 | 0.609 | 0.765 | 0.635 |
| VI  | 21 | 0.560 | 0.860 | 0.452 |
| **all** | **109** | **0.599** (median 0.664) | 0.76 | 0.60 |

## Reading it
- Precision > recall everywhere → the model **under-segments**; worst on the darkest bin (VI recall 0.45, precision 0.86): it finds the bright core of a patch and misses the graded edge. Threshold sweep (0.3–0.4) is the first cheap thing to try.
- No strong skin-tone gradient in Dice (0.53–0.67, bins of 7–34) — but recall clearly drops with darker tone. That's the pattern to track as data grows.
- Ceiling is data, not architecture: ~99 distinct training photos with machine-made masks. Every real traced photo added is worth more than any model change.

## Next
1. Threshold sweep on the saved model (no retraining) → pick by val Dice, report test.
2. Fine-tune on viti-main with 5-fold CV (real masks, small) and compare — but then viti-main is no longer a clean test set; keep both numbers separate.
3. Real data: UC Davis masks (email sent?), clinic partnership, own protocol capture. This is the roadmap's centre of gravity now.

Reproduce: `pseudolabel` → `pseudo_filter` → `train --pseudo data/public/pseudo --holdout roboflow_viti-main --tag _pseudo`
