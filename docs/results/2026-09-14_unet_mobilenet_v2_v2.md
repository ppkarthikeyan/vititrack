# U-Net (MobileNetV2 encoder) — v2, source-photo split, 2026-09-14  ← REFERENCE RESULT

Split by `source_key` (Roboflow augmented copies of one photo stay together). 320 px, 15 epochs, BCE+Dice,
AdamW 3e-4 cosine. MacBook Air, Apple MPS, ~3 min/epoch. Best val Dice 0.898.

## Test set (n = 253 images)

| fitzpatrick_est (ITA) |  n | dice | precision | recall |
|---|---:|---:|---:|---:|
| I   | 90 | 0.804 | 0.819 | 0.817 |
| II  | 13 | 0.899 | 0.918 | 0.909 |
| III | 22 | 0.770 | 0.769 | 0.821 |
| IV  | 17 | 0.825 | 0.915 | 0.793 |
| V   | 71 | 0.823 | 0.902 | 0.804 |
| VI  | 40 | 0.921 | 0.946 | 0.905 |
| **all** | **253** | **0.831** (median 0.915) | | |

## Comparison
| method | split | Dice |
|---|---|---:|
| colour baseline | — | 0.186 |
| U-Net v1 | by filename (leaky) | 0.930 |
| **U-Net v2** | **by source photo** | **0.831** |

Leakage inflated v1 by ~0.10 — always split by source photo on Roboflow exports.

## Observations
- Light/intermediate tones (I, III) lag darkest (VI) by 0.12–0.15 Dice. Bins II–IV are small (n ≤ 22); treat as a hypothesis, not a finding, until the eval set grows.
- On IV–VI the model under-segments (recall 0.79–0.80 < precision 0.90–0.95).
- Median ≫ mean → a tail of badly-failed images. Inspect the lowest-Dice 20 (`data/eval/unet_mobilenet_v2_test_per_image.csv`) before changing the model.

## Next
1. Error analysis on the worst 20 test images (annotation errors vs. model errors).
2. Cheap gains to try: 384 px, 25 epochs, `efficientnet-b0` or `resnet34` encoder, threshold sweep (0.3–0.6).
3. Wire `backend="model"` into `score-vasi` and run on a real protocol session.
