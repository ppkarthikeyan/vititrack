# U-Net (MobileNetV2 encoder) — first run, 2026-09-14

Train: 1,773 imgs · val 230 · test 208 (split by *filename* hash — see caveat). 320 px, 15 epochs, BCE+Dice, AdamW 3e-4 cosine.
Hardware: MacBook Air, Apple MPS, ~3.5 min/epoch. Best val Dice 0.925.

## Test set (n = 208)

| fitzpatrick_est (ITA) |  n | dice | precision | recall |
|---|---:|---:|---:|---:|
| I   | 60 | 0.933 | 0.923 | 0.964 |
| II  | 12 | 0.942 | 0.935 | 0.952 |
| III | 18 | 0.899 | 0.903 | 0.913 |
| IV  | 26 | 0.901 | 0.903 | 0.909 |
| V   | 47 | 0.946 | 0.974 | 0.936 |
| VI  | 45 | 0.934 | 0.961 | 0.919 |
| **all** | **208** | **0.930** (median 0.979) | | |

Colour baseline on the same data: 0.186. No skin-tone bin lags — the light-skin failure of the lightness rule is gone.

## Caveat — likely optimistic
`vitiligo-amw4r` contains 2,118 images derived from 1,133 originals via Roboflow augmentation. This run split by filename, so augmented copies of one photo can sit on both sides of the split. **v2 re-splits by source photo** (`source_key`) and is the number to report.
