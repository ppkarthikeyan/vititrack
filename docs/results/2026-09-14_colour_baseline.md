# Colour-baseline evaluation — 2026-09-14

Backend: `colour` (YCrCb skin mask + L* threshold, delta_L = 18). No training.
Data: Roboflow `viti-main` (93 imgs) + `vitiligo-amw4r` (2,118 imgs), all splits, COCO segmentation masks.
Hardware: MacBook Air, CPU.

| fitzpatrick_est (ITA) |   n | dice | precision | recall |
|---|---:|---:|---:|---:|
| I   | 519 | 0.066 | 0.177 | 0.048 |
| II  | 155 | 0.067 | 0.278 | 0.048 |
| III | 191 | 0.115 | 0.347 | 0.081 |
| IV  | 245 | 0.143 | 0.426 | 0.113 |
| V   | 546 | 0.249 | 0.513 | 0.215 |
| VI  | 555 | 0.313 | 0.487 | 0.304 |
| **all** | **2211** | **0.186** (median 0.001) | | |

Interpretation
- Severe under-segmentation (recall ≪ precision); on >50 % of images the baseline finds nothing.
- Performance *increases* with darker estimated skin tone: lesion/skin lightness contrast is larger. Light skin (I–II) is the hard case for a pure-lightness rule.
- ITA bins on Roboflow images are approximate (unknown white balance, backgrounds).
- Conclusion: threshold tuning will not close this gap; a learned segmentation model is required. This table is the floor it must beat, per bin.

Reproduce: `python -m vititrack.eval.run_baseline`
