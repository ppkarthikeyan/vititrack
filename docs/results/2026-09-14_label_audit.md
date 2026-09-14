# Label audit — 2026-09-14

`python -m vititrack.eval.quality` on the two Roboflow sets:

| dataset | images | rectangularity > 0.99 | median rect. | verdict |
|---|---:|---:|---:|---|
| roboflow_viti-main | 93 | 0 % | 0.72 | real traced polygons |
| roboflow_vitiligo-amw4r | 2,118 | **84 %** | **1.00** | **bounding boxes exported as polygons** |

Consequences
- Every pixel-Dice number computed on amw4r (colour baseline 0.186, U-Net v1 0.930, v2 0.831) measured agreement with rectangles, not lesions. Retracted as segmentation results; kept in docs/results as a record of the pitfall.
- Cleaning to non-box masks leaves 87 training photos → U-Net Dice 0.70 on 17 test photos (too small to report).

Remedy (step 9)
- Box-prompted SAM (`facebook/sam-vit-base`) converts each amw4r box into a pixel mask (`vititrack/models/pseudolabel.py`).
- Train on pseudo-masks; **test only on viti-main's 93 real masks** (`--holdout roboflow_viti-main`), so the reported number never touches a pseudo-label.
- Lesson for the paper: audit label geometry before training. A rectangularity histogram is a 10-line check that would have saved a day.
