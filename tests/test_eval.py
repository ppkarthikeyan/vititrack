import json
from pathlib import Path
import numpy as np, cv2
from vititrack.eval.run_baseline import evaluate

def _make_coco(root: Path, n=3):
    split = root / "train"; split.mkdir(parents=True)
    images, anns = [], []
    for i in range(n):
        h, w = 200, 300
        img = np.zeros((h, w, 3), np.uint8); img[:] = (120, 160, 210)
        x0, y0, x1, y1 = 50 + 10 * i, 50, 150 + 10 * i, 150
        img[y0:y1, x0:x1] = (235, 240, 245)
        name = f"img{i}.jpg"; cv2.imwrite(str(split / name), img)
        images.append({"id": i, "file_name": name, "height": h, "width": w})
        anns.append({"id": i, "image_id": i, "category_id": 1,
                     "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]], "area": 100 * 100, "bbox": [x0, y0, 100, 100], "iscrowd": 0})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 0, "name": "vitiligo-seg"}, {"id": 1, "name": "vitiligo"}]}
    (split / "_annotations.coco.json").write_text(json.dumps(coco))

def test_evaluate_synthetic(tmp_path: Path):
    root = tmp_path / "roboflow_demo"; _make_coco(root)
    df = evaluate([root], out_dir=tmp_path / "eval")
    assert len(df) == 3
    assert df.dice.mean() > 0.9
    assert (tmp_path / "eval" / "baseline_summary.md").exists()
    assert set(df.fitzpatrick_est) <= {"I", "II", "III", "IV", "V", "VI", "unknown"}
