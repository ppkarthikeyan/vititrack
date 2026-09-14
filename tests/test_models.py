import json
from pathlib import Path
import numpy as np, cv2, pytest
from vititrack.models.dataset import build_index, VitiligoSegDataset

def test_dataset_shapes(tmp_path: Path):
    split = tmp_path / "roboflow_x" / "train"; split.mkdir(parents=True)
    imgs, anns = [], []
    for i in range(12):
        img = np.full((100, 120, 3), 150, np.uint8); name = f"i{i}.jpg"; cv2.imwrite(str(split / name), img)
        imgs.append({"id": i, "file_name": name, "height": 100, "width": 120})
        anns.append({"id": i, "image_id": i, "category_id": 1, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]], "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0})
    (split / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "vitiligo"}]}))
    items = build_index([tmp_path / "roboflow_x"])
    assert len(items) == 12 and {it[2] for it in items} <= {"train", "val", "test"}
    ds = VitiligoSegDataset(items, "train", size=64, augment=True)
    if len(ds):
        x, y = ds[0]
        assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64) and y.max() <= 1

def test_source_key_groups_roboflow_augmentations():
    from vititrack.models.dataset import source_key, _bucket
    a = "IMG_0123_jpg.rf.aaaa1111.jpg"; b = "IMG_0123_jpg.rf.bbbb2222.jpg"; c = "IMG_0124_jpg.rf.cccc3333.jpg"
    assert source_key(a) == source_key(b) == "img_0123"
    assert source_key(c) != source_key(a)
    assert _bucket(a) == _bucket(b)

def test_build_index_pseudo_and_holdout(tmp_path: Path):
    import cv2, json, numpy as np
    for name in ("roboflow_a", "roboflow_b"):
        split = tmp_path / name / "train"; split.mkdir(parents=True)
        img = np.full((50, 60, 3), 150, np.uint8); cv2.imwrite(str(split / "x.jpg"), img)
        (split / "_annotations.coco.json").write_text(json.dumps({"images": [{"id": 0, "file_name": "x.jpg", "height": 50, "width": 60}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[0, 0, 59, 0, 59, 49, 0, 49]], "bbox": [0, 0, 60, 50], "area": 3000, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "vitiligo"}]}))
    pseudo = tmp_path / "pseudo" / "roboflow_a"; pseudo.mkdir(parents=True)
    pm = np.zeros((50, 60), np.uint8); pm[10:20, 10:20] = 255; cv2.imwrite(str(pseudo / "x.png"), pm)
    items = build_index([tmp_path / "roboflow_a", tmp_path / "roboflow_b"], pseudo_dir=tmp_path / "pseudo", holdout="roboflow_b")
    by = {it[0].parent.parent.name: it for it in items}
    assert by["roboflow_a"][1].sum() == 100          # pseudo mask replaced the box
    assert by["roboflow_b"][2] == "test"              # holdout goes to test
    assert by["roboflow_b"][1].sum() == 3000          # real COCO mask kept
