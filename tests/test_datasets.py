import pytest
import torch
import numpy as np
from datasets.custom_dataset import CustomHandDataset

def test_custom_dataset_synthetic(tmp_path):
    import json
    import cv2
    
    # Create mock images and labels directories
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "sample_0.png"), img)
    
    ann_data = {
        "image_file": "sample_0.png",
        "pose_name": "fist",
        "keypoints_2d": [[128.0, 128.0]] * 21,
        "keypoints_3d": [[0.0, 0.0, 0.0]] * 21,
        "handedness": "Droite"
    }
    with open(labels_dir / "sample_0.json", "w") as f:
        json.dump(ann_data, f)
        
    ds = CustomHandDataset(custom_dir=str(tmp_path))
    assert len(ds) == 1
    
    sample = ds[0]
    assert "image" in sample
    assert "coords_2d" in sample
    assert "joints_3d" in sample
    
    assert sample["image"].shape == (3, 256, 256)
    assert sample["coords_2d"].shape == (21, 2)
    assert sample["joints_3d"].shape == (21, 3)
    
    # Coords 2D must be normalized in [0, 1]
    assert (sample["coords_2d"] >= 0.0).all() and (sample["coords_2d"] <= 1.0).all()
