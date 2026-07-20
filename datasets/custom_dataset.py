# hand_bone_tracker/datasets/custom_dataset.py
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from PIL import Image
import numpy as np

class CustomHandDataset(Dataset):
    """
    Dataset to load custom hand tracking samples saved via the dashboard.
    Loads images from raw/custom/images/ and JSON files from raw/custom/labels/.
    """
    def __init__(self, custom_dir='data/raw/custom', transform=None):
        self.custom_dir = Path(custom_dir)
        self.transform = transform
        
        self.labels_dir = self.custom_dir / "labels"
        self.images_dir = self.custom_dir / "images"
        
        self.samples = []
        if self.labels_dir.exists():
            for json_path in self.labels_dir.glob("*.json"):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    # Resolve image file path
                    img_name = Path(meta["image_file"]).name
                    img_path = self.images_dir / img_name
                    
                    if img_path.exists():
                        self.samples.append({
                            "img_path": img_path,
                            "keypoints_2d": np.array(meta["keypoints_2d"], dtype=np.float32), # (21, 2)
                            "keypoints_3d": np.array(meta["keypoints_3d"], dtype=np.float32), # (21, 3)
                        })
                except Exception as e:
                    print(f"Error loading custom label {json_path}: {e}")
        
        if len(self.samples) > 0:
            print(f"📦 [CUSTOM DATASET] Loaded {len(self.samples)} custom labeled samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample["img_path"]).convert("RGB")
        img_np = np.array(img)
        
        kpts_2d = sample["keypoints_2d"]
        kpts_3d = sample["keypoints_3d"]
        
        if self.transform:
            # transform expects keypoints in pixels [0, 256]
            kpts_2d_px = kpts_2d * 256.0
            augmented = self.transform(image=img_np, keypoints=kpts_2d_px)
            img_tensor = augmented['image']
            kpts_2d = torch.tensor(augmented['keypoints']) / 256.0
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            kpts_2d = torch.from_numpy(kpts_2d)
            
        return {
            'image': img_tensor,
            'joints_3d': torch.from_numpy(kpts_3d),
            'coords_2d': kpts_2d
        }
