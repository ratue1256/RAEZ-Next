# hand_bone_tracker/datasets/freihand_dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class FreiHANDDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        print(f"📦 [DATASET] Chargement de {h5_path} en mémoire vive (RAM) pour accélérer l'entraînement...", flush=True)
        import time
        t0 = time.time()
        with h5py.File(self.h5_path, 'r') as f:
            self.images = np.array(f['images'])        # uint8 in memory (~4.8GB RAM)
            self.keypoints_3d = np.array(f['keypoints_3d'])
            self.keypoints_2d = np.array(f['keypoints_2d'])
        self.length = len(self.images)
        print(f"✅ [DATASET] Chargement terminé en {time.time() - t0:.1f}s. Taille: {self.length} échantillons.", flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        kpts_3d = self.keypoints_3d[idx]
        kpts_2d = self.keypoints_2d[idx]
        
        if self.transform:
            # Note: transform expects keypoints in pixels [0, 256]
            kpts_2d_px = kpts_2d * 256.0
            augmented = self.transform(image=image, keypoints=kpts_2d_px)
            image = augmented['image']
            kpts_2d = torch.tensor(augmented['keypoints'], dtype=torch.float32) / 256.0
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            kpts_2d = torch.from_numpy(kpts_2d).float()

        return {
            'image': image,
            'joints_3d': torch.from_numpy(kpts_3d).float(),
            'coords_2d': kpts_2d
        }
