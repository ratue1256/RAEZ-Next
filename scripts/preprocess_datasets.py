# hand_bone_tracker/scripts/preprocess_datasets.py
"""
Préprocessing unifié de tous les datasets.
Output : fichiers HDF5 dans data/processed/
Supports splitting Train/Val/Test pour FreiHAND.
"""

import h5py
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import torch
import random
import pickle

# Résolution cible UNIQUE pour tous les datasets
TARGET_SIZE = (256, 256)

# Nombre de keypoints (standard)
NUM_KEYPOINTS = 21

# Ordre des keypoints (MediaPipe convention)
KEYPOINT_NAMES = [
    'WRIST',                                          # 0
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',  # 1-4
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',  # 5-8
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',  # 9-12
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',  # 13-16
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',  # 17-20
]

def save_to_h5(indices, raw_path, keypoints_3d, camera_matrices, output_file, desc):
    """Sauvegarde un sous-ensemble d'indices dans un fichier HDF5."""
    n_samples = len(indices)
    
    with h5py.File(output_file, 'w') as f:
        images_ds = f.create_dataset('images',
                                     shape=(n_samples, 256, 256, 3),
                                     dtype=np.uint8,
                                     chunks=(32, 256, 256, 3), # Better for writing
                                     compression='lzf')
        
        kpts_ds = f.create_dataset('keypoints_3d',
                                   shape=(n_samples, 21, 3),
                                   dtype=np.float32)
        
        kpts_2d_ds = f.create_dataset('keypoints_2d',
                                      shape=(n_samples, 21, 2),
                                      dtype=np.float32)
        
        for i, idx in enumerate(tqdm(indices, desc=desc)):
            img_path = raw_path / 'training' / 'rgb' / f'{idx:08d}.jpg'
            img = cv2.imread(str(img_path))
            if img is None:
                # Fill with zeros or skip? Let's skip and adjust n_samples later or just put zeros.
                # Actually FreiHAND should have all images.
                print(f"Warning: Missing image {img_path}")
                continue
            
            img = cv2.resize(img, TARGET_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            K = camera_matrices[idx]
            xyz = keypoints_3d[idx]
            xyz_proj = (K @ xyz.T).T
            uv = xyz_proj[:, :2] / xyz_proj[:, 2:3]
            
            # Normaliser coordonnées 2D en [0, 1] (basé sur 224x224 original)
            uv[:, 0] /= 224.0
            uv[:, 1] /= 224.0
            
            # Normaliser profondeur (relatif au poignet)
            xyz_normalized = xyz - xyz[0:1, :]
            
            images_ds[i] = img
            kpts_ds[i] = xyz_normalized.astype(np.float32)
            kpts_2d_ds[i] = uv.astype(np.float32)

            if i % 1000 == 0:
                f.flush()

def preprocess_freihand(raw_path: Path, output_dir: Path):
    """Convertit FreiHAND en HDF5 avec split Train/Val/Test."""
    
    print("Chargement des annotations FreiHAND...")
    with open(raw_path / 'training_xyz.json') as f:
        keypoints_3d = np.array(json.load(f))
    
    with open(raw_path / 'training_K.json') as f:
        camera_matrices = np.array(json.load(f))
    
    n_total = len(keypoints_3d)
    n_unique = 32560 # Nombre de poses uniques avant augmentation de background
    
    if n_total != n_unique * 4:
        print(f"Warning: Inattendu n_total={n_total}. Devrait être {n_unique*4}")
        n_unique = n_total // 4
    
    # Split basé sur les poses uniques pour éviter le leakage
    indices = list(range(n_unique))
    random.seed(42)
    random.shuffle(indices)
    
    train_end = int(n_unique * 0.8)
    val_end = train_end + int(n_unique * 0.1)
    
    train_unique = indices[:train_end]
    val_unique = indices[train_end:val_end]
    test_unique = indices[val_end:]
    
    # Étendre aux 4 versions (background original + 3 augmentés)
    def expand_indices(unique_idx):
        expanded = []
        for idx in unique_idx:
            for j in range(4):
                expanded.append(idx + j * n_unique)
        return expanded

    train_idx = expand_indices(train_unique)
    val_idx = expand_indices(val_unique)
    test_idx = expand_indices(test_unique)
    
    print(f"Splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    save_to_h5(train_idx, raw_path, keypoints_3d, camera_matrices, output_dir / 'freihand_train.h5', 'FreiHAND Train')
    save_to_h5(val_idx, raw_path, keypoints_3d, camera_matrices, output_dir / 'freihand_val.h5', 'FreiHAND Val')
    save_to_h5(test_idx, raw_path, keypoints_3d, camera_matrices, output_dir / 'freihand_test.h5', 'FreiHAND Test')

def preprocess_rhd(raw_path: Path, output_path: Path):
    """Convertit RHD en HDF5 normalisé."""
    # Chemins RHD
    anno_path = raw_path / 'training' / 'anno_training.pickle'
    if not anno_path.exists():
        anno_path = raw_path / 'anno_training.pickle'
        
    if not anno_path.exists():
        print(f"Dataset RHD non trouvé dans {raw_path}")
        return

    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)
    
    n_samples = len(anno)
    
    with h5py.File(output_path / 'rhd_train.h5', 'w') as f:
        images_ds = f.create_dataset('images',
                                     shape=(n_samples, 256, 256, 3),
                                     dtype=np.uint8,
                                     chunks=(32, 256, 256, 3),
                                     compression='lzf')
        
        kpts_ds = f.create_dataset('keypoints_3d',
                                   shape=(n_samples, 21, 3),
                                   dtype=np.float32)
        
        kpts_2d_ds = f.create_dataset('keypoints_2d',
                                      shape=(n_samples, 21, 2),
                                      dtype=np.float32)
        
        for i, (sample_id, data) in enumerate(tqdm(anno.items(), desc='RHD Train')):
            img_path = raw_path / 'training' / 'rgb' / f'{sample_id:05d}.png'
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            img = cv2.resize(img, TARGET_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Keypoints 3D et 2D (RHD has 42 keypoints for both hands, we take the first hand or both?)
            # Usually we filter for visible hand. RHD keypoints: 0-20 left, 21-41 right.
            kp_coords_2d = data['uv_vis'][:, :2]
            kp_coords_3d = data['xyz']
            
            # Choose hand (simplified: take first 21)
            kp_2d = kp_coords_2d[:21]
            kp_3d = kp_coords_3d[:21]
            
            # Normaliser 2D en [0, 1] (RHD is 320x320)
            kp_2d[:, 0] /= 320.0
            kp_2d[:, 1] /= 320.0
            
            # Normaliser 3D (relatif au poignet)
            kp_3d_norm = kp_3d - kp_3d[0:1, :]
            
            images_ds[i] = img
            kpts_ds[i] = kp_3d_norm.astype(np.float32)
            kpts_2d_ds[i] = kp_2d.astype(np.float32)

            if i % 1000 == 0:
                f.flush()

    print(f"RHD préprocessé → {output_path}/rhd_train.h5")

if __name__ == '__main__':
    print("Starting preprocessing script...")
    base_raw = Path('data/raw')
    base_processed = Path('data/processed')
    base_processed.mkdir(parents=True, exist_ok=True)
    
    print(f"Base raw path: {base_raw.absolute()}")
    
    # 1. FreiHAND
    xyz_path = base_raw / 'training_xyz.json'
    print(f"Checking for FreiHAND at {xyz_path.absolute()}")
    if xyz_path.exists():
        preprocess_freihand(base_raw, base_processed)
    else:
        # Check subfolder
        freihand_raw = base_raw / 'freihand'
        print(f"Checking for FreiHAND at {freihand_raw.absolute() / 'training_xyz.json'}")
        if (freihand_raw / 'training_xyz.json').exists():
            preprocess_freihand(freihand_raw, base_processed)
        else:
            print("FreiHAND raw files not found.")

    # 2. RHD
    rhd_raw = base_raw / 'rhd'
    if rhd_raw.exists():
        preprocess_rhd(rhd_raw, base_processed)
    else:
        # Search for folder with RHD in name
        rhd_candidates = list(base_raw.glob('*RHD*'))
        if rhd_candidates:
            preprocess_rhd(rhd_candidates[0], base_processed)
        else:
            print("RHD raw files not found.")
