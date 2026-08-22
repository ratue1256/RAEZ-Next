# hand_bone_tracker/scripts/check_env.py
"""Environment sanity check: verifies every heavy dependency imports."""
import torch
import pytorch_lightning  # noqa: F401
import albumentations  # noqa: F401
import h5py  # noqa: F401
import timm  # noqa: F401

print('All imports successful!')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
