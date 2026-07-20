import torch
import pytorch_lightning
import albumentations
import h5py
import timm

print('All imports successful!')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
