# hand_bone_tracker/datasets/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pipeline augmentation ENTRAÎNEMENT
train_transform = A.Compose([
    # Redimensionnement garanti (BUG v1 : absent -> crash sur images non 256px,
    # ex. webcam custom). Doit rester en tête du pipeline.
    A.Resize(height=256, width=256),

    # Translation légère et cadrage (préserve la cohérence 2D/3D sans rotation/chiralité désynchronisée)
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.95, 1.05),
        rotate=0,
        p=0.4
    ),
    
    # Couleur / luminosité
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.6
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.4
    ),
    A.CLAHE(clip_limit=4.0, p=0.3),
    
    # Robustesse occlusion
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 32),
        hole_width_range=(8, 32),
        p=0.4
    ),
    
    # Bruit / flou
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.MotionBlur(blur_limit=(3, 7), p=0.2),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    
    # Normalisation finale (ImageNet stats)
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
],
keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Pipeline VALIDATION (pas d'augmentation, juste normalisation)
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
],
keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
