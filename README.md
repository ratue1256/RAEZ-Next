# hand-bone-tracker

> Tracking 3D main & os temps réel, optimisé CPU (v2.0)

## Objectifs de performance (cibles, à atteindre après entraînement)
> ⚠️ Ce sont des **objectifs de conception**, pas des mesures : aucun poids
> entraîné n'est fourni. Ils se valident par l'entraînement sur FreiHAND/HO3D.

| Métrique | Cible | MediaPipe Hands v0.10 |
|---|---|---|
| MPJPE 3D FreiHAND | ~12 mm | ~15 mm |
| FPS CPU (INT8) | 60+ | 35 |
| Robustesse occlusion | élevée | faible |

## Nouveautés v2.0
- **Backbone mobile** (`mobilenetv3_large_100`) — rapide sur CPU.
- **Couche biomécanique réelle** : cinématique directe différentiable (chaque os
  garde sa direction, longueur bornée anatomiquement) au lieu d'un simple clamp.
- **Supervision 2D directe** + **loss 3D invariante à l'échelle**.
- **Confiance par articulation** (entropie des heatmaps) exploitée à l'inférence.
- **Filtre One-Euro** adaptatif à la confiance (fluidité SOTA temps réel).
- **EMA** des poids, **suivi ROI** adaptatif, rendu coloré par doigt.
- Bugs bloquants corrigés (Kalman `NameError`, `Resize` d'augmentation manquant).

## Installation

### Local (Windows)
```powershell
./setup.ps1
```

### Google Colab
```python
!pip install -r requirements_colab.txt
```

## Usage
```python
from inference.webcam_demo import HandTrackerRT
tracker = HandTrackerRT(model_path='exports/hand_tracker_int8.onnx')
tracker.run_webcam()
```

## Structure du projet
- `configs/`: Fichiers de configuration Hydra (YAML)
- `models/`: Architecture PyTorch et fonctions de loss
- `datasets/`: Scripts de chargement et d'augmentation
- `training/`: Module Lightning, callbacks (EMA, checkpoints)
- `inference/`: Inférence ONNX et filtrage temporel (One-Euro / Kalman)
- `scripts/`: Utilitaires de préprocessing, export et quantization
- `notebooks/`: Tutoriels et expérimentations
