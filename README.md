# Hand Bone Tracker (v2.0)

Real-time 3D hand and skeleton tracking with differentiable biomechanical constraints. Runs on CPU (60+ FPS in INT8) and CUDA.

Includes a complete PyTorch Lightning training pipeline, ONNX export/quantization tools, and a fullstack web dashboard (FastAPI + React).

## Quick Start

### 1. Setup Environment

**Windows:**
```powershell
./setup.ps1
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
Start both the FastAPI backend and the Vite frontend:

```powershell
# Windows
./run_gui.ps1
# or run run_gui.bat
```
Open `http://localhost:5173` to access the live training monitor, model testing, and data collector.

---

## Training

The model trains in 3 progressive stages:

```bash
# Stage 1: Pre-training (Backbone frozen for 5 epochs, then full training)
python scripts/train.py --config configs/training/stage1_pretrain.yaml

# Stage 2: Main training on full dataset (OneCycleLR)
python scripts/train.py --config configs/training/stage2_main.yaml --resume

# Stage 3: Fine-tuning with low learning rate
python scripts/train.py --config configs/training/stage3_finetune.yaml --resume
```

To train on your custom poses collected via the dashboard:
```bash
python scripts/train.py --config configs/training/stage1_pretrain.yaml
# Custom samples in data/raw/custom are automatically detected and mixed in.
```

---

## Export & INT8 Quantization

Convert your best PyTorch checkpoint to ONNX and quantize to INT8:

```bash
# 1. Export PyTorch checkpoint -> ONNX FP32
python scripts/export_onnx.py

# 2. Quantize ONNX -> INT8 (uses calibration data)
python scripts/quantize.py
```

Outputs are saved in `exports/`:
- `exports/hand_tracker.onnx`
- `exports/hand_tracker_simplified.onnx`
- `exports/hand_tracker_int8.onnx`

---

## Real-Time Webcam Demo

Run standalone inference on your webcam with One-Euro temporal filtering:

```python
from inference.webcam_demo import HandTrackerRT

tracker = HandTrackerRT(
    model_path="exports/hand_tracker_int8.onnx",
    use_gpu=True,
    target_fps=30.0
)
tracker.run_webcam(camera_id=0)
```

Press `q` to exit, `r` to reset the tracking ROI.

---

## Architecture

- **Backbone:** `mobilenetv3_large_100` (fast CPU inference).
- **2D Head:** 21 heatmaps (64x64) with soft-argmax integral regression for sub-pixel accuracy.
- **3D Head:** Metric regression with differentiable Forward Kinematics (`BiomechanicalFKLayer`) enforcing realistic bone lengths.
- **Confidence:** Per-joint reliability derived from heatmap entropy ($1 - H / \ln N$).
- **Filtering:** Adaptive One-Euro filter dynamically modulated by joint confidence.

---

## Project Structure

```
hand_bone_tracker/
├── configs/          # Training configs (YAML)
├── datasets/         # FreiHAND, Custom loaders, and augmentations
├── gui/              # FastAPI backend & React/Vite dashboard
├── inference/        # ONNX runtime & One-Euro filter
├── models/           # HandBoneTracker architecture & CombinedLoss
├── scripts/          # train, export_onnx, quantize, preprocess
└── setup.ps1         # Windows automated install script
```
