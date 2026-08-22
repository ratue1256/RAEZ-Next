# hand_bone_tracker/scripts/export_onnx.py
"""
Export ONNX du HandBoneTracker.

Un ExportWrapper renvoie un tuple ordonné et STABLE de tenseurs
(heatmaps, coords_2d, joints_3d, confidence), ce qui donne un graphe ONNX
propre (pas de sortie intermédiaire parasite) consommable tel quel par la webcam
et la quantization.
"""

import os
import sys

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.hand_tracker import HandBoneTracker


OUTPUT_NAMES = ["heatmaps", "coords_2d", "joints_3d", "confidence"]


class ExportWrapper(nn.Module):
    """Aplati le dict de sortie en tuple ordonné pour un export ONNX propre."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        o = self.model(x)
        return o["heatmaps"], o["coords_2d"], o["joints_3d"], o["confidence"]


def _load_weights(model: HandBoneTracker, checkpoint: dict):
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean = {}
    for k, v in state_dict.items():
        clean[k[6:] if k.startswith("model.") else k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[WARN] Poids manquants ({len(missing)}), ex: {list(missing)[:3]}")
    if unexpected:
        print(f"[WARN] Poids inattendus ({len(unexpected)}), ex: {list(unexpected)[:3]}")
    print("[OK] Poids charges.")


def export_to_onnx(checkpoint_path: str, output_path: str):
    backbone_name = "mobilenetv3_large_100"
    checkpoint = None
    if os.path.exists(checkpoint_path):
        print(f"Chargement du checkpoint : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("hyper_parameters", {}).get("config", {})
        backbone_name = config.get("backbone", backbone_name)
        print(f"Backbone détecté : {backbone_name}")
    else:
        print(f"Checkpoint non trouvé : {checkpoint_path}. Export d'un modèle non entraîné (démo).")

    model = HandBoneTracker(backbone_name=backbone_name, pretrained=False)
    if checkpoint is not None:
        _load_weights(model, checkpoint)
    model.eval()

    wrapper = ExportWrapper(model).eval()
    dummy_input = torch.randn(1, 3, 256, 256)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Création du graphe ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "image": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
            "coords_2d": {0: "batch_size"},
            "joints_3d": {0: "batch_size"},
            "confidence": {0: "batch_size"},
        },
    )

    print("Simplification du graphe ONNX...")
    model_onnx = onnx.load(output_path)
    model_simplified, check = simplify(model_onnx)
    assert check, "Simplification ONNX échouée"
    simplified_path = output_path.replace(".onnx", "_simplified.onnx")
    onnx.save(model_simplified, simplified_path)

    print(f"[SUCCESS] Export ONNX -> {simplified_path}")


def _auto_best_checkpoint() -> str:
    from pathlib import Path
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return "checkpoints/last.ckpt"

    ckpts = list(ckpt_dir.glob("**/*.ckpt"))
    best_ckpt, best_mpjpe = None, float("inf")
    for c in ckpts:
        if "val_mpjpe_3d" in c.name:
            try:
                score = float(c.name.split("val_mpjpe_3d=")[-1].replace(".ckpt", ""))
                if score < best_mpjpe:
                    best_mpjpe, best_ckpt = score, c
            except Exception:
                pass
    if best_ckpt is None and ckpts:
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        best_ckpt = ckpts[0]
    return str(best_ckpt) if best_ckpt else "checkpoints/last.ckpt"


if __name__ == "__main__":
    checkpoint_path = _auto_best_checkpoint()
    print(f"Meilleur checkpoint détecté : {checkpoint_path}")
    export_to_onnx(checkpoint_path, "exports/hand_tracker.onnx")
