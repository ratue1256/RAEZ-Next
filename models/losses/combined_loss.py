# hand_bone_tracker/models/losses/combined_loss.py
"""
CombinedHandLoss v2.0 — "ultra poussé"
======================================

Composantes :
  1. heatmap   : MSE entre heatmaps prédites et gaussiennes GT (localisation)
  2. coord2d   : L1 direct sur les coords soft-argmax vs GT 2D  (NOUVEAU — supervise
                 la localisation sous-pixel, absente de la v1)
  3. 3d        : L1 sur les positions 3D relatives au poignet
  4. 3d_norm   : L1 3D invariant à l'échelle (NOUVEAU — robuste aux différences
                 de taille de main / distance caméra)
  5. bone      : cohérence des longueurs d'os (vectorisée)
  6. temporal  : lissage inter-frames (optionnel)

Rétro-compatible : les poids inconnus des anciens configs prennent un défaut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Arbre osseux (20 os) : (parent, enfant)
BONE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
_BONE_START = [a for a, _ in BONE_PAIRS]
_BONE_END = [b for _, b in BONE_PAIRS]


class CombinedHandLoss(nn.Module):
    def __init__(
        self,
        lambda_heatmap: float = 1.0,
        lambda_coord2d: float = 1.0,
        lambda_3d: float = 2.0,
        lambda_3d_norm: float = 1.0,
        lambda_bone: float = 0.5,
        lambda_temporal: float = 0.3,
        heatmap_size: int = 64,
        heatmap_sigma: float = 2.0,
    ):
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_coord2d = lambda_coord2d
        self.lambda_3d = lambda_3d
        self.lambda_3d_norm = lambda_3d_norm
        self.lambda_bone = lambda_bone
        self.lambda_temporal = lambda_temporal
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma

    # ------------------------------------------------------------------ utils
    def generate_gt_heatmaps(self, coords_2d: torch.Tensor) -> torch.Tensor:
        """coords_2d (B, 21, 2) in [0,1] -> gaussiennes (B, 21, S, S)."""
        size, sigma = self.heatmap_size, self.heatmap_sigma
        B, K, _ = coords_2d.shape
        rng = torch.arange(size, device=coords_2d.device, dtype=coords_2d.dtype)
        grid_y, grid_x = torch.meshgrid(rng, rng, indexing="ij")

        kpts_x = (coords_2d[:, :, 0] * (size - 1)).view(B, K, 1, 1)
        kpts_y = (coords_2d[:, :, 1] * (size - 1)).view(B, K, 1, 1)

        diff_x = grid_x.view(1, 1, size, size) - kpts_x
        diff_y = grid_y.view(1, 1, size, size) - kpts_y
        return torch.exp(-(diff_x ** 2 + diff_y ** 2) / (2 * sigma ** 2))

    @staticmethod
    def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_len = torch.linalg.vector_norm(
            pred[:, _BONE_END] - pred[:, _BONE_START], dim=-1
        )
        gt_len = torch.linalg.vector_norm(
            gt[:, _BONE_END] - gt[:, _BONE_START], dim=-1
        )
        return F.l1_loss(pred_len, gt_len)

    @staticmethod
    def scale_invariant_3d_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Aligne l'échelle globale (facteur scalaire par échantillon) avant L1."""
        num = (pred * gt).sum(dim=(1, 2))
        den = (pred * pred).sum(dim=(1, 2)).clamp(min=1e-8)
        scale = (num / den).detach().view(-1, 1, 1)
        return F.l1_loss(scale * pred, gt)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        predictions: dict,
        gt_joints_3d: torch.Tensor,
        gt_coords_2d: torch.Tensor,
        prev_predictions: dict = None,
    ) -> dict:
        device = gt_joints_3d.device

        gt_heatmaps = self.generate_gt_heatmaps(gt_coords_2d)
        l_heatmap = F.mse_loss(predictions["heatmaps"], gt_heatmaps)

        l_coord2d = F.l1_loss(predictions["coords_2d"], gt_coords_2d)

        l_3d = F.l1_loss(predictions["joints_3d"], gt_joints_3d)
        l_3d_norm = self.scale_invariant_3d_loss(predictions["joints_3d"], gt_joints_3d)
        l_bone = self.bone_length_loss(predictions["joints_3d"], gt_joints_3d)

        l_temporal = torch.zeros((), device=device)
        if prev_predictions is not None:
            velocity = predictions["joints_3d"] - prev_predictions["joints_3d"].detach()
            l_temporal = velocity.pow(2).mean()

        total = (
            self.lambda_heatmap * l_heatmap
            + self.lambda_coord2d * l_coord2d
            + self.lambda_3d * l_3d
            + self.lambda_3d_norm * l_3d_norm
            + self.lambda_bone * l_bone
            + self.lambda_temporal * l_temporal
        )

        return {
            "total": total,
            "heatmap": l_heatmap.item(),
            "coord2d": l_coord2d.item(),
            "3d": l_3d.item(),
            "3d_norm": l_3d_norm.item(),
            "bone": l_bone.item(),
            "temporal": float(l_temporal.item()),
        }
