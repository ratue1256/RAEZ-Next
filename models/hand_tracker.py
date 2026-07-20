# hand_bone_tracker/models/hand_tracker.py
"""
HandBoneTracker v2.0  —  "ultra poussé", optimisé CPU temps réel
================================================================

Architecture :
    Backbone mobile timm (features_only, sélection dynamique du dernier stage)
      -> HeatmapHead  : 21 heatmaps 64x64  -> coords 2D par soft-argmax (integral)
      -> RegressionHead3D : 21 x 3 (xyz relatif au poignet, en mètres)
      -> BiomechanicalFKLayer : reconstruction cinématique différentiable
                                (chaque os garde sa direction, longueur bornée)
      -> confidence : score de fiabilité par articulation (0..1), sans paramètre

Objectifs : rapide (backbone mobile), fiable (confiance + contraintes
anatomiques), fluide (soft-argmax sous-pixel + sortie exploitable par One-Euro).

Contrat de sortie (stable, consommé par la loss / l'export ONNX / la webcam) :
    {
      'heatmaps'      : (B, 21, 64, 64),
      'coords_2d'     : (B, 21, 2)  in [0, 1],
      'joints_3d'     : (B, 21, 3)  metres, relatif au poignet, contraint,
      'joints_3d_raw' : (B, 21, 3)  avant contrainte (pour la loss),
      'confidence'    : (B, 21)     in [0, 1],
    }
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ----------------------------------------------------------------------------
# Topologie de la main (convention 21 keypoints, style MediaPipe / FreiHAND)
#   0 = poignet
#   pouce      : 1,2,3,4     index : 5,6,7,8       majeur : 9,10,11,12
#   annulaire  : 13,14,15,16 auriculaire : 17,18,19,20
# ----------------------------------------------------------------------------
# Parent de chaque articulation dans l'arbre cinématique (-1 = racine).
# Propriété clef : parent[i] < i  =>  reconstruction en un seul passage avant.
JOINT_PARENTS = (
    -1,            # 0  poignet
    0, 1, 2, 3,    # pouce
    0, 5, 6, 7,    # index
    0, 9, 10, 11,  # majeur
    0, 13, 14, 15,   # annulaire
    0, 17, 18, 19,   # auriculaire
)

# Longueur maximale plausible de chaque os (mètres). Filet de sécurité
# anatomique généreux : empeche les os aberrants sans brider l'apprentissage.
_META = 0.11   # métacarpien (poignet -> base du doigt)
_PROX = 0.055  # phalange proximale
_MID = 0.040  # phalange intermédiaire
_DIST = 0.035  # phalange distale
MAX_BONE_LENGTH = (
    0.0,                        # 0 (racine)
    _META, _PROX, _MID, _DIST,   # pouce
    _META, _PROX, _MID, _DIST,   # index
    _META, _PROX, _MID, _DIST,   # majeur
    _META, _PROX, _MID, _DIST,   # annulaire
    _META, _PROX, _MID, _DIST,   # auriculaire
)

# Backbones recommandés selon la cible (tous supportent features_only dans timm).
FAST_CPU_BACKBONES = ("mobilenetv3_large_100", "efficientnet_lite0", "efficientnet_b0")


class HeatmapHead(nn.Module):
    """21 heatmaps par upsampling déconvolutionnel : (B, C, h, w) -> (B, 21, 8h, 8w)."""

    def __init__(self, in_channels: int, num_keypoints: int = 21):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv_layers(x)
        return self.final_layer(x)  # (B, 21, 64, 64)


class RegressionHead3D(nn.Module):
    """Régression 3D relative au poignet depuis les features globales : (B, 21, 3)."""

    def __init__(self, in_channels: int, num_keypoints: int = 21):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, num_keypoints * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x).flatten(1)
        x = self.mlp(x)
        return x.view(-1, self.num_keypoints, 3)


class BiomechanicalFKLayer(nn.Module):
    """
    Contraintes anatomiques RÉELLES par cinématique directe (forward kinematics).

    On parcourt l'arbre cinématique poignet -> extrémités : chaque os conserve sa
    DIRECTION prédite mais sa LONGUEUR est bornée à [min_ratio*max, max]. Le résultat
    est anatomiquement cohérent, entièrement différentiable, et exportable en ONNX
    (boucle statique de 21 itérations, sans écriture in-place -> torch.stack).
    """

    def __init__(self, min_ratio: float = 0.2):
        super().__init__()
        self.parents = JOINT_PARENTS
        self.min_ratio = float(min_ratio)
        self.register_buffer(
            "max_len", torch.tensor(MAX_BONE_LENGTH, dtype=torch.float32)
        )

    def forward(self, joints_3d: torch.Tensor) -> torch.Tensor:
        wrist = joints_3d[:, 0:1, :]
        centered = joints_3d - wrist  # poignet à l'origine

        rebuilt = [torch.zeros_like(centered[:, 0, :])]  # articulation 0 = origine
        for i in range(1, len(self.parents)):
            parent = self.parents[i]
            bone = centered[:, i, :] - centered[:, parent, :]
            length = torch.linalg.vector_norm(bone, dim=-1, keepdim=True).clamp(min=1e-6)
            direction = bone / length

            max_l = self.max_len[i]
            clamped = length.clamp(min=float(max_l) * self.min_ratio, max=float(max_l))
            rebuilt.append(rebuilt[parent] + direction * clamped)

        out = torch.stack(rebuilt, dim=1)  # (B, 21, 3)
        return out + wrist


class HandBoneTracker(nn.Module):
    """
    Modèle principal de tracking main + os.

    backbone_name : tout modèle timm compatible features_only.
        CPU rapide (défaut) : 'mobilenetv3_large_100'
        Précision (GPU)     : 'efficientnet_b0' / 'efficientnet_b2'
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        num_keypoints: int = 21,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 5,
        softargmax_beta: float = 100.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.softargmax_beta = softargmax_beta

        # features_only sans out_indices figé -> compatible tous backbones,
        # on prend systématiquement le dernier niveau de features (plus profond).
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        backbone_out_channels = self.backbone.feature_info.channels()[-1]

        self.heatmap_head = HeatmapHead(backbone_out_channels, num_keypoints)
        self.regression_head = RegressionHead3D(backbone_out_channels, num_keypoints)
        self.constraint_layer = BiomechanicalFKLayer()

    # -- gestion du gel du backbone (utilisé par le trainer par étapes) --------
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # -- soft-argmax différentiable (régression intégrale sous-pixel) ----------
    def heatmaps_to_coords(self, heatmaps: torch.Tensor):
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, -1)
        prob = F.softmax(flat * self.softargmax_beta, dim=-1).view(B, K, H, W)

        ys = torch.linspace(0, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        xs = torch.linspace(0, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        coords_x = (prob * grid_x).sum(dim=(-2, -1))
        coords_y = (prob * grid_y).sum(dim=(-2, -1))
        coords = torch.stack([coords_x, coords_y], dim=-1)  # (B, 21, 2)
        return coords, prob

    def _confidence_from_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Fiabilité par articulation = 1 - entropie normalisée de la heatmap.

        Distribution piquée (localisation nette) -> ~1 ; diffuse (incertaine) -> ~0.
        Bornée [0, 1], différentiable, exportable ONNX.
        """
        B, K, H, W = heatmaps.shape
        n = H * W
        prob = F.softmax(heatmaps.view(B, K, n), dim=-1)          # softmax NON durci
        entropy = -(prob * (prob + 1e-12).log()).sum(dim=-1)      # (B, 21)
        return (1.0 - entropy / math.log(n)).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)[-1]
        heatmaps = self.heatmap_head(features)

        coords_2d, _ = self.heatmaps_to_coords(heatmaps)
        confidence = self._confidence_from_heatmaps(heatmaps)

        joints_3d_raw = self.regression_head(features)
        joints_3d = self.constraint_layer(joints_3d_raw)

        return {
            "heatmaps": heatmaps,
            "coords_2d": coords_2d,
            "joints_3d": joints_3d,
            "joints_3d_raw": joints_3d_raw,
            "confidence": confidence,
        }
