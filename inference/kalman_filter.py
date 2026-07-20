# hand_bone_tracker/inference/kalman_filter.py
"""
Lissage temporel du tracking main — deux stratégies :

  * OneEuroFilter    : filtre "1€" adaptatif (SOTA temps réel). Faible latence quand
                       la main bouge vite, fort lissage quand elle est immobile.
                       Adapte aussi le lissage à la confiance par articulation.
                       -> recommandé pour la fluidité (défaut de la webcam).

  * HandKalmanFilter : filtre de Kalman à vitesse constante (21 filtres 3D).
                       Conservé pour compatibilité / comparaison.
"""

import numpy as np
from typing import Optional

try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except Exception:  # filterpy optionnel : One-Euro n'en a pas besoin
    _HAS_FILTERPY = False


# ============================================================================
# Filtre 1€ (One-Euro) — Casiez et al. 2012
# ============================================================================
class OneEuroFilter:
    """
    Lissage adaptatif vitesse/latence pour un tableau de forme quelconque
    (typiquement (21, 3)). Un seul filtre gère toute la main d'un coup.

    freq       : fréquence nominale (FPS).
    min_cutoff : coupure minimale (Hz). Plus bas = plus lisse mais plus de latence.
    beta       : réactivité à la vitesse. Plus haut = moins de latence en mouvement.
    d_cutoff   : coupure du filtrage de la dérivée.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(t_e: float, cutoff):
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def __call__(
        self,
        x: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        t_e = float(dt) if dt else (1.0 / self.freq)

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # Dérivée lissée
        dx = (x - self.x_prev) / t_e
        a_d = self._alpha(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Coupure adaptative : + de vitesse -> + de coupure (moins de latence).
        min_cutoff = self.min_cutoff
        if confidence is not None:
            # Faible confiance -> coupure réduite -> plus de lissage (masque le bruit).
            conf = np.asarray(confidence, dtype=np.float32).reshape(-1, 1)
            min_cutoff = self.min_cutoff * np.clip(conf, 0.1, 1.0)

        cutoff = min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.astype(np.float32)


# ============================================================================
# Filtre de Kalman (vitesse constante) — 21 filtres 3D indépendants
# ============================================================================
class HandKalmanFilter:

    NUM_KEYPOINTS = 21
    STATE_DIM = 6
    OBS_DIM = 3

    def __init__(
        self,
        fps: float = 30.0,
        process_noise_std: float = 0.005,
        measurement_noise_std: float = 0.015,
    ):
        if not _HAS_FILTERPY:
            raise ImportError(
                "filterpy est requis pour HandKalmanFilter. "
                "Installez-le (pip install filterpy) ou utilisez OneEuroFilter."
            )
        self.fps = fps
        self.dt = 1.0 / fps
        self.measurement_noise_std = measurement_noise_std
        self.initialized = False
        self.filters = [
            self._create_filter(process_noise_std, measurement_noise_std)
            for _ in range(self.NUM_KEYPOINTS)
        ]

    def _create_filter(self, q_std: float, r_std: float) -> "KalmanFilter":
        kf = KalmanFilter(dim_x=self.STATE_DIM, dim_z=self.OBS_DIM)
        dt = self.dt

        kf.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float32)

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        q = q_std ** 2
        kf.Q = np.diag([q, q, q, q * 10, q * 10, q * 10]).astype(np.float32)

        r = r_std ** 2  # (v1 : référençait une variable inexistante -> NameError)
        kf.R = np.diag([r, r, r]).astype(np.float32)

        kf.P = np.eye(self.STATE_DIM, dtype=np.float32) * 1.0
        return kf

    def initialize(self, joints_3d: np.ndarray):
        for i, kf in enumerate(self.filters):
            kf.x = np.zeros(self.STATE_DIM, dtype=np.float32)
            kf.x[:3] = joints_3d[i]
        self.initialized = True

    def update(
        self,
        joints_3d: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.initialized:
            self.initialize(joints_3d)
            return joints_3d

        smoothed = np.zeros((self.NUM_KEYPOINTS, 3), dtype=np.float32)
        base_r = self.measurement_noise_std

        for i, kf in enumerate(self.filters):
            kf.predict()

            if confidence is not None:
                conf = max(float(confidence[i]), 0.1)
                r = (base_r / conf) ** 2
                kf.R = np.diag([r, r, r]).astype(np.float32)

            kf.update(joints_3d[i].reshape(3, 1))
            smoothed[i] = kf.x[:3]

        return smoothed

    def reset(self):
        self.initialized = False
        for kf in self.filters:
            kf.P = np.eye(self.STATE_DIM, dtype=np.float32) * 1.0
