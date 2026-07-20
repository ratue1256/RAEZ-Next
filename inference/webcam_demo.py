# hand_bone_tracker/inference/webcam_demo.py
"""
Démo webcam temps réel — Hand & Bone Tracker v2.0 "ultra poussé"
================================================================

Pipeline :
  frame -> (ROI adaptatif) -> preprocess -> ONNX -> coords_2d / joints_3d / confidence
        -> lissage One-Euro (adaptatif à la confiance) -> rendu coloré par doigt

Points forts :
  * Suivi ROI : recadrage autour de la main détectée (robustesse + précision).
  * Gating par confiance : réinitialise le suivi quand la main est perdue.
  * Filtre One-Euro : fluide au repos, réactif en mouvement.
  * Sorties ONNX résolues dynamiquement (compatible modèles avec/sans confidence).
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from kalman_filter import OneEuroFilter

# --- topologie & couleurs (BGR) par doigt -----------------------------------
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # majeur
    (0, 13), (13, 14), (14, 15), (15, 16),  # annulaire
    (0, 17), (17, 18), (18, 19), (19, 20),  # auriculaire
    (5, 9), (9, 13), (13, 17),              # paume
]
_PALM_CONNECTIONS = {(5, 9), (9, 13), (13, 17)}
_FINGER_OF_JOINT = {0: "palm"}
for _f, _base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
    for _j in range(_base, _base + 4):
        _FINGER_OF_JOINT[_j] = _f
FINGER_COLORS = {
    "thumb": (0, 0, 255), "index": (0, 165, 255), "middle": (0, 255, 255),
    "ring": (0, 255, 0), "pinky": (255, 0, 0), "palm": (200, 200, 200),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HandTrackerRT:
    """Hand Tracker temps réel — ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        target_fps: float = 30.0,
        input_size: int = 256,
        conf_threshold: float = 0.25,
        use_roi: bool = True,
    ):
        providers = []
        if use_gpu:
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }))
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 4)

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.use_roi = use_roi

        # Deux filtres One-Euro : un pour la 3D (métrique), un pour la 2D (affichage).
        self.filter_3d = OneEuroFilter(freq=target_fps, min_cutoff=1.0, beta=0.4)
        self.filter_2d = OneEuroFilter(freq=target_fps, min_cutoff=1.5, beta=0.5)

        self.roi = None  # (x, y, w, h) en pixels, ou None = plein cadre
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.perf_counter()

    # ---------------------------------------------------------------- preproc
    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(img)

    # ------------------------------------------------------------------- ROI
    def _square_roi(self, pts_px: np.ndarray, W: int, H: int, margin: float = 0.4):
        x0, y0 = pts_px.min(axis=0)
        x1, y1 = pts_px.max(axis=0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
        side = max(side, 64.0)
        x = int(np.clip(cx - side / 2.0, 0, max(0, W - 1)))
        y = int(np.clip(cy - side / 2.0, 0, max(0, H - 1)))
        w = int(min(side, W - x))
        h = int(min(side, H - y))
        return (x, y, w, h)

    def _outputs_to_dict(self, outputs):
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

    # --------------------------------------------------------------- predict
    def predict(self, frame: np.ndarray) -> dict:
        H, W = frame.shape[:2]

        if self.use_roi and self.roi is not None:
            rx, ry, rw, rh = self.roi
            crop = frame[ry:ry + rh, rx:rx + rw]
            if crop.size == 0:
                crop, (rx, ry, rw, rh) = frame, (0, 0, W, H)
        else:
            crop, (rx, ry, rw, rh) = frame, (0, 0, W, H)

        outputs = self.session.run(self.output_names, {self.input_name: self.preprocess(crop)})
        out = self._outputs_to_dict(outputs)

        coords_2d = out["coords_2d"][0]                       # (21, 2) dans le crop
        joints_3d = out["joints_3d"][0]                       # (21, 3)
        confidence = out["confidence"][0] if "confidence" in out else np.ones(21, np.float32)

        mean_conf = float(np.mean(confidence))

        # Coords normalisées du crop -> pixels plein cadre
        pts_px = coords_2d.copy()
        pts_px[:, 0] = rx + coords_2d[:, 0] * rw
        pts_px[:, 1] = ry + coords_2d[:, 1] * rh

        # Main perdue : on lâche le ROI et on réinitialise le lissage
        if mean_conf < self.conf_threshold:
            self.roi = None
            self.filter_3d.reset()
            self.filter_2d.reset()
        else:
            pts_px = self.filter_2d(pts_px, confidence=confidence)
            joints_3d = self.filter_3d(joints_3d, confidence=confidence)
            if self.use_roi:
                self.roi = self._square_roi(pts_px, W, H)

        return {
            "joints_3d": joints_3d,
            "pts_px": pts_px,
            "confidence": confidence,
            "mean_conf": mean_conf,
        }

    # ---------------------------------------------------------------- render
    def draw_skeleton(self, frame, pts_px, confidence) -> np.ndarray:
        pts = pts_px.astype(int)
        for start, end in BONE_CONNECTIONS:
            # Connexions de paume en gris, os de doigt à la couleur du doigt cible.
            finger = "palm" if (start, end) in _PALM_CONNECTIONS else _FINGER_OF_JOINT.get(end, "palm")
            cv2.line(frame, tuple(pts[start]), tuple(pts[end]),
                     FINGER_COLORS[finger], 2, cv2.LINE_AA)

        for j, pt in enumerate(pts):
            c = float(confidence[j])
            color = (0, 255, 0) if c >= self.conf_threshold else (0, 0, 180)
            radius = 5 if c >= 0.5 else 3
            cv2.circle(frame, tuple(pt), radius, color, -1, cv2.LINE_AA)
        return frame

    def update_fps(self) -> float:
        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.fps_buffer.append(fps)
        return float(np.mean(self.fps_buffer))

    def run_webcam(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Hand Tracker démarré — 'q' pour quitter, 'r' pour réinitialiser le ROI")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict(frame)
            frame = self.draw_skeleton(frame, result["pts_px"], result["confidence"])

            fps = self.update_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}  conf: {result['mean_conf']:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Hand Bone Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                self.roi = None
                self.filter_3d.reset()
                self.filter_2d.reset()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for candidate in ("exports/hand_tracker_int8.onnx",
                      "exports/hand_tracker_simplified.onnx",
                      "exports/hand_tracker.onnx"):
        if os.path.exists(candidate):
            HandTrackerRT(model_path=candidate, use_gpu=True).run_webcam(camera_id=0)
            break
    else:
        print("Modèle non trouvé dans exports/. Exportez et quantifiez d'abord le modèle.")
