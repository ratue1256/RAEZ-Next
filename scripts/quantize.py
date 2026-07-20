# hand_bone_tracker/scripts/quantize.py
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
)
import numpy as np
import onnxruntime as ort
import os

class HandCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_images: list, n_samples: int = 100):
        self.images = calibration_images[:n_samples]
        self.index = 0
    
    def get_next(self):
        if self.index >= len(self.images):
            return None
        img = self.images[self.index]
        self.index += 1
        # Model expects input named 'image'
        return {'image': img[np.newaxis, :].astype(np.float32)}

def load_calibration_data(h5_path: str, n_samples: int = 100):
    if not os.path.exists(h5_path):
        print(f"Dataset de calibration non trouvé : {h5_path}")
        return None
    import h5py
    print(f"Chargement de {n_samples} échantillons de calibration depuis {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        images = f['images'][:n_samples]
    # Convert N, H, W, C to N, C, H, W and normalize [0, 1]
    images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return list(images)

def quantize_model(input_path: str, output_path: str, calibration_data=None):
    if not os.path.exists(input_path):
        print(f"Modèle source non trouvé : {input_path}")
        return

    if calibration_data is not None:
        print("Exécution de la quantization statique (INT8) avec calibration...")
        reader = HandCalibrationReader(calibration_data)
        quantize_static(
            input_path,
            output_path,
            reader,
            quant_format=QuantFormat.QOperator,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=True,
        )
    else:
        print("Exécution de la quantization dynamique (INT8)...")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul', 'Gemm']
        )
    
    # Benchmark
    print("Démarrage du benchmark...")
    sess_fp32 = ort.InferenceSession(input_path)
    sess_int8 = ort.InferenceSession(output_path)
    
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
    import time
    N = 100
    
    start = time.perf_counter()
    for _ in range(N):
        sess_fp32.run(None, {'image': dummy})
    t_fp32 = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(N):
        sess_int8.run(None, {'image': dummy})
    t_int8 = time.perf_counter() - start
    
    fps_fp32 = N / t_fp32
    fps_int8 = N / t_int8
    print(f"FP32 : {fps_fp32:.1f} FPS")
    print(f"INT8 : {fps_int8:.1f} FPS (+{((fps_int8/fps_fp32)-1)*100:.0f}%)")

if __name__ == '__main__':
    h5_path = 'hand_bone_tracker/data/processed/freihand_val.h5'
    calib_data = load_calibration_data(h5_path, n_samples=100)
    
    quantize_model(
        'hand_bone_tracker/exports/hand_tracker_simplified.onnx',
        'hand_bone_tracker/exports/hand_tracker_int8.onnx',
        calibration_data=calib_data
    )
