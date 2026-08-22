import pytest
import numpy as np
from inference.kalman_filter import OneEuroFilter, HandKalmanFilter, _HAS_FILTERPY

def test_one_euro_filter_2d():
    filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
    
    noisy_signal = [np.array([10.0, 20.0]) + np.random.normal(0, 0.5, 2) for _ in range(30)]
    filtered_signal = []
    
    for i, s in enumerate(noisy_signal):
        out = filt(s, dt=1.0 / 30.0)
        filtered_signal.append(out)
        
    assert len(filtered_signal) == 30
    assert filtered_signal[-1].shape == (2,)

def test_one_euro_filter_adaptive_cutoff():
    filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.4)
    val1 = np.zeros((21, 3))
    conf = np.ones(21) * 0.9
    
    out = filt(val1, confidence=conf, dt=1.0 / 30.0)
    assert out.shape == (21, 3)

def test_hand_kalman_filter():
    if not _HAS_FILTERPY:
        pytest.skip("filterpy not installed")
    kf = HandKalmanFilter(fps=30.0)
    
    noisy_3d = np.random.randn(21, 3).astype(np.float32)
    
    pred_3d = kf.update(noisy_3d)
    assert pred_3d.shape == (21, 3)
