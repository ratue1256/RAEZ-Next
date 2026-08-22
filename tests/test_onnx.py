import pytest
import torch
from scripts.export_onnx import ExportWrapper
from models.hand_tracker import HandBoneTracker

def test_export_wrapper():
    model = HandBoneTracker(backbone_name="mobilenetv3_large_100", num_keypoints=21, pretrained=False)
    wrapper = ExportWrapper(model)
    wrapper.eval()
    
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        heatmaps, coords_2d, joints_3d, confidence = wrapper(x)
        
    assert heatmaps.shape == (1, 21, 64, 64)
    assert coords_2d.shape == (1, 21, 2)
    assert joints_3d.shape == (1, 21, 3)
    assert confidence.shape == (1, 21)
