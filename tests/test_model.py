import pytest
import torch
from models.hand_tracker import HandBoneTracker, BiomechanicalFKLayer, JOINT_PARENTS, MAX_BONE_LENGTH

def test_model_initialization():
    model = HandBoneTracker(backbone_name="mobilenetv3_large_100", num_keypoints=21, pretrained=False)
    assert model is not None
    assert model.num_keypoints == 21

def test_model_forward_pass_cpu():
    model = HandBoneTracker(backbone_name="mobilenetv3_large_100", num_keypoints=21, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert "heatmaps" in out
    assert "coords_2d" in out
    assert "joints_3d" in out
    assert "joints_3d_raw" in out
    assert "confidence" in out

    assert out["heatmaps"].shape == (2, 21, 64, 64)
    assert out["coords_2d"].shape == (2, 21, 2)
    assert out["joints_3d"].shape == (2, 21, 3)
    assert out["joints_3d_raw"].shape == (2, 21, 3)
    assert out["confidence"].shape == (2, 21)

def test_softargmax_bounds_and_gradients():
    model = HandBoneTracker(backbone_name="mobilenetv3_large_100", num_keypoints=21, pretrained=False)
    heatmaps = torch.randn(2, 21, 64, 64, requires_grad=True)
    coords, probas = model.heatmaps_to_coords(heatmaps)
    
    assert coords.shape == (2, 21, 2)
    assert (coords >= 0.0).all() and (coords <= 1.0).all()

    loss = coords.sum()
    loss.backward()
    assert heatmaps.grad is not None
    assert not torch.isnan(heatmaps.grad).any()

def test_biomechanical_fk_layer():
    fk = BiomechanicalFKLayer()
    # Create raw joints with wrist at origin
    raw_joints = torch.randn(4, 21, 3)
    raw_joints[:, 0, :] = 0.0
    refined = fk(raw_joints)

    assert refined.shape == (4, 21, 3)
    # Wrist (joint 0) should remain at origin
    assert torch.allclose(refined[:, 0, :], torch.zeros(4, 3), atol=1e-5)

    # Verify bone lengths are bounded by MAX_BONE_LENGTH
    for child in range(1, 21):
        parent = JOINT_PARENTS[child]
        bone_vec = refined[:, child] - refined[:, parent]
        length = torch.norm(bone_vec, dim=-1)
        expected_max = MAX_BONE_LENGTH[child]
        assert (length <= expected_max + 1e-4).all()

def test_confidence_range():
    model = HandBoneTracker(backbone_name="mobilenetv3_large_100", num_keypoints=21, pretrained=False)
    heatmaps = torch.randn(2, 21, 64, 64)
    conf = model._confidence_from_heatmaps(heatmaps)
    assert conf.shape == (2, 21)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()
