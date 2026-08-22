import pytest
import torch
from models.losses.combined_loss import CombinedHandLoss

def test_generate_heatmaps():
    criterion = CombinedHandLoss()
    coords_2d = torch.tensor([[[0.5, 0.5]] * 21]) # (1, 21, 2)
    heatmaps = criterion.generate_gt_heatmaps(coords_2d)
    
    assert heatmaps.shape == (1, 21, 64, 64)
    # The center of each heatmap (index 32, 32) should have high response
    for j in range(21):
        assert heatmaps[0, j, 32, 32].item() > 0.9

def test_scale_invariant_3d_loss_positivity():
    criterion = CombinedHandLoss()
    pred_3d = torch.randn(2, 21, 3)
    target_3d = torch.randn(2, 21, 3)
    
    loss = criterion.scale_invariant_3d_loss(pred_3d, target_3d)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

def test_bone_length_loss_zero_on_target():
    criterion = CombinedHandLoss()
    target_3d = torch.randn(2, 21, 3)
    loss = criterion.bone_length_loss(target_3d, target_3d)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_reprojection_loss():
    criterion = CombinedHandLoss()
    pred_3d = torch.randn(2, 21, 3)
    target_2d = torch.rand(2, 21, 2)
    
    loss = criterion.reprojection_loss(pred_3d, target_2d)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

def test_combined_loss_backward():
    criterion = CombinedHandLoss(
        lambda_heatmap=1.0,
        lambda_coord2d=5.0,
        lambda_3d=2.0,
        lambda_3d_norm=1.0,
        lambda_bone=0.5,
        lambda_reproj=0.5
    )
    
    preds = {
        "heatmaps": torch.randn(2, 21, 64, 64, requires_grad=True),
        "coords_2d": torch.rand(2, 21, 2, requires_grad=True),
        "joints_3d": torch.randn(2, 21, 3, requires_grad=True),
        "joints_3d_raw": torch.randn(2, 21, 3, requires_grad=True)
    }
    gt_joints_3d = torch.randn(2, 21, 3)
    gt_coords_2d = torch.rand(2, 21, 2)
    
    losses = criterion(preds, gt_joints_3d=gt_joints_3d, gt_coords_2d=gt_coords_2d)
    assert "total" in losses
    assert "heatmap" in losses
    assert "coord2d" in losses
    assert "3d" in losses
    assert "bone" in losses
    assert "reproj" in losses
    
    losses["total"].backward()
    assert preds["heatmaps"].grad is not None
    assert preds["coords_2d"].grad is not None
    assert preds["joints_3d"].grad is not None
