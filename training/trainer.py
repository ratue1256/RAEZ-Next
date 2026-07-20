# hand_bone_tracker/training/trainer.py
import pytorch_lightning as L
import torch
from models.hand_tracker import HandBoneTracker
from models.losses.combined_loss import CombinedHandLoss

class HandTrackerLightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        backbone_name = config.get('backbone', 'efficientnet_b0')
        self.model = HandBoneTracker(
            backbone_name=backbone_name,
            pretrained=True,
            freeze_backbone_epochs=config['freeze_backbone_epochs']
        )
        loss_cfg = config['loss']
        self.loss_fn = CombinedHandLoss(
            lambda_heatmap=loss_cfg.get('lambda_heatmap', 1.0),
            lambda_coord2d=loss_cfg.get('lambda_coord2d', 1.0),
            lambda_3d=loss_cfg.get('lambda_3d', 2.0),
            lambda_3d_norm=loss_cfg.get('lambda_3d_norm', 1.0),
            lambda_bone=loss_cfg.get('lambda_bone', 0.5),
            lambda_temporal=loss_cfg.get('lambda_temporal', 0.0),
        )
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_3d = batch['joints_3d']
        gt_2d = batch['coords_2d']
        
        preds = self.model(images)
        losses = self.loss_fn(preds, gt_3d, gt_2d)
        
        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=True)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_3d = batch['joints_3d']
        gt_2d = batch['coords_2d']
        
        preds = self.model(images)
        losses = self.loss_fn(preds, gt_3d, gt_2d)

        # Calculer MPJPE (Mean Per Joint Position Error)
        mpjpe = torch.norm(preds['joints_3d'] - gt_3d, dim=-1).mean()

        self.log_dict({f"val_{k}": v for k, v in losses.items()}, prog_bar=True)
        self.log("val_mpjpe_3d", mpjpe, prog_bar=True)
        self.log("val_confidence", preds['confidence'].mean(), prog_bar=False)
        return mpjpe

    def configure_optimizers(self):
        optimizer_name = self.config['optimizer'].get('name', 'AdamW')
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay'],
                betas=self.config['optimizer'].get('betas', (0.9, 0.999))
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config['optimizer']['lr']
            )
            
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config.get('eta_min', 0.00001)
            )
            return [optimizer], [scheduler]
        elif scheduler_name == 'OneCycleLR':
            total_steps = int(self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config['max_lr'],
                total_steps=total_steps,
                pct_start=scheduler_config.get('pct_start', 0.1),
                div_factor=scheduler_config.get('div_factor', 25.0),
                final_div_factor=scheduler_config.get('final_div_factor', 10000.0)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer

    def on_train_epoch_start(self):
        if self.current_epoch == self.model.freeze_backbone_epochs:
            self.model.unfreeze_backbone()
            print("Backbone unfrozen")
        elif self.current_epoch < self.model.freeze_backbone_epochs:
            self.model.freeze_backbone()
