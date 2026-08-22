# hand_bone_tracker/scripts/train.py
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import yaml
from pathlib import Path
import argparse
import sys
import os

# Racine du projet dans le path -> permet `python scripts/train.py` directement.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.trainer import HandTrackerLightning
from training.callbacks import get_callbacks
from datasets.freihand_dataset import FreiHANDDataset
from torch.utils.data import DataLoader
from datasets.augmentation import train_transform, val_transform

class ConsoleLoggingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n[IA TRAINER] 🔄 Début de l'Époque {trainer.current_epoch}...", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:
            loss_val = None
            if outputs is not None:
                if isinstance(outputs, dict):
                    loss_val = outputs.get("loss") or outputs.get("total")
                elif torch.is_tensor(outputs):
                    loss_val = outputs
            
            if loss_val is None:
                metrics = trainer.callback_metrics
                loss_val = metrics.get("train_total") or metrics.get("train_loss")
                
            if loss_val is not None and hasattr(loss_val, "item"):
                loss_val = loss_val.item()
                
            loss_str = f"loss={loss_val:.4f}" if loss_val is not None else ""
            total_batches = trainer.num_training_batches
            
            if total_batches and total_batches != float('inf') and total_batches > 0:
                percent = (batch_idx / total_batches) * 100
                progress_str = f"{batch_idx}/{total_batches} ({percent:.0f}%)"
            else:
                progress_str = f"étape {batch_idx}"
                
            print(f"[IA TRAINER] 📈 Époque {trainer.current_epoch} : {progress_str} - {loss_str}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        loss_val = metrics.get("train_total")
        loss_str = f"loss={loss_val:.4f}" if loss_val is not None else ""
        print(f"[IA TRAINER] ⏳ Époque {trainer.current_epoch} terminée. {loss_str}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_total")
        mpjpe = metrics.get("val_mpjpe_3d")
        
        val_str = []
        if val_loss is not None:
            val_str.append(f"val_loss={val_loss:.4f}")
        if mpjpe is not None:
            val_str.append(f"MPJPE={mpjpe * 1000:.1f}mm")
            
        metrics_str = ", ".join(val_str)
        print(f"[IA TRAINER] 📊 Validation de l'Époque {trainer.current_epoch} terminée : {metrics_str}\n", flush=True)

def main(config_path, resume=False):
    print("\n" + "="*50)
    print(f"🚀 [IA TRAINER] Démarrage avec config: {config_path}")
    if resume:
        print("🔄 [IA TRAINER] Mode reprise (resume) activé.")
    print("="*50)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Optimisation pour RTX 4070
    torch.set_float32_matmul_precision('medium')
    
    # 1. Logger
    logger = CSVLogger("logs", name=f"stage-{config['stage']}")
    
    # 2. Datasets & Dataloaders
    print("📦 [IA TRAINER] Chargement des datasets...")
    if config['dataset'] == 'freihand':
        train_ds = FreiHANDDataset('data/processed/freihand_train.h5', transform=train_transform)
        val_ds = FreiHANDDataset('data/processed/freihand_val.h5', transform=val_transform)
        
        # Check for custom labeled data to integrate it!
        from datasets.custom_dataset import CustomHandDataset
        from torch.utils.data import ConcatDataset
        try:
            custom_ds = CustomHandDataset(transform=train_transform)
            if len(custom_ds) > 0:
                # Oversample custom data to make it visible in training mix (target around 5000 samples)
                oversample_factor = max(1, 5000 // len(custom_ds))
                print(f"🔗 [IA TRAINER] Intégration de {len(custom_ds)} données personnalisées à l'entraînement (suréchantillonnage: x{oversample_factor}).")
                datasets_to_concat = [train_ds]
                for _ in range(oversample_factor):
                    datasets_to_concat.append(custom_ds)
                train_ds = ConcatDataset(datasets_to_concat)
        except Exception as e:
            print(f"⚠️ [IA TRAINER] Impossible de charger le dataset personnalisé: {e}")
            
    elif config['dataset'] == 'custom':
        from datasets.custom_dataset import CustomHandDataset
        import random
        
        # Check size first
        temp_ds = CustomHandDataset(transform=None)
        if len(temp_ds) == 0:
            raise ValueError("❌ [IA TRAINER] Le dataset personnalisé est vide. Veuillez d'abord capturer des poses sous la webcam dans l'onglet 'My Data'.")
            
        train_ds = CustomHandDataset(transform=train_transform)
        val_ds = CustomHandDataset(transform=val_transform)
        
        # Split (90% train, 10% val)
        n_samples = len(temp_ds)
        val_size = max(1, int(n_samples * 0.1))
        train_size = n_samples - val_size
        
        indices = list(range(n_samples))
        random.seed(42)
        random.shuffle(indices)
        
        train_ds = torch.utils.data.Subset(train_ds, indices[:train_size])
        val_ds = torch.utils.data.Subset(val_ds, indices[train_size:])
        print(f"📦 [IA TRAINER] Entraînement exclusif sur données personnalisées (My Data). Train: {len(train_ds)}, Val: {len(val_ds)}")
        
    elif config['dataset'] == 'synthetic':
        train_ds = FreiHANDDataset('data/processed/synthetic_train.h5', transform=train_transform)
        val_ds = FreiHANDDataset('data/processed/synthetic_val.h5', transform=val_transform)
        print("📦 [IA TRAINER] Chargement de notre propre dataset synthétique (RHD fait maison).")
    else:
        # Fallback to FreiHAND
        train_ds = FreiHANDDataset('data/processed/freihand_train.h5', transform=train_transform)
        val_ds = FreiHANDDataset('data/processed/freihand_val.h5', transform=val_transform)
        
    # Set num_workers to 0 on Windows to avoid process-kill shared memory leaks and pagefile growth
    num_workers = 0 if os.name == 'nt' else config.get('num_workers', 4)
    persistent_workers = num_workers > 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    # 3. Model
    print("🧠 [IA TRAINER] Initialisation du modèle Lightning...")
    model = HandTrackerLightning(config)
    
    # 4. Trainer
    print("⚙️ [IA TRAINER] Configuration du Trainer (RTX 4070 - FP16 Mixed)...")
    
    # Barre de progression explicite
    progress_bar = RichProgressBar(leave=True)
    callbacks = get_callbacks(config)
    callbacks.append(progress_bar)
    callbacks.append(ConsoleLoggingCallback())
    
    precision = '16-mixed' if torch.cuda.is_available() else '32'
    trainer = L.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
        log_every_n_steps=10,
    )
    
    # Détection automatique du dernier checkpoint pour reprise
    ckpt_path = None
    if resume:
        ckpt_dir = Path('checkpoints')
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob('**/*.ckpt'))
            if ckpts:
                # Trier par date de modification (le plus récent en premier)
                ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest_ckpt_path = str(ckpts[0])
                try:
                    checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
                    ckpt_stage = checkpoint.get('hyper_parameters', {}).get('config', {}).get('stage', None)
                    current_stage = config['stage']
                    
                    if ckpt_stage == current_stage:
                        # Même stage : reprise complète (époque, optimiseur et scheduler préservés)
                        ckpt_path = latest_ckpt_path
                        print(f"🔄 [IA TRAINER] Reprise complète du Stage {current_stage} depuis le checkpoint : {latest_ckpt_path}")
                    else:
                        # Stage différent : restaurer uniquement les poids, réinitialiser l'époque à 0
                        print(f"🔄 [IA TRAINER] Transition du Stage {ckpt_stage} vers le Stage {current_stage}.")
                        print(f"📦 [IA TRAINER] Chargement des poids du modèle depuis le checkpoint : {latest_ckpt_path}")
                        state_dict = checkpoint.get('state_dict', checkpoint)
                        model.load_state_dict(state_dict)
                        print("✅ [IA TRAINER] Poids restaurés avec succès pour la nouvelle étape (démarrage à l'époque 0).")
                        ckpt_path = None
                except Exception as e:
                    print(f"⚠️ [IA TRAINER] Erreur lors de l'analyse du checkpoint : {e}")
                    ckpt_path = None
            else:
                print("⚠️ [IA TRAINER] Aucun checkpoint trouvé dans checkpoints/. Démarrage à partir de zéro.")
        else:
            print("⚠️ [IA TRAINER] Dossier checkpoints/ introuvable. Démarrage à partir de zéro.")
    
    # 5. Train
    print("\n🎬 [IA TRAINER] Début de trainer.fit()...\n")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print("\n✅ [IA TRAINER] Entraînement terminé avec succès !")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training/stage1_pretrain.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    
    main(args.config, resume=args.resume)
