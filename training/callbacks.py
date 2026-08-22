import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)


class EMACallback(Callback):
    """
    Moyenne mobile exponentielle (EMA) des poids.

    Les poids EMA sont plus lisses et généralisent mieux -> tracking plus fiable
    et fluide. Stratégie optimisée sans recréer le state_dict complet à chaque batch.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._n = 0

    def on_fit_start(self, trainer, pl_module):
        if not self.shadow:
            self.shadow = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
                if param.dtype.is_floating_point
            }
            self.shadow.update({
                name: buf.detach().clone()
                for name, buf in pl_module.named_buffers()
                if buf.dtype.is_floating_point
            })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._n += 1
        d = min(self.decay, (1 + self._n) / (10 + self._n))  # warmup
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.shadow and param.dtype.is_floating_point:
                    self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)
            for name, buf in pl_module.named_buffers():
                if name in self.shadow and buf.dtype.is_floating_point:
                    self.shadow[name].mul_(d).add_(buf.data, alpha=1.0 - d)

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.shadow:
            return
        self.backup = {}
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])
            for name, buf in pl_module.named_buffers():
                if name in self.shadow:
                    self.backup[name] = buf.data.clone()
                    buf.data.copy_(self.shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
            for name, buf in pl_module.named_buffers():
                if name in self.backup:
                    buf.data.copy_(self.backup[name])
        self.backup = {}

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Le checkpoint sur disque contient les poids EMA.
        if self.shadow and "state_dict" in checkpoint:
            for k, v in self.shadow.items():
                if k in checkpoint["state_dict"]:
                    checkpoint["state_dict"][k] = v.detach().clone()

    def state_dict(self):
        return {"decay": self.decay, "n": self._n,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        self._n = state.get("n", 0)
        self.shadow = {k: v.clone() for k, v in state.get("shadow", {}).items()}


def get_callbacks(config):
    callbacks = []

    mode = config.get("checkpoint", {}).get("mode", config.get("early_stopping", {}).get("mode", "min"))
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="hand-tracker-{epoch:02d}-{val_mpjpe_3d:.4f}",
        save_top_k=config["checkpoint"]["save_top_k"],
        verbose=True,
        monitor=config["checkpoint"]["monitor"],
        mode=mode,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor=config["early_stopping"]["metric"],
        patience=config["early_stopping"]["patience"],
        mode=config["early_stopping"]["mode"],
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # EMA activée par défaut (désactivable via config: ema.enabled = false)
    ema_cfg = config.get("ema", {}) if isinstance(config, dict) else {}
    if ema_cfg.get("enabled", True):
        callbacks.append(EMACallback(decay=ema_cfg.get("decay", 0.999)))

    return callbacks
