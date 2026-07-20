# hand_bone_tracker/training/callbacks.py
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
    et fluide. Stratégie :
      * mise à jour de l'ombre (shadow) à chaque batch, avec warmup du decay ;
      * pendant la validation, on bascule sur les poids EMA (les métriques et la
        sélection du meilleur checkpoint reflètent donc l'EMA) ;
      * le checkpoint sauvegardé contient les poids EMA (injection dans state_dict).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._n = 0

    def _float_state(self, pl_module):
        return {k: v for k, v in pl_module.state_dict().items()
                if v.is_floating_point()}

    def on_fit_start(self, trainer, pl_module):
        if not self.shadow:
            self.shadow = {k: v.detach().clone() for k, v in self._float_state(pl_module).items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._n += 1
        d = min(self.decay, (1 + self._n) / (10 + self._n))  # warmup
        state = pl_module.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(d).add_(state[k].detach(), alpha=1.0 - d)

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.shadow:
            return
        state = pl_module.state_dict()
        self.backup = {k: state[k].detach().clone() for k in self.shadow}
        for k in self.shadow:
            state[k].copy_(self.shadow[k])

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.backup:
            return
        state = pl_module.state_dict()
        for k in self.backup:
            state[k].copy_(self.backup[k])
        self.backup = {}

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Le checkpoint sur disque contient les poids EMA.
        if self.shadow:
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

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="hand-tracker-{epoch:02d}-{val_mpjpe_3d:.4f}",
        save_top_k=config["checkpoint"]["save_top_k"],
        verbose=True,
        monitor=config["checkpoint"]["monitor"],
        mode=config["early_stopping"]["mode"],
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
