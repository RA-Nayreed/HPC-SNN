"""Complete centralized checkpoints including reproducibility state."""

import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, global_step, best_selection_accuracy=-1.0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "rng": rng_state(),
            "resolved_config": config,
            "model_class": type(model).__name__,
            "epoch": epoch,
            "global_step": global_step,
            "best_selection_accuracy": best_selection_accuracy,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if checkpoint["rng"].get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return checkpoint
