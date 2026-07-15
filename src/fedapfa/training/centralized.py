"""Deterministic single-device centralized training and evaluation."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedapfa.datasets.sequence_collation import EventBatch, collate_event_sequences
from fedapfa.training.checkpointing import load_checkpoint, save_checkpoint


class DeviceUnavailableError(RuntimeError):
    pass


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("configuration requests CUDA, but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, config, shuffle):
    if dataset is None:
        return None
    training = config["training"]
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["data_loader_workers"],
        persistent_workers=training["persistent_workers"],
        pin_memory=config["device"] == "cuda",
        collate_fn=collate_event_sequences,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_optimizer(model, config):
    training = config["training"]
    delay = list(model.delay_parameters()) if hasattr(model, "delay_parameters") else []
    delay_ids = {id(parameter) for parameter in delay}
    ordinary = [parameter for parameter in model.parameters() if id(parameter) not in delay_ids]
    groups = [{"params": ordinary, "lr": training["learning_rate"], "name": "weights"}]
    if delay:
        groups.append(
            {
                "params": delay,
                "lr": training["learning_rate"] * training["delay_lr_multiplier"],
                "name": "delay_positions",
            }
        )
    return torch.optim.Adam(groups, weight_decay=training["weight_decay"])


def _move(batch: EventBatch, device, non_blocking):
    return EventBatch(*(value.to(device, non_blocking=non_blocking) for value in batch))


def _attention_stats(model):
    values = {}
    for name, module in model.named_modules():
        stats = getattr(module, "last_statistics", None)
        if stats is not None:
            values[name] = stats
    return values


def run_epoch(model, loader, device, optimizer=None, max_batches=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total = correct = loss_sum = batches = 0
    rate_sums = {}
    rate_weights = {}
    attention_sums = {}
    attention_weights = {}
    non_blocking = device.type == "cuda"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move(batch, device, non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, rates = model(batch.inputs, batch.lengths)
            loss = criterion(logits, batch.labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            count = len(batch.labels)
            total += count
            correct += int((logits.argmax(1) == batch.labels).sum())
            loss_sum += float(loss.detach()) * count
            batches += 1
            valid_steps = int(batch.valid_mask.sum())
            for name, value in rates.items():
                rate_sums[name] = rate_sums.get(name, 0.0) + float(value.detach()) * valid_steps
                rate_weights[name] = rate_weights.get(name, 0) + valid_steps
            for module_name, statistics in _attention_stats(model).items():
                sums = attention_sums.setdefault(module_name, {})
                for statistic, value in statistics.items():
                    sums[statistic] = sums.get(statistic, 0.0) + value * valid_steps
                attention_weights[module_name] = attention_weights.get(module_name, 0) + valid_steps
    if not total:
        raise RuntimeError("loader produced no batches")
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "batches": batches,
        "examples": total,
        "spike_rates": {name: value / rate_weights[name] for name, value in rate_sums.items()},
        "attention": {
            module_name: {statistic: value / attention_weights[module_name] for statistic, value in statistics.items()}
            for module_name, statistics in attention_sums.items()
        },
    }


def train_centralized(model, bundle, config, run_dir, resume_checkpoint=None):
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    model.to(device)
    optimizer = make_optimizer(model, config)
    scheduler = None
    training = config["training"]
    start_epoch = global_step = 0
    best = -1.0
    epochs_without_improvement = 0
    if resume_checkpoint:
        state = load_checkpoint(resume_checkpoint, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best = state.get("best_selection_accuracy", -1.0)
    train_loader = make_loader(bundle.train, config, True)
    validation_loader = make_loader(bundle.validation, config, False)
    path = Path(run_dir)
    checkpoint_dir = path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"fedapfa.{path.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path / "training.log", mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("starting %s on %s with %s", config["name"], device, type(model).__name__)
    metrics_path = path / "metrics.jsonl"
    accepted = False
    final = {}
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, training["epochs"]):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, training["max_train_batches"], training["gradient_clip"]
            )
            global_step += train_metrics["batches"]
            validation_metrics = (
                run_epoch(model, validation_loader, device, None, training["max_validation_batches"])
                if validation_loader
                else None
            )
            selection = validation_metrics["accuracy"] if validation_metrics else train_metrics["accuracy"]
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "epoch_duration_seconds": time.monotonic() - started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            }
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            logger.info(
                "epoch=%d train_accuracy=%.4f validation_accuracy=%s",
                epoch,
                train_metrics["accuracy"],
                None if validation_metrics is None else f"{validation_metrics['accuracy']:.4f}",
            )
            if selection > best:
                best = selection
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best_validation.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best,
                )
            else:
                epochs_without_improvement += 1
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, config, epoch, global_step, best)
            if config["mode"] == "tiny_overfit" and train_metrics["accuracy"] >= training["target_accuracy"]:
                accepted = True
                logger.info("tiny-overfit target reached at epoch %d", epoch)
                break
            patience = training.get("early_stop_patience")
            if patience is not None and epochs_without_improvement >= patience:
                logger.info("early stopping after %d epochs without improvement", patience)
                break
    if config["mode"] == "tiny_overfit" and not accepted:
        logger.error("tiny-overfit target %.3f was not reached", training["target_accuracy"])
    if config["mode"] == "smoke":
        accepted = True
    best_path = checkpoint_dir / "best_validation.pt"
    if bundle.test:
        load_checkpoint(best_path, model)
        test_dataset = bundle.test() if callable(bundle.test) else bundle.test
        bundle.metadata["official_test_accessed"] = True
        if bundle.selected_indices:
            (path / "selected_indices.json").write_text(
                json.dumps(bundle.selected_indices, indent=2, sort_keys=True), encoding="utf-8"
            )
        test_loader = make_loader(test_dataset, config, False)
        final["test"] = run_epoch(model, test_loader, device, None, training["max_test_batches"])
    final.update(
        {
            "accepted": accepted if config["mode"] in {"tiny_overfit", "smoke"} else True,
            "best_selection_accuracy": best,
            "model_class": type(model).__name__,
            "model_metadata": model.model_metadata,
            "protocol": bundle.metadata,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
    )
    (path / "final_metrics.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    (path / "acceptance.json").write_text(
        json.dumps(
            {
                "mode": config["mode"],
                "accepted": final["accepted"],
                "scientific_result": bundle.metadata["scientific_result"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("finished accepted=%s", final["accepted"])
    return final


train = train_centralized
evaluate = run_epoch
