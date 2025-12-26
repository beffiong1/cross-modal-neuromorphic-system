"""
Training pipeline utilities for SNN experiments.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import SupervisedContrastiveLoss
from .settings import TrainingConfig


class TrainingTracker:
    """Logs metrics and handles early stopping + checkpointing."""

    def __init__(self, model_name: str, model: nn.Module, checkpoint_dir: Path, patience: int = 7):
        self.model_name = model_name
        self.model = model
        self.model_path = checkpoint_dir / f"{model_name}_best.pth"
        self.patience = patience
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_ce_loss": [],
            "train_scl_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_ce_loss": [],
            "val_scl_loss": [],
            "best_val_acc": 0.0,
            "epochs_trained": 0,
            "lr": [],
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, epoch: int, metrics: dict) -> bool:
        self.history["train_loss"].append(metrics["train_loss"])
        self.history["train_ce_loss"].append(metrics["train_ce_loss"])
        self.history["train_scl_loss"].append(metrics["train_scl_loss"])
        self.history["train_acc"].append(metrics["train_acc"])
        self.history["val_loss"].append(metrics["val_loss"])
        self.history["val_ce_loss"].append(metrics["val_ce_loss"])
        self.history["val_scl_loss"].append(metrics["val_scl_loss"])
        self.history["val_acc"].append(metrics["val_acc"])
        self.history["lr"].append(metrics["lr"])
        self.history["epochs_trained"] = epoch + 1

        if metrics["val_acc"] > self.best_val_acc:
            self.best_val_acc = metrics["val_acc"]
            self.history["best_val_acc"] = self.best_val_acc
            self.epochs_no_improve = 0
            self.save_checkpoint(epoch)
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve < self.patience

    def save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_acc": self.best_val_acc,
            },
            self.model_path,
        )

    def get_history(self) -> Dict[str, Any]:
        return self.history


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    con_loss_fn: Optional[nn.Module],
    contrastive_weight: float,
    device: torch.device,
    gradient_clip: float,
) -> Tuple[float, float, float, float]:
    total_loss = total_ce_loss = total_con_loss = 0.0
    total_correct = total_samples = 0
    model.train()

    pbar = tqdm(data_loader, desc="[Train]", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        spk_sum, features = model(data)

        loss_ce = ce_loss_fn(spk_sum, labels)
        if contrastive_weight > 0 and con_loss_fn is not None:
            loss_con = con_loss_fn(features, labels)
            loss = loss_ce + contrastive_weight * loss_con
        else:
            loss = loss_ce
            loss_con = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_ce_loss += loss_ce.item() * data.size(0)
        total_con_loss += loss_con.item() * data.size(0)
        total_correct += (spk_sum.argmax(1) == labels).sum().item()
        total_samples += data.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * total_correct / total_samples:.2f}%")

    avg_loss = total_loss / (total_samples + 1e-9)
    avg_ce_loss = total_ce_loss / (total_samples + 1e-9)
    avg_con_loss = total_con_loss / (total_samples + 1e-9)
    avg_acc = (total_correct / (total_samples + 1e-9)) * 100
    return avg_loss, avg_ce_loss, avg_con_loss, avg_acc


def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    ce_loss_fn: nn.Module,
    con_loss_fn: Optional[nn.Module],
    contrastive_weight: float,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    total_loss = total_ce_loss = total_con_loss = 0.0
    total_correct = total_samples = 0
    model.eval()

    pbar = tqdm(data_loader, desc="[Val]", leave=False)
    with torch.no_grad():
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            spk_sum, features = model(data)

            loss_ce = ce_loss_fn(spk_sum, labels)
            if contrastive_weight > 0 and con_loss_fn is not None:
                loss_con = con_loss_fn(features, labels)
                loss = loss_ce + contrastive_weight * loss_con
            else:
                loss = loss_ce
                loss_con = torch.tensor(0.0, device=device)

            total_loss += loss.item() * data.size(0)
            total_ce_loss += loss_ce.item() * data.size(0)
            total_con_loss += loss_con.item() * data.size(0)
            total_correct += (spk_sum.argmax(1) == labels).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / (total_samples + 1e-9)
    avg_ce_loss = total_ce_loss / (total_samples + 1e-9)
    avg_con_loss = total_con_loss / (total_samples + 1e-9)
    avg_acc = (total_correct / (total_samples + 1e-9)) * 100
    return avg_loss, avg_ce_loss, avg_con_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    use_contrastive: bool,
    device: torch.device,
    config: TrainingConfig,
) -> Tuple[nn.Module, Dict[str, Any], float]:
    lr = config.learning_rate
    scl_weight = config.contrastive_weight if use_contrastive else 0.0

    ce_loss_fn = nn.CrossEntropyLoss()
    con_loss_fn = None
    if use_contrastive:
        con_loss_fn = SupervisedContrastiveLoss(
            temperature=config.contrastive_temperature
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    tracker = TrainingTracker(model_name, model, config.checkpoint_dir, config.patience)

    start_time = time.time()
    for epoch in range(config.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_ce, train_scl, train_acc = train_step(
            model,
            train_loader,
            optimizer,
            ce_loss_fn,
            con_loss_fn,
            scl_weight,
            device,
            config.gradient_clip,
        )
        val_loss, val_ce, val_scl, val_acc = test_step(
            model, test_loader, ce_loss_fn, con_loss_fn, scl_weight, device
        )
        scheduler.step()

        if not tracker.log_epoch(
            epoch,
            {
                "train_loss": train_loss,
                "train_ce_loss": train_ce,
                "train_scl_loss": train_scl,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_ce_loss": val_ce,
                "val_scl_loss": val_scl,
                "val_acc": val_acc,
                "lr": current_lr,
            },
        ):
            break

    total_time = (time.time() - start_time) / 60
    print(
        f"Training finished: {model_name} on {dataset_name} "
        f"({tracker.best_val_acc:.2f}% best, {total_time:.2f} min)"
    )
    return model, tracker.history, tracker.best_val_acc
