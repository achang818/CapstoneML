from __future__ import annotations

"""
Training loop and evaluation for journey success prediction model.

Supports:
- Custom learning rate scheduling (cosine annealing + warmup)
- Gradient clipping for stability
- Keyboard interrupt handling for graceful shutdown
- Metrics computation and logging
"""

from dataclasses import dataclass
import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs.
        lr: Initial learning rate for optimizer.
        batch_size: Batch size for training (for reference; actual batching in data loaders).
        grad_clip_norm: Maximum gradient norm for clipping (prevents exploding gradients).
    """
    epochs: int = 12
    lr: float = 5e-4
    batch_size: int = 64
    grad_clip_norm: float = 1.0


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a loader and compute metrics."""

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, time_batch, summary_batch, lengths_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            time_batch = time_batch.to(device)
            summary_batch = summary_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            logits = model(x_batch, time_batch, summary_batch, lengths=lengths_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.cpu().numpy().tolist())

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
    }

    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    optimizer_cfg: Optional[Dict] = None,
) -> bool:
    """Fit model parameters; returns True if interrupted (Ctrl+C)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Configure optimizer
    weight_decay = 0.0
    if optimizer_cfg and isinstance(optimizer_cfg, dict):
        opt_block = optimizer_cfg.get("optimizer") or optimizer_cfg
        weight_decay = float(opt_block.get("weight_decay", 0.0)) if isinstance(opt_block, dict) else 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=weight_decay)

    # Configure schedulers (epoch-level warmup + main scheduler)
    warmup_scheduler = None
    main_scheduler = None
    warmup_epochs = 0
    if optimizer_cfg and isinstance(optimizer_cfg, dict):
        sched_block = optimizer_cfg.get("scheduler", {})
        if isinstance(sched_block, dict):
            sched_type = sched_block.get("type")
            warmup_epochs = int(sched_block.get("warmup_epochs", 0))
            min_lr = float(sched_block.get("min_lr", 1e-6))
            if sched_type == "cosine":
                # Cosine annealing with optional warmup
                # T_max is number of epochs for cosine decay AFTER warmup
                t_max = sched_block.get("t_max")
                if t_max is None:
                    t_max = max(1, config.epochs - warmup_epochs)
                else:
                    t_max = int(t_max)
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

            if warmup_epochs > 0:
                def _warmup_lambda(e: int, we=warmup_epochs):
                    return float(e + 1) / float(max(1, we)) if e < we else 1.0

                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: float(e + 1) / float(max(1, warmup_epochs)))
    interrupted = False

    try:
        for epoch in range(1, config.epochs + 1):
            # Set model to training mode
            model.train()
            total_loss = 0.0
            if epoch == 1:
                LOGGER.info("Epoch 01 warmup: first batches may be slower while kernels initialize.")

            # Training batches with progress bar

            progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config.epochs:02d}", unit="batch")
            for x_batch, time_batch, summary_batch, lengths_batch, y_batch in progress:
                x_batch = x_batch.to(device)
                time_batch = time_batch.to(device)
                summary_batch = summary_batch.to(device)
                lengths_batch = lengths_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass

                optimizer.zero_grad()
                logits = model(x_batch, time_batch, summary_batch, lengths=lengths_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

                total_loss += float(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(1, len(train_loader))
            metrics = evaluate(model, val_loader, device)
            LOGGER.info("Epoch %02d | loss=%.4f | val_acc=%.4f", epoch, avg_loss, metrics["accuracy"])

            # Step schedulers after epoch
            if warmup_scheduler is not None and epoch <= warmup_epochs:
                warmup_scheduler.step()
            elif main_scheduler is not None:
                main_scheduler.step()
    except KeyboardInterrupt:
        # Graceful shutdown: continue to export predictions using the current model state.
        interrupted = True
        LOGGER.warning("Training interrupted by user. Saving current model state and continuing to prediction export.")

    return interrupted