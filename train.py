from __future__ import annotations

"""
Main training pipeline orchestrating data preparation, model training, and inference.

Execution flow:
1. Load config from YAML
2. Generate synthetic data if needed
3. Prepare journey records from event logs
4. Create train/val data loaders with augmentation
5. Train LSTM model with optional learning rate scheduling
6. Export predictions for ongoing journeys
"""

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml

from data_loader import create_data_loaders, create_inference_loader
from pipeline_logging import setup_logging
from model import LSTMClassifier
from preprocessing import SUCCESS_EVENT, prepare_from_event_log
from trainer import TrainConfig, fit


LOGGER = logging.getLogger(__name__)


def generate_synthetic_event_log(
    path: Path,
    n_users: int,
    random_state: int,
    success_probability: float,
    max_start_day_offset: int,
) -> None:
    rng = np.random.default_rng(random_state)
    base_events = [
        "campaign_click",
        "browse_products",
        "view_cart",
        "add_to_cart",
        "begin_checkout",
        "application_web_view",
        "site_registration",
        "place_order_web",
        "place_order_phone",
    ]

    # Generate journeys for each user

    rows = []
    for user_id in range(1, n_users + 1):
            # Random journey length and start time
        journey_len = int(rng.integers(3, 10))
        t0 = pd.Timestamp("2026-01-01") + pd.to_timedelta(int(rng.integers(0, max_start_day_offset + 1)), unit="D")

        events = list(rng.choice(base_events, size=journey_len, replace=True))
        # Success depends on the presence of order_shipped in the journey.
        if rng.random() < success_probability:
            insert_idx = int(rng.integers(low=max(1, journey_len - 2), high=journey_len + 1))
            events.insert(min(insert_idx, len(events)), SUCCESS_EVENT)

        for i, event_name in enumerate(events):
            rows.append(
                {
                    "user_id": user_id,
                    "event_time": (t0 + pd.to_timedelta(i * int(rng.integers(1, 8)), unit="m")).isoformat(),
                    "event_name": str(event_name),
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must parse to a dictionary.")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an LSTM to predict journey success (order shipped)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def predict_ongoing_journeys_to_csv(
    model: LSTMClassifier,
    ongoing_loader,
    output_path: Path,
    decision_threshold: float,
) -> None:
    output_columns = ["id", "order_shipped"]

    device = next(model.parameters()).device
    rows = []
    model.eval()

    with torch.no_grad():
        for (
            x_batch,
            time_batch,
            summary_batch,
            lengths_batch,
            user_ids,
            journey_end_times,
            observed_event_counts,
        ) in ongoing_loader:
            x_batch = x_batch.to(device)
            time_batch = time_batch.to(device)
            summary_batch = summary_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            logits = model(x_batch, time_batch, summary_batch, lengths=lengths_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i, user_id in enumerate(user_ids):
                probability_unsuccessful = float(probs[i, 0])
                probability_successful = float(probs[i, 1])
                rows.append(
                    {
                        "id": str(user_id),
                        "order_shipped": probability_successful,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=output_columns).to_csv(output_path, index=False)
    LOGGER.info("Wrote ongoing journey predictions to %s (%d rows).", output_path, len(rows))


def main() -> None:
    started_at = time.perf_counter()

    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    output_cfg = config.get("outputs", {})
    setup_logging(str(output_cfg.get("log_file_path", "logs/train_lstm.log")))
    LOGGER.info("[1/6] Loading config from %s...", config_path)

    # Extract config sections

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    synthetic_cfg = config["synthetic_data"]

    event_log_path = Path(data_cfg["event_log_path"])
    if not event_log_path.exists():
        LOGGER.info("No event log found at %s. Generating synthetic sample data...", event_log_path)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_event_log(
            path=event_log_path,
            n_users=int(synthetic_cfg["n_users"]),
            random_state=int(synthetic_cfg["random_state"]),
            success_probability=float(synthetic_cfg["success_probability"]),
            max_start_day_offset=int(synthetic_cfg["max_start_day_offset"]),
        )

    LOGGER.info("[2/6] Preparing journey records (this can take a while on large files)...")
    prepared = prepare_from_event_log(
        event_log_path=str(event_log_path),
        event_definitions_path=str(data_cfg["event_definitions_path"]),
        max_len=int(data_cfg["max_len"]),
        success_event=str(data_cfg.get("success_event", SUCCESS_EVENT)),
        inactivity_days_for_unsuccessful=int(data_cfg["inactivity_days_for_unsuccessful"]),
        exclude_ongoing_from_training=bool(data_cfg["exclude_ongoing_from_training"]),
        preprocess_n_jobs=int(data_cfg.get("preprocess_n_jobs", 1)),
        cache_enabled=bool(data_cfg.get("cache_enabled", True)),
        cache_path=str(data_cfg.get("cache_path", "data/cache/prepared_data.pkl")),
    )
    LOGGER.info(
        "Prepared %d finished journeys and %d ongoing journeys.",
        len(prepared.records),
        len(prepared.ongoing_records),
    )

    LOGGER.info("[3/6] Building train/validation data loaders and time features...")
    min_truncation_days = int(data_cfg.get("min_truncation_days", data_cfg.get("min_truncation_events", 1)))
    dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_loader, val_loader = create_data_loaders(
        records=prepared.records,
        batch_size=int(train_cfg["batch_size"]),
        val_split=float(data_cfg["val_split"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        truncation_probability=float(data_cfg["truncation_probability"]),
        min_truncation_days=min_truncation_days,
        random_state=int(data_cfg["split_random_state"]),
        num_workers=dataloader_num_workers,
    )

    # Infer the time feature dimension from the data pipeline so config stays in sync.
    inferred_time_feature_dim = None
    try:
        first_batch = next(iter(train_loader))
        # Batch format: (x_padded, time_padded, summary_batch, lengths, y_batch)
        _, time_batch, _, _, _ = first_batch
        if isinstance(time_batch, torch.Tensor) and time_batch.ndim == 3:
            inferred_time_feature_dim = int(time_batch.shape[2])
    except Exception as exc:
        LOGGER.warning("Could not infer time_feature_dim from loader: %s", exc)

    LOGGER.info("[4/6] Initializing model...")
    # Load optimizer config if provided (can be a path to a YAML file or an inline mapping)
    optimizer_cfg = None
    opt_field = train_cfg.get("optimizer")
    if isinstance(opt_field, str):
        opt_path = Path(opt_field)
        if not opt_path.exists():
            LOGGER.warning("Optimizer config path %s does not exist; proceeding with defaults.", opt_path)
        else:
            optimizer_cfg = load_yaml_config(opt_path)
    elif isinstance(opt_field, dict):
        optimizer_cfg = opt_field

    config = TrainConfig(
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        batch_size=int(train_cfg["batch_size"]),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
    )

    model = LSTMClassifier(
        vocab_size=len(prepared.vocab),
        embedding_dim=int(model_cfg["embedding_dim"]),
        time_feature_dim=int(inferred_time_feature_dim or model_cfg.get("time_feature_dim", 7)),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
        hidden_size=int(model_cfg["hidden_size"]),
        summary_feature_dim=int(model_cfg.get("summary_feature_dim", 5)),
        summary_hidden_dim=int(model_cfg.get("summary_hidden_dim", 32)),
        lstm_layers=int(model_cfg["lstm_layers"]),
        bidirectional=bool(model_cfg["bidirectional"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
    )

    LOGGER.info("[5/6] Starting training loop...")
    interrupted = fit(model, train_loader, val_loader, config, optimizer_cfg=optimizer_cfg)

    ongoing_output_path = Path(output_cfg.get("ongoing_predictions_path", "outputs/ongoing_predictions.csv"))
    decision_threshold = float(output_cfg.get("decision_threshold", 0.5))
    if interrupted:
        LOGGER.info("[6/6] Training was stopped manually; exporting current ongoing predictions...")
    else:
        LOGGER.info("[6/6] Running inference for ongoing journeys and writing predictions...")
    ongoing_loader = create_inference_loader(
        records=prepared.ongoing_records,
        batch_size=int(train_cfg["batch_size"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        num_workers=dataloader_num_workers,
    )
    predict_ongoing_journeys_to_csv(
        model=model,
        ongoing_loader=ongoing_loader,
        output_path=ongoing_output_path,
        decision_threshold=decision_threshold,
    )
    elapsed = time.perf_counter() - started_at
        # Report total execution time
    LOGGER.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()