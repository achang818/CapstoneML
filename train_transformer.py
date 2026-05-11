from __future__ import annotations

"""Train a Transformer encoder model (with positional encoding) for journey success.

This entrypoint mirrors train.py but:
- uses padded batching suitable for Transformer models
- does NOT use summary-statistics features
- uses explicit sinusoidal positional encoding

Run:
    python train_transformer.py --config configs/train_config.yaml
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

from data_loader import create_data_loaders_transformer, create_inference_loader_transformer
from pipeline_logging import setup_logging
from model import TransformerClassifier
from preprocessing import CATALOG_MAIL_EVENT, SUCCESS_EVENT, inject_catalog_mail_at_truncation, prepare_from_event_log
from trainer import TrainConfig, fit_transformer


LOGGER = logging.getLogger(__name__)


def _configure_torch_multiprocessing() -> None:
    """Apply safe defaults for PyTorch multiprocessing."""
    try:
        import torch.multiprocessing as mp

        mp.set_sharing_strategy("file_system")
    except Exception:
        return


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must parse to a dictionary.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer to predict journey success.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def predict_ongoing_journeys_to_csv(
    model: TransformerClassifier,
    ongoing_loader,
    output_path: Path,
) -> None:
    output_columns = ["id", "order_shipped"]

    device = next(model.parameters()).device
    rows = []
    model.eval()

    with torch.no_grad():
        for (
            x,
            time_features,
            src_key_padding_mask,
            lengths_batch,
            user_ids,
            journey_end_times,
            observed_event_counts,
        ) in ongoing_loader:
            x = x.to(device)
            time_features = time_features.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)

            logits = model(x, time_features, src_key_padding_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i, user_id in enumerate(user_ids):
                rows.append({"id": str(user_id), "order_shipped": float(probs[i, 1])})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=output_columns).to_csv(output_path, index=False)
    LOGGER.info("Wrote ongoing journey predictions to %s (%d rows).", output_path, len(rows))


def main() -> None:
    started_at = time.perf_counter()
    _configure_torch_multiprocessing()

    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    output_cfg = config.get("outputs", {})
    setup_logging(str(output_cfg.get("log_file_path", "logs/train_transformer.log")))
    LOGGER.info("[1/6] Loading config from %s...", config_path)

    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    train_cfg = config["training"]

    event_log_path = Path(data_cfg["event_log_path"])
    if not event_log_path.exists():
        raise FileNotFoundError(
            f"Event log not found at {event_log_path}. "
            "Unlike train.py, train_transformer.py does not auto-generate synthetic data."
        )

    LOGGER.info("[2/6] Preparing journey records (cache-aware)...")
    try:
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
    except ValueError as exc:
        LOGGER.error("Preprocessing failed: %s", exc)
        LOGGER.error(
            "To proceed I need an event log with columns for user id, event time, and event name. "
            "Accepted names include user_id/id/customer_id(+account_id), event_time/event_timestamp, and event_name."
        )
        raise

    LOGGER.info(
        "Prepared %d finished journeys and %d ongoing journeys.",
        len(prepared.records),
        len(prepared.ongoing_records),
    )

    LOGGER.info("[3/6] Building train/validation data loaders (no summary stats)...")
    min_truncation_days = int(data_cfg.get("min_truncation_days", data_cfg.get("min_truncation_events", 1)))
    dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))

    train_loader, val_loader = create_data_loaders_transformer(
        records=prepared.records,
        batch_size=int(train_cfg["batch_size"]),
        val_split=float(data_cfg["val_split"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        truncation_probability=float(data_cfg["truncation_probability"]),
        min_truncation_days=min_truncation_days,
        random_state=int(data_cfg["split_random_state"]),
        num_workers=dataloader_num_workers,
    )

    inferred_time_feature_dim = None
    try:
        first_batch = next(iter(train_loader))
        _, time_features, _, _ = first_batch
        inferred_time_feature_dim = int(time_features.shape[-1])
    except Exception as exc:
        LOGGER.warning("Could not infer time_feature_dim from loader: %s", exc)

    LOGGER.info("[4/6] Initializing Transformer model...")

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

    train_config = TrainConfig(
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        batch_size=int(train_cfg["batch_size"]),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
    )

    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    dropout = float(model_cfg.get("dropout", 0.2))
    num_classes = int(model_cfg.get("num_classes", 2))

    model = TransformerClassifier(
        vocab_size=len(prepared.vocab),
        embedding_dim=embedding_dim,
        time_feature_dim=int(inferred_time_feature_dim or model_cfg.get("time_feature_dim", 7)),
        nhead=int(model_cfg.get("transformer_nhead", 4)),
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        dim_feedforward=int(model_cfg.get("transformer_dim_feedforward", 256)),
        num_classes=num_classes,
        dropout=dropout,
        max_len=int(model_cfg.get("transformer_max_len", 4096)),
    )

    LOGGER.info("[5/6] Starting training loop...")
    interrupted = fit_transformer(model, train_loader, val_loader, train_config, optimizer_cfg=optimizer_cfg)

    ongoing_output_path = Path(output_cfg.get("ongoing_predictions_path", "outputs/ongoing_predictions.csv"))
    if interrupted:
        LOGGER.info("[6/6] Training was stopped manually; exporting current ongoing predictions...")
    else:
        LOGGER.info("[6/6] Running inference for ongoing journeys and writing predictions...")

    ongoing_records = prepared.ongoing_records
    if bool(data_cfg.get("counterfactual_catalog_mail_at_truncation", False)):
        catalog_mail_event_id = prepared.vocab.get(CATALOG_MAIL_EVENT)
        if catalog_mail_event_id is None:
            raise ValueError(
                f"counterfactual_catalog_mail_at_truncation is enabled, but '{CATALOG_MAIL_EVENT}' is not present "
                "in event_definitions.csv (so it has no vocab ID)."
            )
        ongoing_records = inject_catalog_mail_at_truncation(
            records=ongoing_records,
            catalog_mail_event_id=int(catalog_mail_event_id),
        )

    ongoing_loader = create_inference_loader_transformer(
        records=ongoing_records,
        batch_size=int(train_cfg["batch_size"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        num_workers=dataloader_num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predict_ongoing_journeys_to_csv(model=model, ongoing_loader=ongoing_loader, output_path=ongoing_output_path)

    elapsed = time.perf_counter() - started_at
    LOGGER.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
