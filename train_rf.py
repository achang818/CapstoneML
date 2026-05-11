from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_loader import build_time_features, split_records_by_time
from pipeline_logging import setup_logging
from preprocessing import CATALOG_MAIL_EVENT, SUCCESS_EVENT, JourneyRecord, inject_catalog_mail_at_truncation, prepare_from_event_log
from train import generate_synthetic_event_log


LOGGER = logging.getLogger(__name__)


def _configure_torch_multiprocessing() -> None:
    """Best-effort guard for PyTorch multiprocessing issues.

    Keeps behavior consistent with train.py even if torch is imported indirectly.
    """
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
    parser = argparse.ArgumentParser(description="Train a Random Forest on journey features.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def _flatten_journey_record(record: JourneyRecord, max_len: int, max_gap_hours: float) -> np.ndarray:
    event_ids = record.event_ids[-max_len:]
    event_times = record.event_times[-max_len:]
    sequence = np.zeros(max_len, dtype=np.int64)
    sequence[-len(event_ids):] = np.asarray(event_ids, dtype=np.int64)

    time_features = build_time_features(
        event_times,
        max_gap_hours,
        journey_start_time=record.event_times[0],
    )
    flattened_time_features = np.zeros((max_len, 2), dtype=np.float32)
    flattened_time_features[-len(time_features):] = time_features

    return np.concatenate([sequence.astype(np.float32), flattened_time_features.reshape(-1)], axis=0)


def _records_to_matrix(records: Sequence[JourneyRecord], max_len: int, max_gap_hours: float) -> np.ndarray:
    return np.vstack([_flatten_journey_record(record, max_len=max_len, max_gap_hours=max_gap_hours) for record in records])


def _build_feature_names(max_len: int) -> List[str]:
    names: List[str] = []
    for i in range(max_len):
        names.append(f"event_id_t{i}")
    for i in range(max_len):
        names.append(f"gap_hours_norm_t{i}")
        names.append(f"elapsed_hours_norm_t{i}")
    return names


def _save_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    LOGGER.info("Saved training dataset to %s", output_path)


def _save_model(model: RandomForestClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    LOGGER.info("Saved RF model to %s", output_path)


def _save_pdp_ice_plots(
    model: RandomForestClassifier,
    x_train: np.ndarray,
    feature_names: Sequence[str],
    random_state: int,
    pdp_path: Path,
    ice_path: Path,
    max_ice_samples: int = 2000,
) -> None:
    top_feature_index = int(np.argmax(model.feature_importances_))
    top_feature_name = feature_names[top_feature_index]
    LOGGER.info("Top feature by importance: %s (index=%d)", top_feature_name, top_feature_index)

    x_train_df = pd.DataFrame(x_train, columns=feature_names)

    pdp_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(
        model,
        x_train_df,
        features=[top_feature_name],
        kind="average",
        ax=ax,
    )
    ax.set_title(f"PDP: {top_feature_name}")
    fig.tight_layout()
    fig.savefig(pdp_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved PDP plot to %s", pdp_path)

    rng = np.random.default_rng(random_state)
    if len(x_train_df) > max_ice_samples:
        idx = rng.choice(len(x_train_df), size=max_ice_samples, replace=False)
        x_ice = x_train_df.iloc[idx]
    else:
        x_ice = x_train_df

    ice_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(
        model,
        x_ice,
        features=[top_feature_name],
        kind="individual",
        subsample=min(200, len(x_ice)),
        random_state=random_state,
        ax=ax,
    )
    ax.set_title(f"ICE: {top_feature_name}")
    fig.tight_layout()
    fig.savefig(ice_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved ICE plot to %s", ice_path)


def _train_rf_and_score(
    records: Sequence[JourneyRecord],
    max_len: int,
    max_gap_hours: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    val_split: float,
) -> Tuple[RandomForestClassifier, float, np.ndarray, np.ndarray]:
    train_records, val_records = split_records_by_time(records, val_split=val_split)
    x_train = _records_to_matrix(train_records, max_len=max_len, max_gap_hours=max_gap_hours)
    y_train = np.asarray([record.label for record in train_records], dtype=np.int64)
    x_val = _records_to_matrix(val_records, max_len=max_len, max_gap_hours=max_gap_hours)
    y_val = np.asarray([record.label for record in val_records], dtype=np.int64)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    LOGGER.info("Cross-validation accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(x_train, y_train)
    val_predictions = model.predict(x_val)
    validation_accuracy = float(accuracy_score(y_val, val_predictions))
    LOGGER.info("Validation accuracy: %.4f", validation_accuracy)
    return model, validation_accuracy, x_train, y_train


def _predict_ongoing_journeys(
    model: RandomForestClassifier,
    ongoing_records,
    max_len: int,
    max_gap_hours: float,
    output_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for record in ongoing_records:
        features = _flatten_journey_record(
            JourneyRecord(
                user_id=record.user_id,
                event_ids=record.event_ids,
                event_times=record.event_times,
                label=0,
                journey_end_time=record.journey_end_time,
            ),
            max_len=max_len,
            max_gap_hours=max_gap_hours,
        )
        probability_successful = float(model.predict_proba(features.reshape(1, -1))[0, 1])
        rows.append({"id": str(record.user_id), "order_shipped": probability_successful})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "order_shipped"]).to_csv(output_path, index=False)
    LOGGER.info("Wrote ongoing journey predictions to %s (%d rows)", output_path, len(rows))


def main() -> None:
    started_at = time.perf_counter()

    _configure_torch_multiprocessing()

    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    output_cfg = config.get("outputs", {})
    data_cfg = config["data"]
    model_cfg = config.get("rf_model", {})
    synthetic_cfg = config["synthetic_data"]

    logger = setup_logging(str(output_cfg.get("log_file_path", "logs/train_rf.log")))
    logger.info("Loading config from %s", config_path)

    event_log_path = Path(data_cfg["event_log_path"])
    if not event_log_path.exists():
        logger.info("No event log found at %s. Generating synthetic sample data...", event_log_path)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_event_log(
            path=event_log_path,
            n_users=int(synthetic_cfg["n_users"]),
            random_state=int(synthetic_cfg["random_state"]),
            success_probability=float(synthetic_cfg["success_probability"]),
            max_start_day_offset=int(synthetic_cfg["max_start_day_offset"]),
        )

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

    rf_settings = {
        "n_estimators": int(model_cfg.get("n_estimators", 300)),
        "max_depth": None if model_cfg.get("max_depth", None) in (None, "null", "None") else int(model_cfg["max_depth"]),
        "min_samples_split": int(model_cfg.get("min_samples_split", 2)),
        "min_samples_leaf": int(model_cfg.get("min_samples_leaf", 1)),
    }

    model, validation_accuracy, _, _ = _train_rf_and_score(
        records=prepared.records,
        max_len=int(data_cfg["max_len"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        random_state=int(data_cfg["split_random_state"]),
        val_split=float(data_cfg["val_split"]),
        **rf_settings,
    )

    # Rebuild training split matrix once so we can save reproducible artifacts.
    train_records, _ = split_records_by_time(prepared.records, val_split=float(data_cfg["val_split"]))
    x_train = _records_to_matrix(
        train_records,
        max_len=int(data_cfg["max_len"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
    )
    y_train = np.asarray([record.label for record in train_records], dtype=np.int64)
    feature_names = _build_feature_names(int(data_cfg["max_len"]))

    model_path = Path(output_cfg.get("rf_model_path", "artifacts/rf_model.joblib"))
    training_data_path = Path(output_cfg.get("rf_training_data_path", "artifacts/rf_training_data.npz"))
    pdp_path = Path(output_cfg.get("rf_pdp_plot_path", "artifacts/rf_top_feature_pdp.png"))
    ice_path = Path(output_cfg.get("rf_ice_plot_path", "artifacts/rf_top_feature_ice.png"))

    _save_model(model, model_path)
    _save_training_data(x_train, y_train, feature_names, training_data_path)
    _save_pdp_ice_plots(
        model=model,
        x_train=x_train,
        feature_names=feature_names,
        random_state=int(data_cfg["split_random_state"]),
        pdp_path=pdp_path,
        ice_path=ice_path,
    )

    output_path = Path(output_cfg.get("ongoing_predictions_path", "outputs/ongoing_predictions.csv"))

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

    _predict_ongoing_journeys(
        model=model,
        ongoing_records=ongoing_records,
        max_len=int(data_cfg["max_len"]),
        max_gap_hours=float(data_cfg["time_feature_max_gap_hours"]),
        output_path=output_path,
    )

    elapsed = time.perf_counter() - started_at
    logger.info("Done in %.1fs | validation accuracy %.4f", elapsed, validation_accuracy)


if __name__ == "__main__":
    main()