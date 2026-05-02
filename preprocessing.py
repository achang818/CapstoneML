"""
Event log preprocessing and journey record building.

This module provides utilities to:
- Normalize event logs with flexible column naming
- Build vocabulary from event names
- Create journey records with binary labels (success/unsuccessful)
- Handle ongoing journeys separately
- Cache prepared data for efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

# Special tokens for embedding
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Events and labels
SUCCESS_EVENT = "order_shipped"
LABEL_UNSUCCESSFUL = 0
LABEL_SUCCESSFUL = 1



@dataclass
class JourneyRecord:
    """A journey record with binary label for training data.
    
    Attributes:
        user_id: Unique identifier for the user.
        event_ids: List of encoded event IDs in the journey.
        event_times: List of timestamps for each event.
        label: Binary label (0=unsuccessful, 1=successful).
        journey_end_time: Timestamp of the last event in the journey.
    """
    user_id: Any
    event_ids: List[int]
    event_times: List[pd.Timestamp]
    label: int
    journey_end_time: pd.Timestamp


@dataclass
class OngoingJourneyRecord:
    """A journey record for ongoing (unlabeled) journeys for inference.
    
    Attributes:
        user_id: Unique identifier for the user.
        event_ids: List of encoded event IDs in the journey.
        event_times: List of timestamps for each event.
        journey_end_time: Timestamp of the last observed event.
    """
    user_id: Any
    event_ids: List[int]
    event_times: List[pd.Timestamp]
    journey_end_time: pd.Timestamp


@dataclass
class PreparedData:
    """Container for all prepared training and inference data.
    
    Attributes:
        records: Finished journeys labeled for training.
        ongoing_records: Active journeys without labels for inference.
        vocab: Mapping from event names to integer IDs.
        max_len: Maximum sequence length in the dataset.
    """
    records: List[JourneyRecord]
    ongoing_records: List[OngoingJourneyRecord]
    vocab: Dict[str, int]
    max_len: int


# ==================== Utility Functions ====================


def get_label_mapping() -> Dict[int, str]:
    """Return human-readable labels for binary classification."""
    return {LABEL_UNSUCCESSFUL: "unsuccessful", LABEL_SUCCESSFUL: "successful"}


def load_event_definitions(path: str) -> pd.DataFrame:
    """Load event definitions from CSV."""
    return pd.read_csv(path)


def build_vocab(events: Iterable[str]) -> Dict[str, int]:
    """Build vocabulary mapping from unique event names.
    
    Special tokens (PAD, UNK) are assigned IDs 0 and 1. Event names are sorted
    and assigned consecutive IDs starting from 2 for reproducibility.
    
    Args:
        events: Iterable of event names.
        
    Returns:
        Dictionary mapping event names to integer IDs.
    """
    unique_events = sorted(set(events))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, event_name in enumerate(unique_events, start=2):
        vocab[event_name] = idx
    return vocab


# ==================== Column Normalization ====================


def _validate_required_columns(events_df: pd.DataFrame) -> None:
    """Validate that required columns exist in the event log."""
    required = {"user_id", "event_time", "event_name"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _find_first_present_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Find the first matching column from candidates (case-insensitive).
    
    Args:
        df: DataFrame to search.
        candidates: List of column names to match (in order of preference).
        
    Returns:
        The actual column name if found, None otherwise.
    """
    lookup = {column.strip().lower(): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def normalize_event_log_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize event log columns to standard names.
    
    Handles various naming conventions for user ID, timestamp, and event name.
    Also cleans whitespace and removes invalid rows (empty, null, header-like).
    
    Args:
        events_df: Raw event log DataFrame.
        
    Returns:
        Normalized DataFrame with columns: user_id, event_time, event_name.
        
    Raises:
        ValueError: If required columns cannot be inferred.
    """
    cleaned = events_df.copy()
    cleaned = cleaned.dropna(how="all")  # Remove completely empty rows

    # Find matching columns with flexible naming
    user_col = _find_first_present_column(
        cleaned,
        ["user_id", "id", "journey_id", "customer_id", "account_id"],
    )
    time_col = _find_first_present_column(
        cleaned,
        ["event_time", "event_timestamp", "timestamp", "ts"],
    )
    event_col = _find_first_present_column(
        cleaned,
        ["event_name", "event", "event_type", "name"],
    )

    # Validate that required columns were found
    missing = []
    if user_col is None:
        missing.append("user identifier (e.g. user_id/id/customer_id)")
    if time_col is None:
        missing.append("event time (e.g. event_time/event_timestamp)")
    if event_col is None:
        missing.append("event name (e.g. event_name)")
    if missing:
        raise ValueError(f"Could not infer required columns from event log: {missing}")

    # Extract and rename columns
    normalized = cleaned[[user_col, time_col, event_col]].copy()
    normalized.columns = ["user_id", "event_time", "event_name"]

    # Handle composite user IDs (customer_id + account_id)
    account_col = _find_first_present_column(cleaned, ["account_id"])
    if user_col.strip().lower() == "customer_id" and account_col is not None:
        account_values = cleaned[account_col].astype(str).str.strip()
        normalized["user_id"] = (
            normalized["user_id"].astype(str).str.strip() + "_" + account_values
        )

    # Clean whitespace
    normalized["user_id"] = normalized["user_id"].astype(str).str.strip()
    normalized["event_name"] = normalized["event_name"].astype(str).str.strip()
    normalized["event_time"] = normalized["event_time"].astype(str).str.strip()

    # Remove empty and invalid rows
    normalized = normalized[
        (normalized["user_id"] != "")
        & (normalized["event_name"] != "")
        & (normalized["event_time"] != "")
        & ~normalized["user_id"].str.lower().isin({"nan", "none", "null"})
        & ~normalized["event_name"].str.lower().isin({"nan", "none", "null"})
        & ~normalized["event_time"].str.lower().isin({"nan", "none", "null"})
    ]
    
    # Remove header-like rows in large exports
    normalized = normalized[
        ~normalized["event_name"].str.lower().isin({"event_name", "sep"})
        & ~normalized["event_time"].str.lower().isin({"event_time", "event_timestamp"})
    ]

    return normalized


# ==================== Journey Labeling and Building ====================


def _label_journey(
    event_names: List[str],
    last_event_time: pd.Timestamp,
    latest_timestamp: pd.Timestamp,
    success_event: str,
    inactivity_days_for_unsuccessful: int,
) -> str:
    """Determine journey state: successful, unsuccessful, or ongoing.
    
    Args:
        event_names: List of event names in the journey.
        last_event_time: Timestamp of the last event.
        latest_timestamp: Latest timestamp in the entire dataset.
        success_event: Name of the success event (e.g., 'order_shipped').
        inactivity_days_for_unsuccessful: Days of inactivity threshold.
        
    Returns:
        Journey state: "successful", "unsuccessful", or "ongoing".
    """
    # Journey is successful if it contains the success event
    if success_event in event_names:
        return "successful"

    # Journey is unsuccessful if inactive for threshold days
    inactive_days = (latest_timestamp - last_event_time).days
    if inactive_days >= inactivity_days_for_unsuccessful:
        return "unsuccessful"
    
    # Otherwise, journey is still ongoing
    return "ongoing"


def truncate_journey_by_time(
    record: JourneyRecord,
    rng: np.random.Generator,
    min_keep_events: int,
) -> JourneyRecord:
    """Create a truncated journey record by randomly cutting at a time point.
    
    Used for data augmentation in training. Preserves at least min_keep_events
    to ensure a minimum journey length.
    
    Args:
        record: Original journey record.
        rng: Random number generator for reproducibility.
        min_keep_events: Minimum number of events to keep.
        
    Returns:
        New JourneyRecord with truncated event sequence.
    """
    if len(record.event_ids) <= 1:
        return record

    start_ns = record.event_times[0].value
    end_ns = record.event_times[-1].value
    min_keep_events = max(1, int(min_keep_events))
    
    # If journey is too short, return as-is
    if len(record.event_ids) <= min_keep_events:
        return record

    # Earliest cutoff: respects minimum event requirement
    earliest_cutoff_ns = record.event_times[min_keep_events - 1].value

    if earliest_cutoff_ns >= end_ns:
        return record

    # Randomly sample a cutoff point within journey duration
    if start_ns >= end_ns:
        keep_len = 1
    else:
        duration_ns = end_ns - start_ns
        frac = float(rng.random())  # Random fraction in [0, 1)
        cutoff_ns = start_ns + int(frac * duration_ns)
        
        # Ensure cutoff respects minimum event requirement
        if cutoff_ns < earliest_cutoff_ns:
            cutoff_ns = earliest_cutoff_ns
        
        # Count events up to cutoff time
        keep_len = sum(event_time.value <= cutoff_ns for event_time in record.event_times)

    # Ensure valid keep_len (at least 1, at most original length - 1)
    keep_len = max(1, min(keep_len, len(record.event_ids) - 1 if len(record.event_ids) > 1 else 1))
    
    return JourneyRecord(
        user_id=record.user_id,
        event_ids=record.event_ids[:keep_len],
        event_times=record.event_times[:keep_len],
        label=record.label,
        journey_end_time=record.event_times[keep_len - 1],
    )


def build_journey_records(
    events_df: pd.DataFrame,
    vocab: Dict[str, int],
    success_event: str = SUCCESS_EVENT,
    inactivity_days_for_unsuccessful: int = 60,
    exclude_ongoing_from_training: bool = True,
) -> Tuple[List[JourneyRecord], List[OngoingJourneyRecord]]:
    """Build journey records from raw event log.
    
    Separates journeys into finished (labeled) and ongoing (unlabeled).
    Finished journeys are classified as successful or unsuccessful.
    Ongoing journeys are excluded from training but used for inference.
    
    Args:
        events_df: Raw event log DataFrame.
        vocab: Vocabulary mapping event names to IDs.
        success_event: Name of the success event.
        inactivity_days_for_unsuccessful: Days threshold for unsuccessful label.
        exclude_ongoing_from_training: Whether to exclude ongoing journeys.
        
    Returns:
        Tuple of (finished journeys, ongoing journeys).
        
    Raises:
        ValueError: If no finished journeys are available for training.
    """
    _validate_required_columns(events_df)

    # Prepare data: convert times and sort
    df = events_df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    if df["event_time"].isna().any():
        raise ValueError("Some event_time values are invalid. Use ISO datetime strings.")

    df = df.sort_values(["user_id", "event_time"])
    latest_timestamp = df["event_time"].max()
    grouped = df.groupby("user_id", sort=False)

    records: List[JourneyRecord] = []
    ongoing_records: List[OngoingJourneyRecord] = []
    unk_id = vocab[UNK_TOKEN]

    # Process each user's journey
    for user_id, journey in grouped:
        journey = journey.sort_values("event_time")
        event_names = journey["event_name"].astype(str).tolist()
        ids = [vocab.get(name, unk_id) for name in event_names]  # Map events to vocab IDs
        last_event_time = journey["event_time"].max()
        
        # Classify journey state
        journey_state = _label_journey(
            event_names=event_names,
            last_event_time=last_event_time,
            latest_timestamp=latest_timestamp,
            success_event=success_event,
            inactivity_days_for_unsuccessful=inactivity_days_for_unsuccessful,
        )

        # Handle ongoing journeys
        if journey_state == "ongoing":
            ongoing_records.append(
                OngoingJourneyRecord(
                    user_id=user_id,
                    event_ids=ids,
                    event_times=journey["event_time"].tolist(),
                    journey_end_time=last_event_time,
                )
            )
            if exclude_ongoing_from_training:
                continue
            continue

        # Handle finished journeys (successful or unsuccessful)
        binary_label = LABEL_SUCCESSFUL if journey_state == "successful" else LABEL_UNSUCCESSFUL
        records.append(
            JourneyRecord(
                user_id=user_id,
                event_ids=ids,
                event_times=journey["event_time"].tolist(),
                label=binary_label,
                journey_end_time=last_event_time,
            )
        )

    if not records:
        raise ValueError(
            "No finished journeys available for training after filtering. "
            "Check inactivity threshold and event log time coverage."
        )

    return records, ongoing_records


def prepare_from_event_log(
    event_log_path: str,
    event_definitions_path: str,
    max_len: int,
    success_event: str = SUCCESS_EVENT,
    inactivity_days_for_unsuccessful: int = 60,
    exclude_ongoing_from_training: bool = True,
    cache_enabled: bool = True,
    cache_path: str = "data/cache/prepared_data.pkl",
) -> PreparedData:
    """Prepare training and inference data from raw event logs.
    
    This is the main entry point for data preparation. It:
    1. Checks for cached data (for efficiency on large datasets)
    2. Normalizes event log columns
    3. Builds vocabulary from event definitions
    4. Creates journey records with labels
    5. Caches results for future runs
    
    Args:
        event_log_path: Path to event log CSV.
        event_definitions_path: Path to event definitions CSV.
        max_len: Retained for compatibility/config metadata. The LSTM pipeline uses full journeys
            (variable length via packing) and does not truncate to max_len. The RF baseline uses
            max_len for fixed-width flattening/padding.
        success_event: Name of the success event (default: 'order_shipped').
        inactivity_days_for_unsuccessful: Days threshold for unsuccessful (default: 60).
        exclude_ongoing_from_training: Whether to exclude ongoing journeys from training.
        cache_enabled: Whether to use caching.
        cache_path: Path to cache file.
        
    Returns:
        PreparedData object containing records, ongoing records, vocab, and max_len.
        
    Raises:
        ValueError: If event log is malformed or no finished journeys exist.
    """
    event_log = Path(event_log_path)
    event_defs_file = Path(event_definitions_path)
    cache_file = Path(cache_path)

    # Build cache metadata to detect when cache is stale

    cache_metadata = {
        "event_log_path": str(event_log.resolve()),
        "event_log_mtime_ns": event_log.stat().st_mtime_ns if event_log.exists() else -1,
        "event_definitions_path": str(event_defs_file.resolve()),
        "event_definitions_mtime_ns": event_defs_file.stat().st_mtime_ns if event_defs_file.exists() else -1,
        "max_len": int(max_len),
        "success_event": str(success_event),
        "inactivity_days_for_unsuccessful": int(inactivity_days_for_unsuccessful),
        "exclude_ongoing_from_training": bool(exclude_ongoing_from_training),
    }

    # Try to load from cache if enabled and valid

    if cache_enabled and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("metadata") == cache_metadata
                and isinstance(cached.get("prepared_data"), PreparedData)
            ):
                LOGGER.info("Loaded prepared data cache from %s", cache_file)
                return cached["prepared_data"]
        except Exception:
            # Cache corruption or schema mismatch should not block training.
            pass

    event_defs = load_event_definitions(event_definitions_path)
    vocab = build_vocab(event_defs["event_name"].astype(str).tolist())

    events_df = pd.read_csv(event_log_path, low_memory=False, on_bad_lines="skip")
    events_df = normalize_event_log_columns(events_df)
    records, ongoing_records = build_journey_records(
        events_df=events_df,
        vocab=vocab,
        success_event=success_event,
        inactivity_days_for_unsuccessful=inactivity_days_for_unsuccessful,
        exclude_ongoing_from_training=exclude_ongoing_from_training,

        # Create prepared data object
    )

    prepared = PreparedData(records=records, ongoing_records=ongoing_records, vocab=vocab, max_len=max_len)

    # Cache results if enabled

    if cache_enabled:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump({"metadata": cache_metadata, "prepared_data": prepared}, f)
        LOGGER.info("Saved prepared data cache to %s", cache_file)

    return prepared