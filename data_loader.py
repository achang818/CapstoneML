from __future__ import annotations

"""
Dataset loaders and feature engineering for journey LSTM training.

This module provides:
- Time-based feature engineering (time gaps, elapsed times, calendar features)
- Journey dataset classes for training and inference
- Data augmentation via journey truncation
- DataLoader creation for batch processing
"""

from bisect import bisect_right
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from preprocessing import JourneyRecord, OngoingJourneyRecord, truncate_journey_by_time


def run_length_encode_journey(
    event_ids: Sequence[int],
    event_times: Sequence[pd.Timestamp],
) -> Tuple[List[int], List[pd.Timestamp], np.ndarray]:
    """Collapse consecutive identical events and return per-step repeat counts.

    This is a simple run-length encoding (RLE) over the event stream:
    consecutive duplicates are merged into one step, and a repeat-count feature
    (how many times the event repeated consecutively) is returned.

    The representative timestamp for a run is the first timestamp in that run.

    Returns:
        compressed_event_ids: Event IDs with consecutive duplicates collapsed.
        compressed_event_times: Timestamps aligned to compressed_event_ids.
        repeat_counts_log: Float32 array of shape (len(compressed_event_ids),) with log1p(count).
    """
    if len(event_ids) != len(event_times):
        raise ValueError("event_ids and event_times must have the same length")
    if len(event_ids) == 0:
        return [], [], np.zeros((0,), dtype=np.float32)

    compressed_ids: List[int] = []
    compressed_times: List[pd.Timestamp] = []
    counts: List[int] = []

    current_id = int(event_ids[0])
    current_time = event_times[0]
    current_count = 1

    for event_id, event_time in zip(event_ids[1:], event_times[1:]):
        eid = int(event_id)
        if eid == current_id:
            current_count += 1
            continue

        compressed_ids.append(current_id)
        compressed_times.append(current_time)
        counts.append(current_count)

        current_id = eid
        current_time = event_time
        current_count = 1

    compressed_ids.append(current_id)
    compressed_times.append(current_time)
    counts.append(current_count)

    repeat_counts_log = np.log1p(np.asarray(counts, dtype=np.float32))
    return compressed_ids, compressed_times, repeat_counts_log


def build_time_features(
    event_times: Sequence[pd.Timestamp],
    max_gap_hours: float,
    journey_start_time: pd.Timestamp | None = None,
) -> np.ndarray:
    """Build temporal features: time gap and elapsed time since journey start.

    Normalizes both features by dividing by max_gap_hours.
    Note: values may exceed 1.0 if gaps exceed max_gap_hours.
    
    Args:
        event_times: Sequence of event timestamps.
        max_gap_hours: Maximum time gap in hours for normalization.
        journey_start_time: Override journey start (default: first event).
        
    Returns:
        Array of shape (len(event_times), 2) with normalized gap and elapsed features.
    """
    if not event_times:
        return np.zeros((0, 2), dtype=np.float32)

    # Prevent division by zero.
    clipped_max_gap = max(float(max_gap_hours), 1e-6)
    first_event_time = journey_start_time or event_times[0]
    previous_time = event_times[0]
    rows: List[List[float]] = []

    for current_time in event_times:
        gap_hours = max(0.0, (current_time - previous_time).total_seconds() / 3600.0)
        elapsed_hours = max(0.0, (current_time - first_event_time).total_seconds() / 3600.0)
        rows.append(
            [
                gap_hours / clipped_max_gap,
                elapsed_hours / clipped_max_gap,
            ]
        )
        previous_time = current_time

    return np.asarray(rows, dtype=np.float32)


def build_lstm_time_features(
    event_times: Sequence[pd.Timestamp],
    max_gap_hours: float,
    journey_start_time: pd.Timestamp | None = None,
    repeat_counts_log: np.ndarray | None = None,
) -> np.ndarray:
    """Build enriched time features: temporal + calendar information.
    
    Combines time gaps/elapsed times with cyclical calendar features
    (sin/cos of hour and day of week) to help model learn temporal patterns.
    
    Args:
        event_times: Sequence of event timestamps.
        max_gap_hours: Maximum time gap in hours for normalization.
        journey_start_time: Override journey start (default: first event).
        
    Returns:
        Array of shape (len(event_times), 7 or 8) with all time features.
        If repeat_counts_log is provided, an additional column is appended.
    """
    # Build base time features (gaps and elapsed)
    base_features = build_time_features(
        event_times=event_times,
        max_gap_hours=max_gap_hours,
        journey_start_time=journey_start_time,
    )
    if len(event_times) == 0:
        return base_features

    calendar_rows: List[List[float]] = []
    for timestamp in event_times:
        hour_fraction = (timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0) / 24.0
        day_fraction = float(timestamp.dayofweek) / 7.0
        hour_angle = 2.0 * np.pi * hour_fraction
        day_angle = 2.0 * np.pi * day_fraction
        calendar_rows.append(
            [
                float(np.sin(hour_angle)),
                float(np.cos(hour_angle)),
                float(np.sin(day_angle)),
                float(np.cos(day_angle)),
                float(timestamp.dayofweek >= 5),
            ]
        )

    calendar_features = np.asarray(calendar_rows, dtype=np.float32)
    features = np.concatenate([base_features, calendar_features], axis=1)

    if repeat_counts_log is not None:
        repeat_counts_log = np.asarray(repeat_counts_log, dtype=np.float32).reshape(-1, 1)
        if len(repeat_counts_log) != len(features):
            raise ValueError("repeat_counts_log must have same length as event_times")
        features = np.concatenate([features, repeat_counts_log], axis=1)

    return features


def build_journey_summary_features(
    event_ids: Sequence[int],
    event_times: Sequence[pd.Timestamp],
    max_gap_hours: float,
) -> np.ndarray:
    if len(event_ids) == 0 or len(event_times) == 0:
        return np.zeros(5, dtype=np.float32)

    event_count = float(len(event_ids))
    unique_event_count = float(len(set(int(event_id) for event_id in event_ids)))

    # Calculate inter-event gap statistics

    if len(event_times) > 1:
        gap_hours = np.asarray(
            [
                max(0.0, (current_time - previous_time).total_seconds() / 3600.0)
                for previous_time, current_time in zip(event_times[:-1], event_times[1:])
            ],
            dtype=np.float32,
        )
        mean_gap_hours = float(gap_hours.mean())
        max_gap_hours_observed = float(gap_hours.max())
    else:
        mean_gap_hours = 0.0
        max_gap_hours_observed = 0.0

    gap_norm_divisor = max(float(max_gap_hours), 1e-6)
    mean_gap_norm = mean_gap_hours / gap_norm_divisor
    max_gap_norm = max_gap_hours_observed / gap_norm_divisor

    unique_event_ratio = unique_event_count / max(event_count, 1.0)
    duration_hours = max(0.0, (event_times[-1] - event_times[0]).total_seconds() / 3600.0)
    event_density = event_count / max(duration_hours, 1.0)

    return np.asarray(
        [
            np.log1p(event_count),
            np.log1p(mean_gap_norm),
            np.log1p(max_gap_norm),
            unique_event_ratio,
            np.log1p(event_density),
        ],
        dtype=np.float32,
    )


def build_lstm_sequence_features(
    event_ids: Sequence[int],
    event_times: Sequence[pd.Timestamp],
    max_gap_hours: float,
    journey_start_time: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, int]:
    compressed_ids, compressed_times, repeat_counts_log = run_length_encode_journey(
        event_ids=event_ids,
        event_times=event_times,
    )
    sequence = np.asarray(compressed_ids, dtype=np.int64)
    time_features = build_lstm_time_features(
        compressed_times,
        max_gap_hours=max_gap_hours,
        journey_start_time=journey_start_time,
        repeat_counts_log=repeat_counts_log,
    )
    length = len(sequence)
    return sequence, time_features, length


def split_records_by_time(
    records: Sequence[JourneyRecord],
    val_split: float,
    random_state: int = 42,
) -> Tuple[List[JourneyRecord], List[JourneyRecord]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    if len(records) < 2:
        raise ValueError("Need at least two journeys to perform a random split.")

    rng = np.random.default_rng(random_state)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    split_index = int(np.floor(len(shuffled_records) * (1.0 - val_split)))
    split_index = min(max(split_index, 1), len(shuffled_records) - 1)
    return shuffled_records[:split_index], shuffled_records[split_index:]


def augment_training_records(
    records: Sequence[JourneyRecord],
    truncation_probability: float,
    min_truncation_days: int,
    random_state: int,
) -> List[JourneyRecord]:
    rng = np.random.default_rng(random_state)
    augmented: List[JourneyRecord] = []

    for record in records:
        augmented.append(record)
        if truncation_probability <= 0.0:
            continue

    # Calculate number of truncations based on journey duration

        journey_duration_days = max(
            1,
            int(
                np.ceil(
                    (record.event_times[-1] - record.event_times[0]).total_seconds() / 86400.0
                )
            ),
        )
        num_truncations = max(1, int(np.ceil(truncation_probability * journey_duration_days)))
        for _ in range(num_truncations):
            truncated_record = truncate_journey_by_time(
                record=record,
                rng=rng,
                min_keep_events=min_truncation_days,
            )
            augmented.append(truncated_record)

    return augmented


class JourneyDataset(Dataset):
    """PyTorch Dataset for training on journey records with optional augmentation.

    Supports data augmentation via journey truncation. Each journey can generate multiple
    samples (original + truncated versions). Uses lazy augmentation to avoid memory overhead.
    """
    def __init__(
        self,
        records: Sequence[JourneyRecord],
        max_gap_hours: float,
        truncation_probability: float = 0.0,
        min_truncation_days: int = 1,
        random_state: int = 42,
    ) -> None:
        self.records = list(records)
        self.max_gap_hours = float(max_gap_hours)
        self.truncation_probability = float(truncation_probability)
        self.min_truncation_days = int(min_truncation_days)
        self.random_state = int(random_state)
        self.max_sequence_length = max((len(r.event_ids) for r in self.records), default=1)

        # Build sample counting for augmented dataset indexing.
        self._sample_counts: List[int] = []
        self._cumulative_counts: List[int] = []
        running_total = 0
        for record in self.records:
            num_samples = 1
            if self.truncation_probability > 0.0 and len(record.event_times) > 1:
                journey_duration_days = max(
                    1,
                    int(np.ceil((record.event_times[-1] - record.event_times[0]).total_seconds() / 86400.0)),
                )
                num_samples += max(1, int(np.ceil(self.truncation_probability * journey_duration_days)))
            self._sample_counts.append(num_samples)
            running_total += num_samples
            self._cumulative_counts.append(running_total)

    def __len__(self) -> int:
        return self._cumulative_counts[-1] if self._cumulative_counts else 0

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map flattened sample index to (record_index, sample_offset).
        
        Uses bisect for efficient O(log n) lookup in cumulative sample counts.
        """
        record_idx = bisect_right(self._cumulative_counts, idx)
        previous_total = self._cumulative_counts[record_idx - 1] if record_idx > 0 else 0
        sample_offset = idx - previous_total
        return record_idx, sample_offset

    def _select_record(self, record_idx: int, sample_offset: int) -> JourneyRecord:
        record = self.records[record_idx]
        if sample_offset == 0 or self.truncation_probability <= 0.0 or len(record.event_ids) <= 1:
            return record

        seed = np.uint32((self.random_state * 1_000_003 + record_idx * 97 + sample_offset) % (2**32))
        rng = np.random.default_rng(seed)
        return truncate_journey_by_time(
            record=record,
            rng=rng,
            min_keep_events=self.min_truncation_days,
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        record_idx, sample_offset = self._resolve_index(idx)
        record = self._select_record(record_idx, sample_offset)
        sequence, features, length = build_lstm_sequence_features(
            event_ids=record.event_ids,
            event_times=record.event_times,
            max_gap_hours=self.max_gap_hours,
            journey_start_time=record.event_times[0],
        )
        summary = build_journey_summary_features(
            event_ids=record.event_ids,
            event_times=record.event_times,
            max_gap_hours=self.max_gap_hours,
        )
        return (
            torch.from_numpy(sequence).long(),
            torch.from_numpy(features).float(),
            torch.from_numpy(summary).float(),
            length,
            torch.tensor(record.label, dtype=torch.long),
        )


def collate_journey_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    sequences, time_features, summary_features, lengths, labels = zip(*batch)
    return list(sequences), list(time_features), torch.stack(summary_features), torch.stack(labels)


class OngoingJourneyDataset(Dataset):
    """PyTorch Dataset for inference on unlabeled ongoing journeys.

    Materializes all tensors in memory during initialization for efficient batching
    during inference. Stores metadata (user_ids, timestamps) for prediction exports.
    """
    def __init__(self, records: Sequence[OngoingJourneyRecord], max_gap_hours: float) -> None:
        sequences: List[np.ndarray] = []
        time_features: List[np.ndarray] = []
        summary_features: List[np.ndarray] = []
        lengths: List[int] = []
        user_ids: List[str] = []
        journey_end_times: List[str] = []
        observed_event_counts: List[int] = []

        iterator = tqdm(records, desc="Materializing ongoing tensors", unit="journey", leave=False)
        for record in iterator:
            sequence, features, length = build_lstm_sequence_features(
                event_ids=record.event_ids,
                event_times=record.event_times,
                max_gap_hours=max_gap_hours,
                journey_start_time=record.event_times[0],
            )

            sequences.append(sequence)
            time_features.append(features)
            summary_features.append(
                build_journey_summary_features(
                    event_ids=record.event_ids,
                    event_times=record.event_times,
                    max_gap_hours=max_gap_hours,
                )
            )
            lengths.append(length)
            user_ids.append(str(record.user_id))
            journey_end_times.append(record.journey_end_time.isoformat())
            observed_event_counts.append(len(record.event_ids))

        self.x = [torch.from_numpy(sequence).long() for sequence in sequences]
        self.time_features = [torch.from_numpy(features).float() for features in time_features]
        self.summary_features = [torch.from_numpy(features).float() for features in summary_features]
        self.lengths = lengths
        self.user_ids = user_ids
        self.journey_end_times = journey_end_times
        self.observed_event_counts = observed_event_counts

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str, int]:
        return (
            self.x[idx],
            self.time_features[idx],
            self.summary_features[idx],
            self.lengths[idx],
            self.user_ids[idx],
            self.journey_end_times[idx],
            self.observed_event_counts[idx],
        )


def collate_ongoing_journey_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str, int]]
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
    List[int],
    List[str],
    List[str],
    List[int],
]:
    sequences, time_features, summary_features, lengths, user_ids, journey_end_times, observed_event_counts = zip(*batch)
    return (
        list(sequences),
        list(time_features),
        torch.stack(summary_features),
        list(lengths),
        list(user_ids),
        list(journey_end_times),
        list(observed_event_counts),
    )


def create_data_loaders(
    records: Sequence[JourneyRecord],
    batch_size: int,
    val_split: float,
    max_gap_hours: float,
    truncation_probability: float,
    min_truncation_days: int,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_records, val_records = split_records_by_time(records, val_split=val_split, random_state=random_state)

    train_loader = DataLoader(
        JourneyDataset(
            train_records,
            max_gap_hours=max_gap_hours,
            truncation_probability=truncation_probability,
            min_truncation_days=min_truncation_days,
            random_state=random_state,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_journey_batch,
        num_workers=int(num_workers),
        persistent_workers=bool(int(num_workers) > 0),
    )
    val_loader = DataLoader(
        JourneyDataset(val_records, max_gap_hours=max_gap_hours),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_journey_batch,
        num_workers=int(num_workers),
        persistent_workers=bool(int(num_workers) > 0),
    )
    return train_loader, val_loader


def create_inference_loader(
    records: Sequence[OngoingJourneyRecord],
    batch_size: int,
    max_gap_hours: float,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for inference on ongoing journeys.

    Args:
        records: Ongoing journey records.
        batch_size: Batch size.
        max_gap_hours: Max time gap for feature normalization.

    Returns:
        DataLoader configured for inference without shuffling.
    """
    return DataLoader(
        OngoingJourneyDataset(records, max_gap_hours=max_gap_hours),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ongoing_journey_batch,
        num_workers=int(num_workers),
        persistent_workers=bool(int(num_workers) > 0),
    )