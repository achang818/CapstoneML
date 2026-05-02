# Journey Success LSTM Starter

This starter project trains an LSTM classifier to predict finished outcomes: `successful` vs `unsuccessful`.

Journey states in raw data are defined as:
- `successful`: journey contains `order_shipped`
- `unsuccessful`: no `order_shipped` and at least 60 days have passed since that journey's last event (measured against the latest event timestamp in the dataset)
- `ongoing`: no `order_shipped` and fewer than 60 days since the last event

Training behavior:
- `ongoing` journeys are excluded from supervised training
- finished journeys (`successful` and `unsuccessful`) are used as labeled data
- training is split by journey end time, not randomly
- preprocessing creates truncated training prefixes at random timestamps to imitate ongoing, partially observed journeys
- the number of truncated copies per journey is scaled by journey duration in days, not by event count
- the minimum retained truncation history is now day-based, not event-count based
- each event carries time features for gap since previous action and elapsed time since journey start
- the encoder uses a small CNN front-end before a bidirectional LSTM to capture local event patterns and longer journey context
- after training, the pipeline writes predictions for currently ongoing journeys to a CSV file
- if you stop training manually, the script still exports predictions for ongoing journeys using the current model state

## Files

- `event_definitions.csv`: event metadata you shared
- `data/sample_events_template.csv`: sample input format
- `preprocessing.py`: sequence building and label creation
- `data_loader.py`: `Dataset` and train/validation `DataLoader` creation
- `model.py`: LSTM classifier architecture
- `trainer.py`: training loop and evaluation metrics
- `train.py`: LSTM training entrypoint that orchestrates the full pipeline
- `train_rf.py`: Random Forest training entrypoint with cross-validation
- `configs/train_config.yaml`: all model, training, split, and preprocessing hyperparameters
- `preprocess.py`: legacy preprocessing script
- `train_lstm.py`: legacy single-file training script
- `requirements.txt`: Python dependencies

## Expected Event Log Format

Your event log CSV should look like:

- `user_id`: journey/user identifier
- `event_time`: timestamp (ISO format recommended)
- `event_name`: event string (from `event_definitions.csv`)

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train with real data:

```bash
python train.py --config configs/train_config.yaml
```

3. If the configured event log file is missing, the script auto-generates synthetic data so you can test the full pipeline.

## YAML Hyperparameters

All hyperparameters now live in `configs/train_config.yaml`, including:

- sequence length (`data.max_len`)
- train/validation split (`data.val_split`, `data.split_random_state`)
- time feature scaling (`data.time_feature_max_gap_hours`)
- truncation sampling (`data.truncation_probability`, `data.min_truncation_days`)
- inactivity cutoff (`data.inactivity_days_for_unsuccessful`)
- model size (`model.embedding_dim`, `model.time_feature_dim`, `model.time_embedding_dim`, `model.cnn_channels`, `model.cnn_kernel_size`, `model.hidden_size`, `model.lstm_layers`, `model.bidirectional`, `model.dropout`)
- RF model settings (`rf_model.n_estimators`, `rf_model.max_depth`, `rf_model.min_samples_split`, `rf_model.min_samples_leaf`)
- number of classes (`model.num_classes`)
- optimization (`training.epochs`, `training.batch_size`, `training.lr`, `training.grad_clip_norm`)
- output path, threshold, and log file (`outputs.ongoing_predictions_path`, `outputs.decision_threshold`, `outputs.log_file_path`)
- RF artifacts (`outputs.rf_model_path`, `outputs.rf_training_data_path`, `outputs.rf_pdp_plot_path`, `outputs.rf_ice_plot_path`)
- synthetic data generation controls (`synthetic_data.*`)

Edit the YAML, then run:

```bash
python train.py --config configs/train_config.yaml
```

Both `train.py` and `train_rf.py` write console output and log files through the shared logging setup.

`train_rf.py` additionally saves:
- trained model artifact (`.joblib`)
- training matrix/labels used for fitting (`.npz`)
- PDP and ICE plots for the single most important RF feature (`.png`)

## Next Improvements

- Add train/test split by time (not random) to avoid leakage.
- Add class weighting for imbalanced outcomes.
- Save model and vocab for inference service.
- Add feature engineering (time gaps, stage transitions, milestone embeddings).
