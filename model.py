from __future__ import annotations

"""
LSTM-based neural network for customer journey success prediction.

Architecture:
- Event embedding layer
- Time feature projection layer
- Bidirectional LSTM (optional)
- Classification head with dropout
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for journey success prediction.

    Combines event embeddings with temporal features, processes through an LSTM,
    and uses a small MLP head to predict binary outcome (successful/unsuccessful).
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        time_feature_dim: int,
        time_embedding_dim: int,
        hidden_size: int,
        summary_feature_dim: int = 5,
        summary_hidden_dim: int = 32,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Embedding for categorical event IDs.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Project time feature vectors to a learnable embedding space.
        self.time_projection = nn.Linear(time_feature_dim, time_embedding_dim)

        # LSTM consumes concatenated (event_embedding || time_embedding).
        lstm_input_size = embedding_dim + time_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Summary-feature encoder (non-recurrent path).
        self.summary_encoder = nn.Sequential(
            nn.Linear(int(summary_feature_dim), int(summary_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion head: concat(LSTM_state, summary_state) -> logits.
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        fusion_input_size = int(lstm_output_size) + int(summary_hidden_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x_batch: list[torch.Tensor],
        time_features_batch: list[torch.Tensor],
        summary_features: torch.Tensor,
    ) -> torch.Tensor:
        # Combine event embeddings and projected time features per sequence.
        combined_sequences = []
        for x, time_features in zip(x_batch, time_features_batch):
            event_embeddings = self.embedding(x)
            projected_time_features = self.time_projection(time_features)
            combined_sequences.append(torch.cat([event_embeddings, projected_time_features], dim=-1))

        # Pack sequences for LSTM (handles variable lengths efficiently).
        packed = pack_sequence(combined_sequences, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        # Extract final hidden state.
        if self.lstm.bidirectional:
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_state = h_n[-1]

        summary_state = self.summary_encoder(summary_features)
        fused = torch.cat([final_state, summary_state], dim=1)
        logits = self.fusion_head(fused)
        return logits