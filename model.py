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
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (absolute positions).

    This matches the formulation from "Attention Is All You Need".
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout))

        d_model = int(d_model)
        max_len = int(max_len)
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: Tensor of shape (B, L, D)
        """
        if x.dim() != 3:
            raise ValueError("Expected x to have shape (B, L, D)")
        length = x.size(1)
        if length > self.pe.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds max_len {self.pe.size(0)} for positional encoding."
            )
        x = x + self.pe[:length, :].unsqueeze(0)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier for journey success prediction.

    Inputs are event IDs and per-step time features (no summary statistics).
    Sequences are expected to be padded to a common length with a padding mask.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        time_feature_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        max_len: int = 4096,
    ) -> None:
        super().__init__()

        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.time_projection = nn.Linear(int(time_feature_dim), int(embedding_dim))

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=int(embedding_dim),
            dropout=float(dropout),
            max_len=int(max_len),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(embedding_dim),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(int(embedding_dim))

        self.head = nn.Sequential(
            nn.Linear(int(embedding_dim), 64),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, int(num_classes)),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor (B, L) event IDs.
            time_features: FloatTensor (B, L, F) per-step time features.
            src_key_padding_mask: BoolTensor (B, L) True for PAD positions.
        """
        if x.dim() != 2:
            raise ValueError("x must have shape (B, L)")
        if time_features.dim() != 3:
            raise ValueError("time_features must have shape (B, L, F)")
        if time_features.shape[0] != x.shape[0] or time_features.shape[1] != x.shape[1]:
            raise ValueError("x and time_features must agree on (B, L)")

        token_emb = self.embedding(x)
        time_emb = self.time_projection(time_features)
        h = token_emb + time_emb
        h = self.positional_encoding(h)

        encoded = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        encoded = self.final_norm(encoded)

        if src_key_padding_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            if src_key_padding_mask.dim() != 2:
                raise ValueError("src_key_padding_mask must have shape (B, L)")
            valid = (~src_key_padding_mask).to(encoded.dtype)
            summed = (encoded * valid.unsqueeze(-1)).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            pooled = summed / denom

        logits = self.head(pooled)
        return logits


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
        x_batch: list[torch.Tensor] | torch.Tensor,
        time_features_batch: list[torch.Tensor] | torch.Tensor,
        summary_features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fast path: padded tensors + lengths.
        if isinstance(x_batch, torch.Tensor):
            if not isinstance(time_features_batch, torch.Tensor):
                raise TypeError("time_features_batch must be a Tensor when x_batch is a Tensor")
            if lengths is None:
                raise ValueError("lengths must be provided when using padded batch tensors")
            if x_batch.ndim != 2:
                raise ValueError(f"x_batch must have shape (B, L); got {tuple(x_batch.shape)}")
            if time_features_batch.ndim != 3:
                raise ValueError(
                    f"time_features_batch must have shape (B, L, F); got {tuple(time_features_batch.shape)}"
                )

            event_embeddings = self.embedding(x_batch)
            projected_time_features = self.time_projection(time_features_batch)
            combined = torch.cat([event_embeddings, projected_time_features], dim=-1)

            packed = pack_padded_sequence(
                combined,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        else:
            # Backward-compatible path: list of variable-length tensors.
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