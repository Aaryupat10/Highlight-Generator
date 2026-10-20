"""Project-trained sequence model for ordered match commentary windows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from text_classifier import TfidfEncoder


class MatchHalfAttentionNet(torch.nn.Module):
    """BiGRU plus temporal self-attention and per-window multi-task heads."""

    def __init__(self, dimensions: int, event_classes: int, hidden: int = 48, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.input_projection = torch.nn.Linear(dimensions + 1, hidden)
        self.recurrent = torch.nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.attention = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.eventness_head = torch.nn.Linear(hidden, 1)
        self.event_type_head = torch.nn.Linear(hidden, event_classes)

    def forward(self, features: torch.Tensor, time_gaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([features, time_gaps.unsqueeze(-1)], dim=-1)
        x = torch.relu(self.input_projection(x))
        x, _ = self.recurrent(x)
        attended, weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm(x + self.dropout(attended))
        return self.eventness_head(x).squeeze(-1), self.event_type_head(x), weights


class MatchHalfSequenceModel:
    def __init__(self, checkpoint: dict[str, Any]):
        self.encoder = TfidfEncoder(checkpoint["vocabulary"], checkpoint["idf"])
        self.event_classes = list(checkpoint["event_classes"])
        self.network = MatchHalfAttentionNet(self.encoder.dimensions, len(self.event_classes), int(checkpoint["hidden_dimensions"]), int(checkpoint["attention_heads"]), float(checkpoint["dropout"]))
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self.supervision = str(checkpoint.get("supervision", "weak event annotations"))

    @classmethod
    def load(cls, path: str | Path) -> "MatchHalfSequenceModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "match_half_attention_mlp":
            raise ValueError("Checkpoint is not a match-half attention model")
        return cls(checkpoint)

    def predict(self, texts: list[str], times: list[float]) -> dict[str, Any]:
        if len(texts) != len(times):
            raise ValueError("Sequence texts and times must have the same length")
        if not texts:
            return {"eventness": [], "event_types": [], "attention": []}
        x = torch.stack([self.encoder.vectorize(text) for text in texts]).unsqueeze(0)
        gaps = [0.0] + [max(0.0, min(120.0, float(times[i]) - float(times[i - 1]))) / 120.0 for i in range(1, len(times))]
        gap_tensor = torch.tensor(gaps, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            binary_logits, type_logits, attention = self.network(x, gap_tensor)
            eventness = torch.sigmoid(binary_logits.squeeze(0)).tolist()
            type_probs = torch.softmax(type_logits.squeeze(0), dim=-1).tolist()
            # Each query window receives attention to all other windows. Average
            # over heads and return its top temporal context positions.
            attention = attention.squeeze(0).mean(dim=0).tolist()
        return {"eventness": eventness, "event_types": [{"label": self.event_classes[max(range(len(row)), key=row.__getitem__)], "score": max(row), "probabilities": dict(zip(self.event_classes, row))} for row in type_probs], "attention": attention}
