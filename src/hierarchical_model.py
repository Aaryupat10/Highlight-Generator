"""Project-owned two-tier sequence model for commentary windows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from text_classifier import TfidfEncoder


class HierarchicalCommentaryNet(torch.nn.Module):
    """Local window sequence → match-half context → separate task heads."""

    def __init__(self, dimensions: int, event_classes: int, hidden: int = 32, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.projection = torch.nn.Linear(dimensions + 1, hidden)
        self.local_gru = torch.nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.local_attention = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.local_norm = torch.nn.LayerNorm(hidden)
        self.global_gru = torch.nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.global_attention = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.global_norm = torch.nn.LayerNorm(hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.eventness_head = torch.nn.Linear(hidden * 2, 1)
        self.highlight_head = torch.nn.Linear(hidden * 2, 1)
        self.event_type_head = torch.nn.Linear(hidden * 2, event_classes)

    def forward(self, chunks: list[torch.Tensor], gaps: list[torch.Tensor]) -> dict[str, Any]:
        if not chunks or len(chunks) != len(gaps):
            raise ValueError("Each sequence needs at least one temporal chunk and matching gaps")
        local_states = []
        local_weights = []
        for features, time_gaps in zip(chunks, gaps):
            x = torch.cat([features, time_gaps.unsqueeze(-1)], dim=-1).unsqueeze(0)
            local, _ = self.local_gru(torch.relu(self.projection(x)))
            attended, weights = self.local_attention(local, local, local, need_weights=True, average_attn_weights=True)
            local = self.local_norm(local + self.dropout(attended))
            local_states.append(local.squeeze(0))
            local_weights.append(weights.squeeze(0))
        chunk_vectors = torch.stack([state.mean(dim=0) for state in local_states]).unsqueeze(0)
        global_state, _ = self.global_gru(chunk_vectors)
        global_context, global_weights = self.global_attention(global_state, global_state, global_state, need_weights=True, average_attn_weights=True)
        global_state = self.global_norm(global_state + self.dropout(global_context)).squeeze(0)
        outputs = []
        for chunk_index, local in enumerate(local_states):
            context = global_state[chunk_index].unsqueeze(0).expand(local.size(0), -1)
            joined = torch.cat([local, context], dim=-1)
            outputs.append({
                "eventness": self.eventness_head(joined).squeeze(-1),
                "highlight": self.highlight_head(joined).squeeze(-1),
                "event_types": self.event_type_head(joined),
                "local_attention": local_weights[chunk_index],
            })
        return {"chunks": outputs, "global_attention": global_weights.squeeze(0)}


class HierarchicalMatchModel:
    def __init__(self, checkpoint: dict[str, Any]):
        self.encoder = TfidfEncoder(checkpoint["vocabulary"], checkpoint["idf"])
        self.event_classes = list(checkpoint["event_classes"])
        self.chunk_seconds = float(checkpoint.get("chunk_seconds", 300))
        self.network = HierarchicalCommentaryNet(
            self.encoder.dimensions,
            len(self.event_classes),
            int(checkpoint.get("hidden_dimensions", 32)),
            int(checkpoint.get("attention_heads", 4)),
            float(checkpoint.get("dropout", 0.25)),
        )
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self.supervision = str(checkpoint.get("supervision", "separate weak event and human-highlight judgments"))

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalMatchModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "hierarchical_commentary_multitask_v1":
            raise ValueError("Checkpoint is not a two-tier hierarchical commentary model")
        return cls(checkpoint)

    def predict(self, texts: list[str], times: list[float]) -> dict[str, Any]:
        if len(texts) != len(times):
            raise ValueError("Sequence texts and times must have the same length")
        if not texts:
            return {"eventness": [], "highlight": [], "event_types": [], "attention": []}
        chunk_keys = sorted({int(max(0.0, float(t)) // self.chunk_seconds) for t in times})
        key_to_chunk = {key: [] for key in chunk_keys}
        for i, seconds in enumerate(times):
            key_to_chunk[int(max(0.0, float(seconds)) // self.chunk_seconds)].append(i)
        chunks: list[torch.Tensor] = []
        gaps: list[torch.Tensor] = []
        for key in chunk_keys:
            indices = key_to_chunk[key]
            chunks.append(torch.stack([self.encoder.vectorize(texts[i]) for i in indices]))
            chunk_times = [float(times[i]) for i in indices]
            chunk_gaps = [0.0] + [max(0.0, min(120.0, chunk_times[j] - chunk_times[j - 1])) / 120.0 for j in range(1, len(chunk_times))]
            gaps.append(torch.tensor(chunk_gaps, dtype=torch.float32))
        with torch.inference_mode():
            result = self.network(chunks, gaps)
        count = len(texts)
        eventness = [0.0] * count
        highlight = [0.0] * count
        event_types: list[dict[str, Any] | None] = [None] * count
        attention = [[0.0] * count for _ in range(count)]
        global_weights = result["global_attention"].tolist()
        for source_chunk, key in enumerate(chunk_keys):
            source_indices = key_to_chunk[key]
            for local_index, source_i in enumerate(source_indices):
                row = result["chunks"][source_chunk]
                eventness[source_i] = float(torch.sigmoid(row["eventness"][local_index]).item())
                highlight[source_i] = float(torch.sigmoid(row["highlight"][local_index]).item())
                probabilities = torch.softmax(row["event_types"][local_index], dim=-1).tolist()
                family = max(range(len(probabilities)), key=probabilities.__getitem__)
                event_types[source_i] = {"label": self.event_classes[family], "score": probabilities[family], "probabilities": dict(zip(self.event_classes, probabilities))}
                local_row = row["local_attention"][local_index].tolist()
                for target_chunk, target_key in enumerate(chunk_keys):
                    target_indices = key_to_chunk[target_key]
                    mass = float(global_weights[source_chunk][target_chunk])
                    if target_chunk == source_chunk:
                        local_mass = sum(local_row)
                        if local_mass <= 0:
                            for target_i in target_indices:
                                attention[source_i][target_i] += mass / len(target_indices)
                        else:
                            for position, target_i in enumerate(target_indices):
                                attention[source_i][target_i] += mass * local_row[position] / local_mass
                    else:
                        for target_i in target_indices:
                            attention[source_i][target_i] += mass / len(target_indices)
        return {"eventness": eventness, "highlight": highlight, "event_types": event_types, "attention": attention}
