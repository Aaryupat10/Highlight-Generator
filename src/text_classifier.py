"""Small deterministic bag-of-words model for commentary highlight ranking."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

FEATURES = 8192
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def text_features(text: str) -> list[str]:
    """Return lexical, capitalization, punctuation, and discourse features."""
    raw_tokens = TOKEN_RE.findall(text)
    tokens = [token.casefold() for token in raw_tokens]
    features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    words = [token for token in raw_tokens if any(ch.isalpha() for ch in token)]
    caps_words = [word for word in words if len(word) > 1 and word.isupper()]
    for word in caps_words:
        features.append(f"__CAPS_WORD__{word}")
    letter_count = sum(ch.isalpha() for ch in text)
    caps_ratio = sum(ch.isupper() for ch in text if ch.isalpha()) / max(1, letter_count)
    if caps_ratio >= 0.5:
        features.append("__CAPS_RATIO_HIGH__")
    elif caps_ratio >= 0.2:
        features.append("__CAPS_RATIO_MID__")
    exclamation_count = text.count("!")
    if exclamation_count:
        features.append("__EXCLAMATION__")
        features.append("__EXCLAMATION_MULTIPLE__" if exclamation_count > 1 else "__EXCLAMATION_SINGLE__")
    if caps_words:
        features.append("__HAS_CAPS_WORDS__")
    # Keep negation as lexical features and add a cue for the classifier.
    if any(token in {"no", "not", "never", "cannot", "can't", "couldn't", "didn't", "without"} for token in tokens):
        features.append("__NEGATION__")
    return features


class TfidfEncoder:
    """Small project-owned TF-IDF encoder for the course-based model option."""

    def __init__(self, vocabulary: dict[str, int], idf: list[float]):
        self.vocabulary = vocabulary
        self.idf = idf
        self.dimensions = len(vocabulary)

    @classmethod
    def fit(cls, texts: list[str], max_features: int = FEATURES, min_df: int = 1) -> "TfidfEncoder":
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(text_features(text)))
        candidates = [(feature, count) for feature, count in document_frequency.items() if count >= min_df]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected = candidates[:max_features]
        vocabulary = {feature: index for index, (feature, _) in enumerate(selected)}
        n_documents = max(1, len(texts))
        idf = [math.log((1 + n_documents) / (1 + count)) + 1.0 for _, count in selected]
        return cls(vocabulary, idf)

    def vectorize(self, text: str) -> torch.Tensor:
        counts = Counter(text_features(text))
        vec = torch.zeros(self.dimensions, dtype=torch.float32)
        for feature, count in counts.items():
            index = self.vocabulary.get(feature)
            if index is not None:
                tf = 1.0 + math.log(count)
                vec[index] = tf * self.idf[index]
        norm = torch.linalg.vector_norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class MultiTaskMLP(torch.nn.Module):
    """Shared text encoder with eventness and weak event-family output heads."""

    def __init__(self, dimensions: int, hidden: int, event_classes: int, dropout: float = 0.4):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(dimensions, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.eventness_head = torch.nn.Linear(hidden, 1)
        self.event_type_head = torch.nn.Linear(hidden, event_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        return self.eventness_head(hidden).squeeze(-1), self.event_type_head(hidden)


class PipelineMultiTaskMLP(torch.nn.Module):
    """Shared project-trained text encoder with separate evidence targets.

    Event relatedness and broad event family use SoccerNet's timestamp labels;
    subjective highlight value uses only human-reviewed highlight/routine rows.
    Keeping the heads separate prevents the two kinds of supervision from being
    treated as interchangeable labels.
    """

    def __init__(self, dimensions: int, hidden: int, event_classes: int, dropout: float = 0.35):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(dimensions, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.eventness_head = torch.nn.Linear(hidden, 1)
        self.highlight_head = torch.nn.Linear(hidden, 1)
        self.event_type_head = torch.nn.Linear(hidden, event_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        return (
            self.eventness_head(hidden).squeeze(-1),
            self.highlight_head(hidden).squeeze(-1),
            self.event_type_head(hidden),
        )


def vectorize(text: str, dimensions: int = FEATURES) -> torch.Tensor:
    tokens = [t.casefold() for t in TOKEN_RE.findall(text)]
    feats = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    vec = torch.zeros(dimensions, dtype=torch.float32)
    for feature in feats:
        idx = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little") % dimensions
        vec[idx] += 1.0
    norm = torch.linalg.vector_norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class CommentaryClassifier:
    def __init__(self, weights: torch.Tensor | None = None, bias: torch.Tensor | None = None, dimensions: int = FEATURES):
        self.weights = weights.detach().cpu().float().reshape(dimensions) if weights is not None else None
        self.bias = bias.detach().cpu().float().reshape(()) if bias is not None else None
        self.dimensions = dimensions
        self.supervision = "unknown supervision"
        self.model = None
        self.encoder = None

    @classmethod
    def load(cls, path: str | Path) -> "CommentaryClassifier":
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") == "pipeline_multitask_tfidf_mlp_v1":
            encoder = TfidfEncoder(checkpoint["vocabulary"], checkpoint["idf"])
            model = cls(dimensions=encoder.dimensions)
            model.encoder = encoder
            model.model = PipelineMultiTaskMLP(
                model.dimensions,
                int(checkpoint["hidden_dimensions"]),
                len(checkpoint["event_classes"]),
                float(checkpoint.get("dropout", 0.35)),
            )
            model.model.load_state_dict(checkpoint["state_dict"])
            model.model.eval()
            model.event_classes = list(checkpoint["event_classes"])
            model.supervision = str(checkpoint.get("supervision", "separate weak event and human highlight labels"))
            model.highlight_supervision = str(checkpoint.get("highlight_supervision", "human-reviewed highlight/routine windows"))
            model.model_type = str(checkpoint["model_type"])
            return model
        if checkpoint.get("model_type") == "multitask_tfidf_mlp":
            encoder = TfidfEncoder(checkpoint["vocabulary"], checkpoint["idf"])
            model = cls(dimensions=encoder.dimensions)
            model.encoder = encoder
            model.model = MultiTaskMLP(model.dimensions, int(checkpoint["hidden_dimensions"]), len(checkpoint["event_classes"]), float(checkpoint.get("dropout", 0.4)))
            model.model.load_state_dict(checkpoint["state_dict"])
            model.model.eval()
            model.event_classes = list(checkpoint["event_classes"])
            model.supervision = str(checkpoint.get("supervision", "weak SoccerNet event type labels and mixed event/routine labels"))
            return model
        if checkpoint.get("feature_type") == "tfidf":
            encoder = TfidfEncoder(checkpoint["vocabulary"], checkpoint["idf"])
            model = cls(dimensions=encoder.dimensions)
            model.encoder = encoder
            if checkpoint.get("model_type") == "mlp":
                hidden = int(checkpoint.get("hidden_dimensions", [32])[0])
                model.model = torch.nn.Sequential(
                    torch.nn.Linear(model.dimensions, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(checkpoint.get("dropout", 0.4))),
                    torch.nn.Linear(hidden, 1),
                )
                model.model.load_state_dict(checkpoint["state_dict"])
                model.model.eval()
            else:
                model.weights = checkpoint["weights"].detach().cpu().float().reshape(model.dimensions)
                model.bias = checkpoint["bias"].detach().cpu().float().reshape(())
            model.supervision = str(checkpoint.get("supervision", "unknown supervision"))
            return model
        if checkpoint.get("model_type") == "mlp":
            hidden = int(checkpoint.get("hidden_dimensions", [32])[0])
            model = cls(dimensions=int(checkpoint["dimensions"]))
            model.model = torch.nn.Sequential(
                torch.nn.Linear(model.dimensions, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.Linear(hidden, 1),
            )
            model.model.load_state_dict(checkpoint["state_dict"])
            model.model.eval()
        else:
            # Load checkpoints from the earlier linear pilot model.
            model = cls(checkpoint["weights"], checkpoint["bias"], int(checkpoint["dimensions"]))
        model.supervision = str(checkpoint.get("supervision", "unknown supervision"))
        return model

    def probability(self, text: str) -> float:
        x = self.encoder.vectorize(text) if self.encoder is not None else vectorize(text, self.dimensions)
        with torch.inference_mode():
            if isinstance(self.model, PipelineMultiTaskMLP):
                _, highlight_logits, _ = self.model(x)
                return float(torch.sigmoid(highlight_logits.reshape(())).item())
            if isinstance(self.model, MultiTaskMLP):
                eventness_logits, _ = self.model(x)
                return float(torch.sigmoid(eventness_logits.reshape(())).item())
            if self.model is not None:
                return float(torch.sigmoid(self.model(x).reshape(())).item())
            return float(torch.sigmoid(torch.dot(self.weights, x) + self.bias).item())

    def predict_event_type(self, text: str) -> dict[str, Any] | None:
        if not isinstance(self.model, (MultiTaskMLP, PipelineMultiTaskMLP)):
            return None
        x = self.encoder.vectorize(text)
        with torch.inference_mode():
            logits = self.model(x)[-1]
            probs = torch.softmax(logits.reshape(-1), dim=0)
        idx = int(torch.argmax(probs).item())
        return {"label": self.event_classes[idx], "score": float(probs[idx].item()), "probabilities": {label: float(prob) for label, prob in zip(self.event_classes, probs.tolist())}}

    def event_probability(self, text: str) -> float | None:
        """Return the separate weak-label event-relatedness score when present."""
        if not isinstance(self.model, PipelineMultiTaskMLP):
            return None
        x = self.encoder.vectorize(text)
        with torch.inference_mode():
            event_logits, _, _ = self.model(x)
        return float(torch.sigmoid(event_logits.reshape(())).item())

    def explain(self, text: str, top_n: int = 6) -> list[dict[str, Any]]:
        """Return TF-IDF feature ablations that lower the model score most.

        This is a local perturbation explanation, not an attention map or a
        causal explanation. Hashed-feature checkpoints do not preserve words.
        """
        if self.encoder is None:
            return []
        x = self.encoder.vectorize(text)
        active = torch.nonzero(x, as_tuple=False).flatten().tolist()
        feature_by_index = {index: feature for feature, index in self.encoder.vocabulary.items()}
        def probability_for(vector: torch.Tensor) -> float:
            if isinstance(self.model, PipelineMultiTaskMLP):
                _, highlight_logits, _ = self.model(vector)
                return float(torch.sigmoid(highlight_logits.reshape(())).item())
            if isinstance(self.model, MultiTaskMLP):
                eventness_logits, _ = self.model(vector)
                return float(torch.sigmoid(eventness_logits.reshape(())).item())
            if self.model is not None:
                return float(torch.sigmoid(self.model(vector).reshape(())).item())
            return float(torch.sigmoid(torch.dot(self.weights, vector) + self.bias).item())
        with torch.inference_mode():
            base = probability_for(x)
            results = []
            for index in active:
                ablated = x.clone()
                ablated[index] = 0.0
                score = probability_for(ablated)
                delta = base - score
                if delta > 0:
                    feature = feature_by_index.get(index, "")
                    results.append({"feature": feature, "score_drop_when_removed": round(delta, 4)})
        return sorted(results, key=lambda item: item["score_drop_when_removed"], reverse=True)[:top_n]
