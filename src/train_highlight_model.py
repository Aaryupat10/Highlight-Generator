#!/usr/bin/env python3
"""Train and evaluate a commentary event/highlight classifier from labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from text_classifier import FEATURES, vectorize


def train(x: torch.Tensor, y: torch.Tensor, epochs: int = 220) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    weights = torch.zeros(FEATURES, requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05, weight_decay=0.03)
    positives = max(1, int(y.sum().item()))
    negatives = max(1, len(y) - positives)
    pos_weight = torch.tensor([min(negatives / positives, 5.0)], dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights + bias, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    return weights.detach(), bias.detach()


def make_mlp(hidden: int = 32, dropout: float = 0.4) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURES, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, 1),
    )


def train_mlp(x: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> torch.nn.Module:
    torch.manual_seed(17)
    model = make_mlp()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    positives = max(1, int(y.sum().item()))
    negatives = max(1, len(y) - positives)
    pos_weight = torch.tensor([min(negatives / positives, 5.0)], dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def measure(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    predicted = probabilities >= 0.5
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"n": len(labels), "precision_at_0.5": precision, "recall_at_0.5": recall,
            "f1_at_0.5": 2 * precision * recall / max(1e-12, precision + recall)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/manual_annotation_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/commentary_highlight.pt"))
    args = parser.parse_args()
    if not args.labels.is_file():
        parser.error(f"Manual labels file not found: {args.labels}")

    by_match: dict[str, list[tuple[str, int]]] = defaultdict(list)
    source_types: set[str] = set()
    ignored = 0
    with args.labels.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"match_id", "english_commentary_context", "human_label"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("CSV must contain match_id, english_commentary_context, and human_label columns.")
        for row in reader:
            label = (row.get("human_label") or "").strip().casefold()
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            source_types.add((row.get("source") or "human annotation").strip())
            if label == "unclear" or not label:
                ignored += 1
                continue
            if label not in {"highlight", "routine"} or not text or not match:
                parser.error(f"Invalid annotation row for match {match!r}: label={label!r}")
            by_match[match].append((text, int(label == "highlight")))

    if len(by_match) < 2:
        parser.error("Need labeled examples from at least two matches for match-held-out evaluation.")
    all_rows = [row for rows in by_match.values() for row in rows]
    counts = Counter(label for _, label in all_rows)
    if not counts[0] or not counts[1]:
        parser.error("Need at least one highlight and one routine annotation.")
    has_weak = any("SoccerNet event timestamp" in item for item in source_types)
    has_human = any("Human-reviewed" in item or item == "human annotation" for item in source_types)
    if has_weak and has_human:
        supervision = "mixed SoccerNet event timestamps and human-reviewed highlights"
    elif has_weak:
        supervision = "SoccerNet timestamp-aligned event labels"
    else:
        supervision = "human-reviewed commentary windows"
    print(f"Training examples ({supervision}): {len(all_rows)} ({counts[1]} event/highlight, {counts[0]} routine; {ignored} unclear/blank skipped)")
    for match, rows in sorted(by_match.items()):
        count = Counter(label for _, label in rows)
        print(f"{match}: {len(rows)} ({count[1]} highlight, {count[0]} routine)")

    # Hold out complete matches so no passage from a test game enters training.
    evaluation = {}
    linear_evaluation = {}
    for held_out, eval_rows in sorted(by_match.items()):
        train_rows = [row for match, rows in by_match.items() if match != held_out for row in rows]
        if not train_rows or len({label for _, label in eval_rows}) < 2:
            continue
        x_train = torch.stack([vectorize(text) for text, _ in train_rows])
        y_train = torch.tensor([label for _, label in train_rows], dtype=torch.float32)
        x_eval = torch.stack([vectorize(text) for text, _ in eval_rows])
        y_eval = torch.tensor([label for _, label in eval_rows], dtype=torch.int64)
        weights, bias = train(x_train, y_train)
        linear_result = measure(torch.sigmoid(x_eval @ weights + bias), y_eval)
        linear_evaluation[held_out] = linear_result
        model = train_mlp(x_train, y_train)
        with torch.inference_mode():
            result = measure(torch.sigmoid(model(x_eval).squeeze(1)), y_eval)
        evaluation[held_out] = result
        print(f"Held-out {held_out} MLP: {json.dumps(result)} | linear baseline: {json.dumps(linear_result)}")

    x_all = torch.stack([vectorize(text) for text, _ in all_rows])
    y_all = torch.tensor([label for _, label in all_rows], dtype=torch.float32)
    model = train_mlp(x_all, y_all)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "mlp", "state_dict": model.state_dict(), "dimensions": FEATURES,
                "hidden_dimensions": [32], "training_matches": sorted(by_match),
                "supervision": supervision, "features": "hashed unigram and bigram counts"}, args.out)
    summary = {
        "model": str(args.out), "training_matches": sorted(by_match),
        "supervision": supervision,
        "training_examples": len(all_rows), "positive_examples": counts[1],
        "routine_examples": counts[0], "unclear_or_blank_skipped": ignored,
        "leave_one_match_out": evaluation,
        "linear_baseline_leave_one_match_out": linear_evaluation,
        "model_type": "PyTorch feed-forward neural network (8192 inputs, 32 hidden units, ReLU, dropout, sigmoid output)",
        "warning": "Exploratory match-held-out evaluation. SoccerNet event labels supervise event-related commentary, while human labels reflect subjective highlight review; interpret the held-out match metrics by their label source.",
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved trained commentary model: {args.out}")
    print(f"Saved match-held-out evaluation: {metrics_path}")


if __name__ == "__main__":
    main()
