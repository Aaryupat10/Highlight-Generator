#!/usr/bin/env python3
"""Train a project-owned TF-IDF + PyTorch MLP and compare with a linear baseline."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from text_classifier import TfidfEncoder


def make_mlp(dimensions: int, hidden: int = 32, dropout: float = 0.4) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dimensions, hidden), torch.nn.ReLU(),
        torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1),
    )


def fit_linear(x: torch.Tensor, y: torch.Tensor, epochs: int = 220) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    weights = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05, weight_decay=0.03)
    positive = max(1, int(y.sum().item()))
    negative = max(1, len(y) - positive)
    pos_weight = torch.tensor([min(negative / positive, 5.0)])
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights + bias, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    return weights.detach(), bias.detach()


def fit_mlp(x: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> torch.nn.Module:
    torch.manual_seed(17)
    model = make_mlp(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    positive = max(1, int(y.sum().item()))
    negative = max(1, len(y) - positive)
    pos_weight = torch.tensor([min(negative / positive, 5.0)])
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def scores(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
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
    parser.add_argument("--labels", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/commentary_tfidf_mlp.pt"))
    parser.add_argument("--exclude-match", help="Train the saved checkpoint without this entire match (for an honest match-held-out review).")
    args = parser.parse_args()
    if not args.labels.is_file():
        parser.error(f"Labels file not found: {args.labels}")

    by_match: dict[str, list[tuple[str, int]]] = defaultdict(list)
    source_names: set[str] = set()
    skipped = 0
    with args.labels.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"match_id", "english_commentary_context", "human_label"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("CSV must contain match_id, english_commentary_context, and human_label columns.")
        for row in reader:
            label = (row.get("human_label") or "").strip().casefold()
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            source_names.add((row.get("source") or "human annotation").strip())
            if not text or not match or label in {"", "unclear"}:
                skipped += 1
                continue
            if label not in {"highlight", "routine"}:
                parser.error(f"Unexpected label {label!r} in match {match!r}")
            by_match[match].append((text, int(label == "highlight")))

    all_rows = [row for rows in by_match.values() for row in rows]
    counts = Counter(label for _, label in all_rows)
    if len(by_match) < 2 or not counts[0] or not counts[1]:
        parser.error("Need both labels and at least two matches for match-held-out evaluation.")
    if args.exclude_match and args.exclude_match not in by_match:
        parser.error(f"Match to exclude is not in the labels file: {args.exclude_match}")
    training_matches = sorted(match for match in by_match if match != args.exclude_match)
    training_rows = [row for match in training_matches for row in by_match[match]]
    training_counts = Counter(label for _, label in training_rows)
    if not training_counts[0] or not training_counts[1]:
        parser.error("The final training subset must contain both positive and routine examples.")
    supervision = "mixed SoccerNet event timestamps and human-reviewed highlights" if len(source_names) > 1 else next(iter(source_names))
    print(f"Examples: {len(all_rows)} ({counts[1]} positive, {counts[0]} routine, {skipped} skipped); matches: {len(by_match)}")
    print(f"Supervision: {supervision}")

    mlp_results: dict[str, dict[str, float | int]] = {}
    linear_results: dict[str, dict[str, float | int]] = {}
    for held_out, eval_rows in sorted(by_match.items()):
        train_rows = [row for match, rows in by_match.items() if match != held_out for row in rows]
        if len({label for _, label in eval_rows}) < 2 or not train_rows:
            continue
        encoder = TfidfEncoder.fit([text for text, _ in train_rows])
        x_train = torch.stack([encoder.vectorize(text) for text, _ in train_rows])
        y_train = torch.tensor([label for _, label in train_rows], dtype=torch.float32)
        x_eval = torch.stack([encoder.vectorize(text) for text, _ in eval_rows])
        y_eval = torch.tensor([label for _, label in eval_rows], dtype=torch.int64)
        weights, bias = fit_linear(x_train, y_train)
        linear_results[held_out] = scores(torch.sigmoid(x_eval @ weights + bias), y_eval)
        model = fit_mlp(x_train, y_train)
        with torch.inference_mode():
            mlp_results[held_out] = scores(torch.sigmoid(model(x_eval).squeeze(1)), y_eval)
        print(f"{held_out}: MLP {json.dumps(mlp_results[held_out])}; linear {json.dumps(linear_results[held_out])}")

    # Fit a deployable model on all current labels after the held-out comparison.
    texts = [text for text, _ in training_rows]
    encoder = TfidfEncoder.fit(texts)
    x_all = torch.stack([encoder.vectorize(text) for text in texts])
    y_all = torch.tensor([label for _, label in training_rows], dtype=torch.float32)
    model = fit_mlp(x_all, y_all)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "mlp", "feature_type": "tfidf", "state_dict": model.state_dict(),
        "dimensions": encoder.dimensions, "hidden_dimensions": [32], "dropout": 0.4,
        "vocabulary": encoder.vocabulary, "idf": encoder.idf,
        "training_matches": training_matches, "excluded_match": args.exclude_match or "",
        "training_examples": len(training_rows), "supervision": supervision,
        "features": "project-fitted unigram/bigram TF-IDF plus normalized punctuation, negation, hypothetical, retrospective, live-action, and excitement cues",
    }, args.out)

    summary = {
        "model": str(args.out), "model_type": "project-trained TF-IDF + PyTorch MLP",
        "training_matches": training_matches, "final_model_excluded_match": args.exclude_match or "",
        "training_examples": len(training_rows), "positive_examples": training_counts[1],
        "routine_examples": training_counts[0], "all_labeled_examples": len(all_rows), "skipped": skipped,
        "supervision": supervision, "threshold": 0.5,
        "leave_one_match_out_mlp": mlp_results,
        "leave_one_match_out_linear_baseline": linear_results,
        "warning": "Exploratory leave-one-match-out results. SoccerNet event timestamp labels indicate event-related windows, not necessarily editorial highlight quality; human-reviewed highlights are subjective. Each held-out fold fits its vocabulary and IDF on training matches only.",
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {args.out}\nSaved evaluation: {metrics_path}")


if __name__ == "__main__":
    main()
