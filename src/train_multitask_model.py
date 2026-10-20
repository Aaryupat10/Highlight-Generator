#!/usr/bin/env python3
"""Train a team-built shared TF-IDF MLP with eventness and event-family heads."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from text_classifier import MultiTaskMLP, TfidfEncoder

EVENT_CLASSES = ["goal", "shot_or_chance", "card"]


def event_family(value: str) -> str | None:
    label = re.sub(r"\s+", " ", (value or "").strip().casefold().replace("-", " "))
    if label in {"goal", "goals"}:
        return "goal"
    if label in {"shots on target", "shots off target", "shot", "chance_or_shot", "chance or shot", "chance", "shots", "shot on target", "shot off target"}:
        return "shot_or_chance"
    if "card" in label:
        return "card"
    return None


def load_examples(path: Path) -> tuple[dict[str, list[dict]], str, int]:
    by_match: dict[str, list[dict]] = defaultdict(list)
    sources: set[str] = set()
    skipped = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"match_id", "english_commentary_context", "human_label", "event_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs these fields: {', '.join(sorted(required))}")
        for row in reader:
            match = (row.get("match_id") or "").strip()
            text = (row.get("english_commentary_context") or "").strip()
            label = (row.get("human_label") or "").strip().casefold()
            if not match or not text or label in {"", "unclear"}:
                skipped += 1
                continue
            if label not in {"highlight", "routine"}:
                raise ValueError(f"Unexpected label {label!r} for match {match!r}")
            event_type = event_family(row.get("event_type", ""))
            by_match[match].append({"text": text, "eventness": int(label == "highlight"), "event_type": event_type})
            sources.add((row.get("source") or "unknown source").strip())
    supervision = "mixed weak SoccerNet event timestamps and human-reviewed labels" if len(sources) > 1 else next(iter(sources), "unknown supervision")
    return by_match, supervision, skipped


def fit_model(rows: list[dict], epochs: int = 140) -> tuple[MultiTaskMLP, TfidfEncoder, dict[str, int]]:
    torch.manual_seed(23)
    encoder = TfidfEncoder.fit([row["text"] for row in rows])
    x = torch.stack([encoder.vectorize(row["text"]) for row in rows])
    y_binary = torch.tensor([row["eventness"] for row in rows], dtype=torch.float32)
    event_rows = [(i, EVENT_CLASSES.index(row["event_type"])) for i, row in enumerate(rows) if row["event_type"] in EVENT_CLASSES]
    if len(event_rows) < 10:
        raise ValueError("Need at least 10 rows with a mapped event_type to train the event-family head.")
    indices = torch.tensor([i for i, _ in event_rows], dtype=torch.long)
    targets = torch.tensor([label for _, label in event_rows], dtype=torch.long)
    class_counts = Counter(targets.tolist())
    event_weights = torch.tensor([min(len(targets) / max(1, class_counts.get(i, 0)) / len(EVENT_CLASSES), 5.0) for i in range(len(EVENT_CLASSES))], dtype=torch.float32)
    positive = max(1, int(y_binary.sum().item()))
    negative = max(1, len(y_binary) - positive)
    binary_weight = torch.tensor([min(negative / positive, 5.0)])
    model = MultiTaskMLP(encoder.dimensions, hidden=32, event_classes=len(EVENT_CLASSES), dropout=0.35)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.015)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        binary_logits, type_logits = model(x)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, y_binary, pos_weight=binary_weight)
        type_loss = torch.nn.functional.cross_entropy(type_logits[indices], targets, weight=event_weights)
        loss = binary_loss + 0.7 * type_loss
        loss.backward()
        optimizer.step()
    model.eval()
    return model, encoder, {EVENT_CLASSES[i]: class_counts.get(i, 0) for i in range(len(EVENT_CLASSES))}


def metrics(y_true: list[int], y_pred: list[int], classes: list[int]) -> dict:
    per_class = {}
    f1_values = []
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[str(cls)] = {"support": sum(y == cls for y in y_true), "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {"n": len(y_true), "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true)), "macro_f1": sum(f1_values) / max(1, len(f1_values)), "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/commentary_multitask_tfidf.pt"))
    parser.add_argument("--epochs", type=int, default=140)
    args = parser.parse_args()
    if not args.labels.is_file():
        parser.error(f"Labels file not found: {args.labels}")
    by_match, supervision, skipped = load_examples(args.labels)
    if len(by_match) < 2:
        parser.error("Need labeled examples from at least two matches for match-held-out evaluation.")
    all_rows = [row for rows in by_match.values() for row in rows]
    binary_counts = Counter(row["eventness"] for row in all_rows)
    event_counts = Counter(row["event_type"] for row in all_rows if row["event_type"])
    print(f"Rows: {len(all_rows)}; event/highlight {binary_counts[1]}, routine {binary_counts[0]}, skipped {skipped}")
    print(f"Mapped event-family labels: {dict(event_counts)}")

    binary_predictions: list[int] = []
    binary_truth: list[int] = []
    type_predictions: list[int] = []
    type_truth: list[int] = []
    folds = {}
    for held_out, eval_rows in sorted(by_match.items()):
        train_rows = [row for match, rows in by_match.items() if match != held_out for row in rows]
        if len(train_rows) < 10:
            continue
        model, encoder, _ = fit_model(train_rows, args.epochs)
        eval_indices = [i for i, row in enumerate(eval_rows) if row["event_type"] in EVENT_CLASSES]
        if not eval_indices:
            continue
        x_eval = torch.stack([encoder.vectorize(eval_rows[i]["text"]) for i in eval_indices])
        with torch.inference_mode():
            _, type_logits = model(x_eval)
            predicted = torch.argmax(type_logits, dim=1).tolist()
        truth = [EVENT_CLASSES.index(eval_rows[i]["event_type"]) for i in eval_indices]
        type_predictions.extend(predicted)
        type_truth.extend(truth)
        folds[held_out] = metrics(truth, predicted, sorted(set(truth)))
        print(f"Held-out {held_out} event-family: {json.dumps(folds[held_out])}")

    # Binary head uses all labeled windows and is checked on complete held-out matches.
    binary_folds = {}
    for held_out, eval_rows in sorted(by_match.items()):
        train_rows = [row for match, rows in by_match.items() if match != held_out for row in rows]
        if not train_rows or len({row["eventness"] for row in eval_rows}) < 2:
            continue
        model, encoder, _ = fit_model(train_rows, args.epochs)
        x_eval = torch.stack([encoder.vectorize(row["text"]) for row in eval_rows])
        with torch.inference_mode():
            logits, _ = model(x_eval)
            pred = (torch.sigmoid(logits) >= 0.5).int().tolist()
        truth = [row["eventness"] for row in eval_rows]
        result = metrics(truth, pred, [0, 1])
        binary_folds[held_out] = result
        binary_predictions.extend(pred)
        binary_truth.extend(truth)
        print(f"Held-out {held_out} eventness: n={result['n']} accuracy={result['accuracy']:.3f} macro-F1={result['macro_f1']:.3f}")

    final_model, encoder, final_counts = fit_model(all_rows, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "multitask_tfidf_mlp", "state_dict": final_model.state_dict(), "dimensions": encoder.dimensions, "hidden_dimensions": 32, "dropout": 0.35, "vocabulary": encoder.vocabulary, "idf": encoder.idf, "event_classes": EVENT_CLASSES, "event_type_training_counts": final_counts, "training_matches": sorted(by_match), "training_examples": len(all_rows), "supervision": supervision, "features": "project-fitted TF-IDF unigram/bigram features with shared MLP trunk"}, args.out)
    summary = {"model": str(args.out), "model_type": "project-trained shared TF-IDF MLP with eventness and event-family heads", "event_classes": EVENT_CLASSES, "event_type_training_counts": final_counts, "training_examples": len(all_rows), "matches": sorted(by_match), "supervision": supervision, "held_out_event_family_folds": folds, "held_out_eventness_folds": binary_folds, "warning": "Exploratory match-held-out results. Event-family labels are derived from weak SoccerNet event annotations; class coverage is limited, especially for cards. The model predicts broad event families, not players, excitement, or subjective highlight quality."}
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {args.out}\nSaved evaluation: {metrics_path}")


if __name__ == "__main__":
    main()
