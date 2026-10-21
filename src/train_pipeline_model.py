#!/usr/bin/env python3
"""Train the project TF-IDF model with correctly separated supervision.

SoccerNet event alignments train event-relatedness and event-family heads.
Human-reviewed highlight/routine windows train a distinct highlight-value head.
Whole matches are held out together in evaluation so neither task leaks a
match's wording into its own training fold.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from text_classifier import PipelineMultiTaskMLP, TfidfEncoder

EVENT_CLASSES = ["goal", "shot_or_chance", "card"]


def event_family(value: str) -> str | None:
    label = re.sub(r"\s+", " ", (value or "").strip().casefold().replace("-", " "))
    if label in {"goal", "goals"}:
        return "goal"
    if label in {
        "shots on target", "shots off target", "shot", "chance_or_shot",
        "chance or shot", "chance", "shots", "shot on target", "shot off target",
    }:
        return "shot_or_chance"
    if "card" in label:
        return "card"
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_data(weak_path: Path, manual_path: Path, reviewed_path: Path, assistant_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    weak_rows: list[dict[str, Any]] = []
    for row in _read_csv(weak_path):
        match = (row.get("match_id") or "").strip()
        text = (row.get("english_commentary_context") or "").strip()
        label = (row.get("human_label") or "").strip().casefold()
        if not match or not text or label not in {"highlight", "routine"}:
            continue
        weak_rows.append({
            "match_id": match,
            "text": text,
            "eventness": int(label == "highlight"),
            "event_type": event_family(row.get("event_type", "")),
            "source": (row.get("source") or "SoccerNet event alignment").strip(),
        })

    human_rows: list[dict[str, Any]] = []
    for row in _read_csv(manual_path):
        label = (row.get("human_label") or "").strip().casefold()
        if label in {"highlight", "routine"}:
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            if text and match:
                human_rows.append({"match_id": match, "text": text, "highlight": int(label == "highlight"), "source": "manual_annotation_context.csv"})

    reviewed = [row for row in _read_csv(reviewed_path) if (row.get("source") or "").startswith("Human-reviewed")]
    for row in reviewed:
        label = (row.get("human_label") or "").strip().casefold()
        text = (row.get("english_commentary_context") or "").strip()
        match = (row.get("match_id") or "").strip()
        if label in {"highlight", "routine"} and text and match:
            human_rows.append({"match_id": match, "text": text, "highlight": int(label == "highlight"), "source": "human-reviewed candidate/control"})

    # Optional text-only assistant labels augment training only. They are
    # deliberately excluded from human-only held-out evaluation below.
    if assistant_path and assistant_path.is_file():
        for row in _read_csv(assistant_path):
            label = (row.get("human_label") or "").strip().casefold()
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            if label in {"highlight", "routine"} and text and match and (row.get("annotation_source") or "").startswith("assistant"):
                human_rows.append({"match_id": match, "text": text, "highlight": int(label == "highlight"), "source": "assistant text-only annotation"})

    if not weak_rows:
        raise ValueError(f"No usable SoccerNet event examples in {weak_path}")
    if not human_rows:
        raise ValueError(f"No usable human highlight labels in {manual_path} or {reviewed_path}")
    weak_matches = {row["match_id"] for row in weak_rows}
    # Keep the first observation for an exact duplicate, so reviewed labels
    # retain priority over later assistant pseudo-labels.
    deduplicated = {}
    for row in human_rows:
        key = (row["match_id"], row["text"])
        if key not in deduplicated or (deduplicated[key]["source"] == "assistant text-only annotation" and row["source"] != "assistant text-only annotation"):
            deduplicated[key] = row
    human_rows = list(deduplicated.values())
    human_matches = {row["match_id"] for row in human_rows}
    matches = weak_matches | human_matches
    if len(matches) < 2:
        raise ValueError("Need at least two distinct matches for match-held-out evaluation")
    has_assistant = any(row["source"] == "assistant text-only annotation" for row in human_rows)
    supervision = ("separate SoccerNet event-alignment labels, human-reviewed highlight labels, and explicitly sourced assistant text-only pseudo-labels"
                   if has_assistant else "separate SoccerNet event-alignment labels and human-reviewed highlight labels")
    return weak_rows, human_rows, supervision


def _class_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    labels = [EVENT_CLASSES.index(row["event_type"]) for row in rows if row["event_type"] in EVENT_CLASSES]
    counts = Counter(labels)
    total = max(1, len(labels))
    return torch.tensor([min(total / max(1, counts.get(i, 0)) / len(EVENT_CLASSES), 8.0) for i in range(len(EVENT_CLASSES))])


def fit_model(weak_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]], epochs: int) -> tuple[PipelineMultiTaskMLP, TfidfEncoder]:
    texts = [row["text"] for row in weak_rows] + [row["text"] for row in human_rows]
    encoder = TfidfEncoder.fit(texts)
    model = PipelineMultiTaskMLP(encoder.dimensions, hidden=48, event_classes=len(EVENT_CLASSES), dropout=0.35)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)

    x_weak = torch.stack([encoder.vectorize(row["text"]) for row in weak_rows])
    y_event = torch.tensor([row["eventness"] for row in weak_rows], dtype=torch.float32)
    family_indices = [i for i, row in enumerate(weak_rows) if row["event_type"] in EVENT_CLASSES]
    x_human = torch.stack([encoder.vectorize(row["text"]) for row in human_rows])
    y_highlight = torch.tensor([row["highlight"] for row in human_rows], dtype=torch.float32)

    n_positive = max(1, int(y_event.sum().item()))
    n_negative = max(1, len(y_event) - n_positive)
    event_pos_weight = torch.tensor([min(n_negative / n_positive, 6.0)])
    n_positive = max(1, int(y_highlight.sum().item()))
    n_negative = max(1, len(y_highlight) - n_positive)
    highlight_pos_weight = torch.tensor([min(n_negative / n_positive, 6.0)])
    if family_indices:
        family_targets = torch.tensor([EVENT_CLASSES.index(weak_rows[i]["event_type"]) for i in family_indices], dtype=torch.long)
        type_weights = _class_weights(weak_rows)
    else:
        family_targets = torch.empty(0, dtype=torch.long)
        type_weights = torch.ones(len(EVENT_CLASSES))

    torch.manual_seed(47)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        event_logits, _, event_type_logits = model(x_weak)
        _, highlight_logits, _ = model(x_human)
        event_loss = torch.nn.functional.binary_cross_entropy_with_logits(event_logits, y_event, pos_weight=event_pos_weight)
        highlight_loss = torch.nn.functional.binary_cross_entropy_with_logits(highlight_logits, y_highlight, pos_weight=highlight_pos_weight)
        if family_indices:
            family_loss = torch.nn.functional.cross_entropy(event_type_logits[family_indices], family_targets, weight=type_weights)
        else:
            family_loss = torch.zeros((), dtype=torch.float32)
        # Human-highlight judgments receive equal priority to weak eventness;
        # broad event-family prediction is an auxiliary task.
        loss = event_loss + highlight_loss + 0.5 * family_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model, encoder


def binary_metrics(y_true: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, Any]:
    y_pred = [int(value >= threshold) for value in probabilities]
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1_positive = 2 * precision * recall / max(1e-12, precision + recall)
    precision_negative = tn / max(1, tn + fn)
    recall_negative = tn / max(1, tn + fp)
    f1_negative = 2 * precision_negative * recall_negative / max(1e-12, precision_negative + recall_negative)
    return {
        "n": len(y_true), "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision, "recall": recall,
        "f1_positive": f1_positive, "macro_f1": (f1_positive + f1_negative) / 2,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def family_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    values = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[EVENT_CLASSES[label]] = {"support": sum(y == label for y in y_true), "precision": precision, "recall": recall, "f1": f1}
        values.append(f1)
    return {"n": len(y_true), "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true)), "macro_f1": sum(values) / max(1, len(values)), "per_class": per_class}


def _predict(model: PipelineMultiTaskMLP, encoder: TfidfEncoder, texts: list[str]) -> tuple[list[float], list[float], list[int]]:
    x = torch.stack([encoder.vectorize(text) for text in texts])
    with torch.inference_mode():
        event_logits, highlight_logits, family_logits = model(x)
        return (
            torch.sigmoid(event_logits).tolist(),
            torch.sigmoid(highlight_logits).tolist(),
            torch.argmax(family_logits, dim=1).tolist(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-labels", type=Path, default=Path("data/event_aligned_context.csv"))
    parser.add_argument("--human-labels", type=Path, default=Path("data/manual_annotation_context.csv"))
    parser.add_argument("--reviewed-labels", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    parser.add_argument("--assistant-labels", type=Path, default=None, help="Opt-in assistant text-only pseudo-labels for an explicitly marked experiment; never used by default")
    parser.add_argument("--out", type=Path, default=Path("models/commentary_pipeline_multitask.pt"))
    parser.add_argument("--epochs", type=int, default=160)
    args = parser.parse_args()
    for path in (args.weak_labels, args.human_labels, args.reviewed_labels):
        if not path.is_file():
            parser.error(f"Required label file not found: {path}")
    weak_rows, human_rows, supervision = load_data(args.weak_labels, args.human_labels, args.reviewed_labels, args.assistant_labels)
    matches = sorted({row["match_id"] for row in weak_rows + human_rows})
    print(f"Weak event windows: {len(weak_rows)} across {len({row['match_id'] for row in weak_rows})} matches")
    print(f"Highlight training windows: {len(human_rows)} across {len({row['match_id'] for row in human_rows})} matches")
    print(f"Assistant text-only training windows: {sum(row['source'] == 'assistant text-only annotation' for row in human_rows)} (excluded from human held-out metrics)")
    print(f"Human labels: {dict(Counter(row['highlight'] for row in human_rows))}")
    print("Evaluation: leave one entire match out; threshold fixed at 0.50")

    fold_results: dict[str, Any] = {}
    aggregate = {"eventness": {0: [], 1: []}, "human_highlight": {0: [], 1: []}, "event_family": {"true": [], "pred": []}}
    for held_out in matches:
        train_weak = [row for row in weak_rows if row["match_id"] != held_out]
        train_human = [row for row in human_rows if row["match_id"] != held_out]
        test_weak = [row for row in weak_rows if row["match_id"] == held_out]
        test_human = [row for row in human_rows if row["match_id"] == held_out and row["source"] != "assistant text-only annotation"]
        if not train_weak or not train_human:
            continue
        model, encoder = fit_model(train_weak, train_human, args.epochs)
        fold: dict[str, Any] = {}
        if test_weak:
            event_scores, _, predicted_families = _predict(model, encoder, [row["text"] for row in test_weak])
            y_event = [row["eventness"] for row in test_weak]
            fold["eventness"] = binary_metrics(y_event, event_scores)
            aggregate["eventness"][0].extend(y_event)
            aggregate["eventness"][1].extend(event_scores)
            known = [i for i, row in enumerate(test_weak) if row["event_type"] in EVENT_CLASSES]
            if known:
                y_type = [EVENT_CLASSES.index(test_weak[i]["event_type"]) for i in known]
                p_type = [predicted_families[i] for i in known]
                fold["event_family"] = family_metrics(y_type, p_type)
                aggregate["event_family"]["true"].extend(y_type)
                aggregate["event_family"]["pred"].extend(p_type)
        if test_human:
            _, highlight_scores, _ = _predict(model, encoder, [row["text"] for row in test_human])
            y_highlight = [row["highlight"] for row in test_human]
            fold["human_highlight"] = binary_metrics(y_highlight, highlight_scores)
            aggregate["human_highlight"][0].extend(y_highlight)
            aggregate["human_highlight"][1].extend(highlight_scores)
        fold_results[held_out] = fold
        print(f"Held out {held_out}: {json.dumps(fold, sort_keys=True)}")

    aggregate_results = {
        "eventness": binary_metrics(aggregate["eventness"][0], aggregate["eventness"][1]) if aggregate["eventness"][0] else {},
        "human_highlight": binary_metrics(aggregate["human_highlight"][0], aggregate["human_highlight"][1]) if aggregate["human_highlight"][0] else {},
        "event_family": family_metrics(aggregate["event_family"]["true"], aggregate["event_family"]["pred"]) if aggregate["event_family"]["true"] else {},
    }
    print("Aggregate held-out metrics:", json.dumps(aggregate_results, sort_keys=True))

    final_model, encoder = fit_model(weak_rows, human_rows, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "pipeline_multitask_tfidf_mlp_v1",
        "state_dict": final_model.state_dict(),
        "dimensions": encoder.dimensions,
        "hidden_dimensions": 48,
        "dropout": 0.35,
        "vocabulary": encoder.vocabulary,
        "idf": encoder.idf,
        "event_classes": EVENT_CLASSES,
        "training_matches": matches,
        "weak_event_examples": len(weak_rows),
        "highlight_training_examples": len(human_rows),
        "human_only_highlight_examples": sum(row["source"] != "assistant text-only annotation" for row in human_rows),
        "assistant_text_only_examples": sum(row["source"] == "assistant text-only annotation" for row in human_rows),
        "supervision": supervision,
        "highlight_supervision": ("existing human-reviewed labels plus explicitly marked assistant text-only pseudo-labels; SoccerNet event alignment is kept separate"
                                  if any(row["source"] == "assistant text-only annotation" for row in human_rows)
                                  else "existing human-reviewed labels only; SoccerNet event alignment is kept separate"),
        "features": "project-fitted unigram/bigram TF-IDF plus punctuation, capitalization, and negation cues",
    }, args.out)
    report = {
        "model": str(args.out),
        "model_type": "project-trained shared TF-IDF/PyTorch MLP with separate eventness, human-highlight, and weak event-family heads",
        "supervision": supervision,
        "event_family_classes": EVENT_CLASSES,
        "weak_event_examples": len(weak_rows),
        "highlight_training_examples": len(human_rows),
        "human_only_highlight_examples": sum(row["source"] != "assistant text-only annotation" for row in human_rows),
        "assistant_text_only_examples": sum(row["source"] == "assistant text-only annotation" for row in human_rows),
        "training_matches": matches,
        "evaluation": "leave-one-match-out; TF-IDF vocabulary and IDF fit on training matches in each fold; fixed threshold 0.50",
        "folds": fold_results,
        "aggregate_held_out": aggregate_results,
        "warnings": [
            "SoccerNet event timing labels supervise event-relatedness and event family, not subjective highlight value.",
            "Assistant text-only annotations are opt-in pseudo-labels, not human ground truth; if supplied, they are omitted from held-out human metrics.",
            "Human highlight labels are a small pilot; their held-out metrics are exploratory and should not be presented as broad reliability.",
            "Event-family accuracy may hide weak recall for rare goal/card classes; inspect per-class precision and recall.",
            "The model is trained on English commentary. Non-English inputs require ASR translation to English and can inherit translation errors.",
        ],
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.out}\nSaved evaluation: {metrics_path}")


if __name__ == "__main__":
    main()
