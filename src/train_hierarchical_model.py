#!/usr/bin/env python3
"""Train/evaluate a two-tier commentary model with separate label meanings."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from hierarchical_model import HierarchicalCommentaryNet
from text_classifier import TfidfEncoder
from train_pipeline_model import EVENT_CLASSES, binary_metrics, event_family, family_metrics


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = []
        for row in csv.DictReader(handle):
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            label = (row.get("human_label") or "").strip().casefold()
            if not text or not match or label not in {"highlight", "routine"}:
                continue
            try:
                time = float(row.get("anchor_seconds") or 0)
            except ValueError:
                continue
            result.append({
                "match": match,
                "half": str(row.get("half") or "1"),
                "time": time,
                "text": text,
                "binary": int(label == "highlight"),
                "family": event_family(row.get("event_type", "")),
                "source": "human-reviewed annotation",
            })
    return result


def get_data(weak_path: Path, human_path: Path, reviewed_path: Path, assistant_path: Path | None = None) -> tuple[list[dict], list[dict]]:
    weak = read_rows(weak_path)
    human = read_rows(human_path)
    with reviewed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = (row.get("source") or "").casefold()
            label = (row.get("human_label") or "").strip().casefold()
            text = (row.get("english_commentary_context") or "").strip()
            match = (row.get("match_id") or "").strip()
            if "human-reviewed" not in source or label not in {"highlight", "routine"} or not text or not match:
                continue
            try:
                time = float(row.get("anchor_seconds") or 0)
            except ValueError:
                continue
            human.append({"match": match, "half": str(row.get("half") or "1"), "time": time, "text": text, "binary": int(label == "highlight"), "family": None, "source": "human-reviewed candidate/control"})
    if assistant_path and assistant_path.is_file():
        with assistant_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source = (row.get("annotation_source") or "").casefold()
                label = (row.get("human_label") or "").strip().casefold()
                text = (row.get("english_commentary_context") or "").strip()
                match = (row.get("match_id") or "").strip()
                try:
                    time = float(row.get("anchor_seconds") or 0)
                except ValueError:
                    continue
                if source.startswith("assistant") and label in {"highlight", "routine"} and text and match:
                    human.append({"match": match, "half": str(row.get("half") or "1"), "time": time, "text": text, "binary": int(label == "highlight"), "family": None, "source": "assistant text-only annotation"})
    # Remove exact duplicate reviewed windows while preserving source priority.
    unique = {}
    for row in human:
        key = (row["match"], row["half"], row["time"], row["text"])
        if key not in unique or (unique[key].get("source") == "assistant text-only annotation" and row.get("source") != "assistant text-only annotation"):
            unique[key] = row
    human = list(unique.values())
    if not weak or not human:
        raise ValueError("Need weak event windows and human-reviewed highlight/routine windows")
    return weak, human


def chunk_rows(rows: list[dict], chunk_seconds: float) -> list[list[dict]]:
    sorted_rows = sorted(rows, key=lambda row: row["time"])
    chunks: dict[int, list[dict]] = defaultdict(list)
    for row in sorted_rows:
        chunks[int(max(0, row["time"]) // chunk_seconds)].append(row)
    return [chunks[key] for key in sorted(chunks)]


def prepare_group(rows: list[dict], encoder: TfidfEncoder, chunk_seconds: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    chunks = chunk_rows(rows, chunk_seconds)
    features, gaps = [], []
    for chunk in chunks:
        features.append(torch.stack([encoder.vectorize(row["text"]) for row in chunk]))
        times = [row["time"] for row in chunk]
        gaps.append(torch.tensor([0.0] + [max(0.0, min(120.0, times[i] - times[i - 1])) / 120 for i in range(1, len(times))], dtype=torch.float32))
    return features, gaps


def group_examples(rows: list[dict]) -> list[list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["match"], row["half"])].append(row)
    return [sorted(values, key=lambda row: row["time"]) for _, values in sorted(groups.items())]


def build_model(weak: list[dict], human: list[dict], epochs: int, chunk_seconds: float) -> tuple[HierarchicalCommentaryNet, TfidfEncoder]:
    torch.manual_seed(73)
    encoder = TfidfEncoder.fit([row["text"] for row in weak + human])
    model = HierarchicalCommentaryNet(encoder.dimensions, len(EVENT_CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.03)
    event_groups, human_groups = group_examples(weak), group_examples(human)
    family_rows = [row for row in weak if row["family"] in EVENT_CLASSES]
    family_counts = Counter(row["family"] for row in family_rows)
    total = max(1, len(family_rows))
    family_weights = torch.tensor([min(total / max(1, family_counts.get(name, 0)) / len(EVENT_CLASSES), 8) for name in EVENT_CLASSES])
    order = [("weak", group) for group in event_groups] + [("human", group) for group in human_groups]
    rng = random.Random(73)
    for _ in range(epochs):
        rng.shuffle(order)
        model.train()
        for task, group in order:
            chunks, gaps = prepare_group(group, encoder, chunk_seconds)
            output = model(chunks, gaps)["chunks"]
            ordered_rows = [row for chunk in chunk_rows(group, chunk_seconds) for row in chunk]
            logits = torch.cat([part["eventness" if task == "weak" else "highlight"] for part in output])
            targets = torch.tensor([row["binary"] for row in ordered_rows], dtype=torch.float32)
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss = binary_loss
            if task == "weak":
                family_logits = torch.cat([part["event_types"] for part in output])
                family_targets = [EVENT_CLASSES.index(row["family"]) for row in ordered_rows if row["family"] in EVENT_CLASSES]
                if family_targets:
                    valid = [i for i, row in enumerate(ordered_rows) if row["family"] in EVENT_CLASSES]
                    loss = loss + 0.5 * torch.nn.functional.cross_entropy(family_logits[valid], torch.tensor(family_targets), weight=family_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model, encoder


def predict(model: HierarchicalCommentaryNet, encoder: TfidfEncoder, rows: list[dict], chunk_seconds: float) -> tuple[list[float], list[float], list[int]]:
    if not rows:
        return [], [], []
    chunks, gaps = prepare_group(rows, encoder, chunk_seconds)
    flat = [row for chunk in chunk_rows(rows, chunk_seconds) for row in chunk]
    with torch.inference_mode():
        outputs = model(chunks, gaps)["chunks"]
        eventness = torch.sigmoid(torch.cat([x["eventness"] for x in outputs])).tolist()
        highlight = torch.sigmoid(torch.cat([x["highlight"] for x in outputs])).tolist()
        families = torch.argmax(torch.cat([x["event_types"] for x in outputs]), dim=1).tolist()
    return eventness, highlight, families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-labels", type=Path, default=Path("data/event_aligned_context.csv"))
    parser.add_argument("--human-labels", type=Path, default=Path("data/manual_annotation_context.csv"))
    parser.add_argument("--reviewed-labels", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    parser.add_argument("--assistant-labels", type=Path, default=None, help="Opt-in assistant text-only pseudo-labels for an explicitly marked experiment; never used by default")
    parser.add_argument("--out", type=Path, default=Path("models/commentary_hierarchical.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chunk-seconds", type=float, default=300)
    args = parser.parse_args()
    weak, human = get_data(args.weak_labels, args.human_labels, args.reviewed_labels, args.assistant_labels)
    assistant_count = sum(row.get("source") == "assistant text-only annotation" for row in human)
    supervision = ("separate SoccerNet event alignment, human-reviewed labels, and explicitly marked assistant text-only pseudo-labels"
                  if assistant_count else "separate SoccerNet event alignment and human-reviewed highlight/routine labels")
    matches = sorted({row["match"] for row in weak + human})
    print(f"Weak event windows: {len(weak)}; highlight training windows: {len(human)}; assistant text-only windows: {assistant_count}; whole-match folds: {len(matches)}")
    aggregate: dict[str, list] = {"event_true": [], "event_scores": [], "human_true": [], "human_scores": [], "family_true": [], "family_pred": []}
    fold_metrics = {}
    for held_out in matches:
        train_weak = [row for row in weak if row["match"] != held_out]
        train_human = [row for row in human if row["match"] != held_out]
        test_weak = [row for row in weak if row["match"] == held_out]
        test_human = [row for row in human if row["match"] == held_out]
        if not train_weak or not train_human:
            continue
        model, encoder = build_model(train_weak, train_human, args.epochs, args.chunk_seconds)
        fold = {}
        if test_weak:
            sorted_rows = [row for group in group_examples(test_weak) for row in group]
            event_scores, families = [], []
            for group in group_examples(test_weak):
                score_part, _, family_part = predict(model, encoder, group, args.chunk_seconds)
                event_scores.extend(score_part); families.extend(family_part)
            truth = [row["binary"] for row in sorted_rows]
            fold["event_relatedness"] = binary_metrics(truth, event_scores)
            aggregate["event_true"].extend(truth); aggregate["event_scores"].extend(event_scores)
            valid = [(row, families[i]) for i, row in enumerate(sorted_rows) if row["family"] in EVENT_CLASSES]
            if valid:
                aggregate["family_true"].extend(EVENT_CLASSES.index(row["family"]) for row, _ in valid)
                aggregate["family_pred"].extend(pred for _, pred in valid)
        eval_human = [row for row in test_human if row.get("source") != "assistant text-only annotation"]
        if eval_human:
            sorted_human = [row for group in group_examples(eval_human) for row in group]
            scores = []
            for group in group_examples(eval_human):
                _, score_part, _ = predict(model, encoder, group, args.chunk_seconds)
                scores.extend(score_part)
            truth = [row["binary"] for row in sorted_human]
            fold["human_highlight"] = binary_metrics(truth, scores)
            aggregate["human_true"].extend(truth); aggregate["human_scores"].extend(scores)
        fold_metrics[held_out] = fold
        print(f"Held out {held_out}: {json.dumps(fold, sort_keys=True)}")
    metrics = {
        "model_type": "two-tier project-trained hierarchical GRU with temporal attention",
        "training_counts": {"weak_event_relatedness": len(weak), "highlight_training_rows": len(human), "assistant_text_only_rows": sum(row.get("source") == "assistant text-only annotation" for row in human), "human_only_evaluation_rows": sum(row.get("source") != "assistant text-only annotation" for row in human), "matches": len(matches)},
        "chunk_seconds": args.chunk_seconds,
        "threshold": 0.5,
        "leave_one_match_out": fold_metrics,
        "aggregate_event_relatedness": binary_metrics(aggregate["event_true"], aggregate["event_scores"]),
        "aggregate_human_highlight": binary_metrics(aggregate["human_true"], aggregate["human_scores"]),
        "aggregate_event_family": family_metrics(aggregate["family_true"], aggregate["family_pred"]) if aggregate["family_true"] else {},
        "warning": ("Exploratory only. Assistant text-only annotations are opt-in pseudo-labels and are excluded from human-only evaluation; sparse windows and limited match coverage do not establish broad generalization. Attention weights are not causal explanations."
                    if assistant_count else "Exploratory only. Sparse windows and limited match coverage do not establish broad generalization. Attention weights are not causal explanations."),
    }
    final_model, encoder = build_model(weak, human, args.epochs, args.chunk_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "hierarchical_commentary_multitask_v1", "state_dict": final_model.state_dict(),
        "vocabulary": encoder.vocabulary, "idf": encoder.idf, "dimensions": encoder.dimensions,
        "hidden_dimensions": 32, "attention_heads": 4, "dropout": 0.25,
        "event_classes": EVENT_CLASSES, "chunk_seconds": args.chunk_seconds,
        "training_matches": matches, "training_examples": {"weak": len(weak), "highlight_all": len(human), "assistant_text_only": sum(row.get("source") == "assistant text-only annotation" for row in human)},
        "supervision": supervision,
    }, args.out)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved hierarchical model: {args.out}\nSaved metrics: {args.out.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
