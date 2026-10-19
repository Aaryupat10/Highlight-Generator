#!/usr/bin/env python3
"""Train/evaluate an experimental match-half BiGRU temporal-attention model."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from sequence_model import MatchHalfAttentionNet
from text_classifier import TfidfEncoder
from train_multitask_model import EVENT_CLASSES, event_family


def read_sequences(path: Path) -> tuple[dict[tuple[str, str], list[dict]], str, int]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sources: set[str] = set()
    skipped = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            match = (row.get("match_id") or "").strip()
            half = str(row.get("half") or "1").strip()
            text = (row.get("english_commentary_context") or "").strip()
            label = (row.get("human_label") or "").strip().casefold()
            if not match or not text or label not in {"highlight", "routine"}:
                skipped += 1
                continue
            try:
                seconds = float(row.get("anchor_seconds") or 0)
            except ValueError:
                seconds = 0.0
            groups[(match, half)].append({"match": match, "half": half, "time": seconds, "text": text,
                                          "eventness": int(label == "highlight"), "event_type": event_family(row.get("event_type", ""))})
            sources.add((row.get("source") or "unknown source").strip())
    for values in groups.values():
        values.sort(key=lambda row: row["time"])
    supervision = "weak SoccerNet event timestamps plus manually reviewed examples" if len(sources) > 1 else next(iter(sources), "unknown")
    return groups, supervision, skipped


def class_weights(rows: list[dict]) -> torch.Tensor:
    counts = Counter(EVENT_CLASSES.index(row["event_type"]) for row in rows if row["event_type"] in EVENT_CLASSES)
    if not counts:
        return torch.ones(len(EVENT_CLASSES))
    total = sum(counts.values())
    return torch.tensor([min(total / max(1, counts.get(i, 0)) / len(EVENT_CLASSES), 12.0) for i in range(len(EVENT_CLASSES))], dtype=torch.float32)


def fit(groups: list[list[dict]], epochs: int = 35) -> tuple[MatchHalfAttentionNet, TfidfEncoder]:
    rows = [row for group in groups for row in group]
    torch.manual_seed(31)
    encoder = TfidfEncoder.fit([row["text"] for row in rows])
    model = MatchHalfAttentionNet(encoder.dimensions, len(EVENT_CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.025)
    positives = sum(row["eventness"] for row in rows)
    negatives = len(rows) - positives
    positive_weight = torch.tensor([min(negatives / max(1, positives), 5.0)], dtype=torch.float32)
    type_weight = class_weights(rows)
    for _ in range(epochs):
        model.train()
        for group in groups:
            x = torch.stack([encoder.vectorize(row["text"]) for row in group]).unsqueeze(0)
            times = [row["time"] for row in group]
            gaps = [0.0] + [max(0.0, min(120.0, times[i] - times[i - 1])) / 120.0 for i in range(1, len(times))]
            gap_tensor = torch.tensor(gaps, dtype=torch.float32).unsqueeze(0)
            y_binary = torch.tensor([[row["eventness"] for row in group]], dtype=torch.float32)
            type_indices = [i for i, row in enumerate(group) if row["event_type"] in EVENT_CLASSES]
            optimizer.zero_grad()
            logits, type_logits, _ = model(x, gap_tensor)
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_binary, pos_weight=positive_weight)
            if type_indices:
                targets = torch.tensor([EVENT_CLASSES.index(group[i]["event_type"]) for i in type_indices], dtype=torch.long)
                type_loss = torch.nn.functional.cross_entropy(type_logits[0, type_indices], targets, weight=type_weight)
                loss = binary_loss + 0.6 * type_loss
            else:
                loss = binary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    return model, encoder


def summarize(truth: list[int], predicted: list[int]) -> dict:
    result = {"n": len(truth), "accuracy": sum(a == b for a, b in zip(truth, predicted)) / max(1, len(truth))}
    values = []
    for label in sorted(set(truth)):
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        values.append(f1)
        result[str(label)] = {"support": sum(a == label for a in truth), "precision": precision, "recall": recall, "f1": f1}
    result["macro_f1"] = sum(values) / max(1, len(values))
    return result


def predict_groups(model: MatchHalfAttentionNet, encoder: TfidfEncoder, groups: list[list[dict]]) -> list[dict]:
    predictions = []
    for group in groups:
        x = torch.stack([encoder.vectorize(row["text"]) for row in group]).unsqueeze(0)
        times = [row["time"] for row in group]
        gaps = [0.0] + [max(0.0, min(120.0, times[i] - times[i - 1])) / 120.0 for i in range(1, len(times))]
        with torch.inference_mode():
            logits, type_logits, _ = model(x, torch.tensor(gaps, dtype=torch.float32).unsqueeze(0))
            binary = (torch.sigmoid(logits.squeeze(0)) >= 0.5).int().tolist()
            typed = torch.argmax(type_logits.squeeze(0), dim=-1).tolist()
        predictions.extend((row, binary[i], typed[i]) for i, row in enumerate(group))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/commentary_sequence_attention.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    if not args.labels.is_file():
        parser.error(f"Labels file not found: {args.labels}")
    grouped, supervision, skipped = read_sequences(args.labels)
    matches = sorted({key[0] for key in grouped})
    if len(matches) < 3:
        parser.error("Need examples from at least three matches for match-held-out evaluation.")
    print(f"Grouped sequences: {len(grouped)} match-halves, {len(matches)} matches; {sum(map(len, grouped.values()))} windows; {skipped} skipped")
    event_counts = Counter(row["event_type"] for group in grouped.values() for row in group if row["event_type"])
    print(f"Weak event-family counts: {dict(event_counts)}")

    heldout = {}
    y_true: list[int] = []
    y_pred: list[int] = []
    for match in matches:
        train_groups = [rows for (m, _), rows in grouped.items() if m != match]
        eval_groups = [rows for (m, _), rows in grouped.items() if m == match]
        if not train_groups or not eval_groups:
            continue
        model, encoder = fit(train_groups, args.epochs)
        predicted = predict_groups(model, encoder, eval_groups)
        truth_match = [row["eventness"] for row, _, _ in predicted]
        pred_match = [pred for _, pred, _ in predicted]
        heldout[match] = summarize(truth_match, pred_match)
        y_true.extend(truth_match)
        y_pred.extend(pred_match)
        print(f"Held-out {match}: {json.dumps(heldout[match])}")
    aggregate = summarize(y_true, y_pred)
    print(f"Aggregate eventness: {json.dumps(aggregate)}")

    all_groups = list(grouped.values())
    final_model, encoder = fit(all_groups, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "match_half_attention_mlp", "state_dict": final_model.state_dict(), "vocabulary": encoder.vocabulary, "idf": encoder.idf, "dimensions": encoder.dimensions, "hidden_dimensions": 48, "attention_heads": 4, "dropout": 0.25, "event_classes": EVENT_CLASSES, "training_matches": matches, "training_examples": sum(map(len, all_groups)), "supervision": supervision, "features": "project-fitted TF-IDF commentary contexts with time gaps, BiGRU and temporal self-attention"}, args.out)
    report = {"model_type": "team-trained BiGRU with temporal self-attention and eventness/event-family heads", "training_matches": matches, "training_sequences": len(all_groups), "training_examples": sum(map(len, all_groups)), "event_family_counts": dict(event_counts), "supervision": supervision, "leave_one_match_out_eventness": heldout, "aggregate_leave_one_match_out_eventness": aggregate, "warning": "Exploratory small-data result. Sequence rows are sparse windows, not continuous audio. Weak SoccerNet event labels supervise eventness and broad event families; the attention weights are learned context weights, not human explanations or ground truth. Do not claim generalization without more independently reviewed matches."}
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved sequence model: {args.out}\nSaved evaluation: {metrics_path}")


if __name__ == "__main__":
    main()
