#!/usr/bin/env python3
"""Add reviewed candidate/control windows to the weakly supervised examples."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Invalid timestamp: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True, help="Completed candidate review CSV")
    parser.add_argument("--transcript", type=Path, default=Path("outputs/autolang_full/transcript.csv"))
    parser.add_argument("--weak-labels", type=Path, default=Path("data/event_aligned_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/event_and_manual_review_context.csv"))
    args = parser.parse_args()

    transcript = []
    with args.transcript.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            transcript.append((float(row["start_seconds"]), float(row["end_seconds"]),
                               row.get("text_english") or row.get("text") or ""))

    review_rows = []
    with args.reviews.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("review_label") or "").strip().casefold()
            if label in {"highlight", "missed highlight", "useful"}:
                target = "highlight"
            elif label in {"routine", "wrong"}:
                target = "routine" if label == "routine" else "unclear"
            else:
                continue
            center = seconds(row["start_timestamp"])
            context = " ".join(text for start, end, text in transcript
                               if end >= center - 15 and start <= center + 15).strip()
            if not context:
                continue
            review_rows.append({
                "match_id": "crystal_palace_mancity", "half": 1,
                "window_start_seconds": round(max(0, center - 15), 2),
                "window_end_seconds": round(center + 15, 2), "anchor_seconds": round(center, 2),
                "english_commentary_context": context, "human_label": target,
                "event_type": (row.get("verified_event") or row.get("event") or "reviewed_control"),
                "annotator": "team_review", "notes": row.get("notes", ""),
                "source": "Human-reviewed Crystal Palace examples",
            })

    if not review_rows:
        parser.error("No completed review labels found. Fill review_label before building training data.")
    with args.weak_labels.open(encoding="utf-8-sig", newline="") as f:
        weak_rows = list(csv.DictReader(f))
        fields = list(weak_rows[0]) if weak_rows else list(review_rows[0])
    for field in review_rows[0]:
        if field not in fields:
            fields.append(field)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(weak_rows)
        writer.writerows(review_rows)
    positives = sum(r["human_label"] == "highlight" for r in review_rows)
    negatives = sum(r["human_label"] == "routine" for r in review_rows)
    unclear = sum(r["human_label"] == "unclear" for r in review_rows)
    print(f"Added {len(review_rows)} reviewed windows: {positives} highlight, {negatives} routine, {unclear} unclear.")
    print(f"Combined dataset saved: {args.out} ({len(weak_rows)} weak + {len(review_rows)} human-reviewed rows).")
    print("The trainer holds out complete matches, so the reviewed match remains excluded from its own evaluation fold.")


if __name__ == "__main__":
    main()
