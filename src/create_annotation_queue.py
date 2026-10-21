#!/usr/bin/env python3
"""Create a blinded, match-balanced human review batch from aligned windows.

Weak SoccerNet event labels are used only to balance the queue. The source
label is deliberately omitted; reviewers decide highlight/routine from text
and, when available, the matching footage.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


OUTPUT_FIELDS = [
    "review_id", "assignee", "match_id", "half", "window_start_seconds", "window_end_seconds",
    "anchor_seconds", "english_commentary_context", "human_label", "event_type",
    "excitement_label", "focal_player", "pronoun_link", "role_action", "role_target",
    "annotation_source",
    "annotator", "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/event_aligned_context.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/human_review_batch_01.csv"))
    parser.add_argument("--per-class", type=int, default=5, help="Rows per weak class, per match-half")
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")

    grouped: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: {"event": [], "routine": []})
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("human_label") or "").strip().casefold()
            key = "event" if label == "highlight" else "routine" if label == "routine" else ""
            match = (row.get("match_id") or "").strip()
            if key and match and (row.get("english_commentary_context") or "").strip():
                grouped[(match, str(row.get("half") or "1"))][key].append(row)

    rng = random.Random(args.seed)
    output: list[dict[str, str]] = []
    for group_number, ((match, half), classes) in enumerate(sorted(grouped.items())):
        selected: list[dict[str, str]] = []
        for label in ("event", "routine"):
            candidates = list(classes[label])
            rng.shuffle(candidates)
            chosen: list[dict[str, str]] = []
            for row in candidates:
                try:
                    anchor = float(row.get("anchor_seconds") or 0)
                except ValueError:
                    continue
                if all(abs(anchor - float(other.get("anchor_seconds") or 0)) >= 25 for other in chosen):
                    chosen.append(row)
                    if len(chosen) == args.per_class:
                        break
            selected.extend(chosen)
        rng.shuffle(selected)
        for position, row in enumerate(selected):
            anchor = float(row.get("anchor_seconds") or 0)
            output.append({
                "review_id": f"R{len(output)+1:04d}",
                "assignee": f"Member {(group_number + position) % 3 + 1}",
                "match_id": match,
                "half": half,
                "window_start_seconds": str(max(0, anchor - 15)),
                "window_end_seconds": str(anchor + 15),
                "anchor_seconds": str(anchor),
                "english_commentary_context": row["english_commentary_context"],
                "human_label": "",
                "event_type": "",
                "excitement_label": "",
                "focal_player": "",
                "pronoun_link": "",
                "role_action": "",
                "role_target": "",
                "annotation_source": "",
                "annotator": "",
                "notes": "",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    print(f"Created {len(output)} blinded review rows across {len(grouped)} match-half groups: {args.out.resolve()}")
    print("Reviewers: label the commentary/footage independently; use highlight, routine, or unclear. Source weak labels are not included.")


if __name__ == "__main__":
    main()
