#!/usr/bin/env python3
"""Align SoccerNet event annotations with Echoes transcripts for event training.

This creates weak event-supervision examples: commentary near an annotated
match event is positive; commentary well away from those events is routine.
The labels identify event-related commentary, not subjective highlight quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MATCHES = {
    "arsenal_tottenham": "2015-11-08 - 19-00 Arsenal 1 - 1 Tottenham",
    "leicester_arsenal": "2015-09-26 - 17-00 Leicester 2 - 5 Arsenal",
    "liverpool_aston_villa": "2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa",
    "mancity_westbrom": "2016-04-09 - 19-30 Manchester City 2 - 1 West Brom",
    "manutd_westbrom": "2015-11-07 - 18-00 Manchester United 2 - 0 West Brom",
    "newcastle_chelsea": "2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea",
}
POSITIVE_LABELS = {"goal", "shots on target", "shots off target", "yellow card", "red card"}


def norm_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def load_segments(path: Path, half: int) -> list[tuple[float, float, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["segments"]
    values = raw.values() if isinstance(raw, dict) else raw
    result = []
    for item in values:
        if isinstance(item, dict):
            start, end, text = item["start"], item["end"], item["text"]
        else:
            start, end, text = item[:3]
        if str(text).strip():
            result.append((float(start), float(end), str(text).strip()))
    return sorted(result)


def context_at(segments: list[tuple[float, float, str]], center: float, radius: float = 15.0) -> str:
    return " ".join(text for start, end, text in segments if end >= center - radius and start <= center + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SoccerNet/england_epl/2015-2016"))
    parser.add_argument("--transcripts", type=Path, default=Path("data/echoes"))
    parser.add_argument("--out", type=Path, default=Path("data/event_aligned_context.csv"))
    parser.add_argument("--negative-every", type=float, default=120.0, help="Seconds between routine negative samples")
    args = parser.parse_args()
    rows: list[dict[str, str | float | int]] = []

    for match_id, match_dir in MATCHES.items():
        labels_path = args.dataset_root / match_dir / "Labels-v2.json"
        if not labels_path.is_file():
            print(f"Skipping {match_id}: missing {labels_path}")
            continue
        annotations = json.loads(labels_path.read_text(encoding="utf-8"))["annotations"]
        important = [a for a in annotations if norm_label(a.get("label", "")) in POSITIVE_LABELS]
        for half in (1, 2):
            transcript_path = args.transcripts / f"{match_id}_{half}_asr_en.json"
            if not transcript_path.is_file():
                print(f"Skipping {match_id} half {half}: missing {transcript_path}")
                continue
            segments = load_segments(transcript_path, half)
            if not segments:
                continue
            duration = max(end for _, end, _ in segments)
            events = []
            for annotation in important:
                try:
                    annotated_half, position_ms = int(annotation["gameTime"].split(" - ")[0]), float(annotation["position"])
                except (KeyError, ValueError):
                    continue
                if annotated_half != half:
                    continue
                seconds = position_ms / 1000.0
                text = context_at(segments, seconds)
                if text:
                    events.append((seconds, norm_label(annotation["label"]), text))
            for seconds, event_type, text in events:
                rows.append({"match_id": match_id, "half": half, "anchor_seconds": round(seconds, 2),
                             "english_commentary_context": text, "human_label": "highlight",
                             "event_type": event_type, "source": "SoccerNet event timestamp"})

            # Sample routine speech far from annotated key events, keeping the
            # same match-level split boundary for a clean evaluation.
            event_times = [t for t, _, _ in events]
            t = 30.0
            while t < duration - 15:
                if all(abs(t - event_time) > 45.0 for event_time in event_times):
                    text = context_at(segments, t)
                    if text:
                        rows.append({"match_id": match_id, "half": half, "anchor_seconds": round(t, 2),
                                     "english_commentary_context": text, "human_label": "routine",
                                     "event_type": "routine_sample", "source": "sampled away from key events"})
                t += args.negative_every

    if not rows:
        parser.error("No examples were created; check the dataset and transcript paths.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    positives = sum(row["human_label"] == "highlight" for row in rows)
    matches = sorted({str(row["match_id"]) for row in rows})
    print(f"Wrote {len(rows)} aligned windows from {len(matches)} matches to {args.out}")
    print(f"Event-positive windows: {positives}; routine samples: {len(rows) - positives}")
    print("These are weak labels for event detection; they are not human judgments of highlight quality.")


if __name__ == "__main__":
    main()
