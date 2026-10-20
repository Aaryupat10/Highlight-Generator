#!/usr/bin/env python3
"""Reapply current highlight rules to an existing timestamped transcript CSV."""

import argparse
import csv
from pathlib import Path

from pipeline import Segment, rank_segments, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a saved transcript without rerunning Whisper or translation.")
    parser.add_argument("transcript_csv", type=Path, help="Existing transcript.csv from the pipeline")
    parser.add_argument("--video-name", default="source video", help="Name to show in the results")
    parser.add_argument("--out", type=Path, help="Output folder; defaults to the transcript CSV folder")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--merge-gap", type=float, default=12.0)
    parser.add_argument("--classifier-model", type=Path, help="Optional trained event/highlight classifier")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", help="Set translate when the saved text is Whisper's direct English translation")
    args = parser.parse_args()

    source = args.transcript_csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Transcript CSV does not exist: {source}")
    segments = []
    with source.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            segments.append(Segment(
                float(row["start_seconds"]),
                float(row["end_seconds"]),
                row.get("text", ""),
                text_english=row.get("text_english", "") or (row.get("text", "") if args.task == "translate" else ""),
            ))

    classifier = None
    if args.classifier_model:
        from text_classifier import CommentaryClassifier
        classifier = CommentaryClassifier.load(args.classifier_model.expanduser())
    highlights = rank_segments(segments, args.min_score, args.merge_gap, classifier)
    out = args.out.expanduser().resolve() if args.out else source.parent
    write_outputs(out, segments, highlights, Path(args.video_name), "saved transcript", classifier, args.task)
    print(f"Re-ranked {len(segments)} timestamped segments; candidates: {len(highlights)}")
    print(f"Updated results: {out}")


if __name__ == "__main__":
    main()
