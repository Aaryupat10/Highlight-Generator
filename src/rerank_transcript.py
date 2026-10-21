#!/usr/bin/env python3
"""Reapply current highlight rules to an existing timestamped transcript CSV."""

import argparse
import csv
from pathlib import Path

from pipeline import Segment, rank_segments, write_outputs
from entity_resolution import load_entity_aliases


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a saved transcript without rerunning Whisper or translation.")
    parser.add_argument("transcript_csv", type=Path, help="Existing transcript.csv from the pipeline")
    parser.add_argument("--video-name", default="source video", help="Name to show in the results")
    parser.add_argument("--out", type=Path, help="Output folder; defaults to the transcript CSV folder")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--merge-gap", type=float, default=12.0)
    parser.add_argument("--classifier-model", type=Path, help="Optional trained event/highlight classifier")
    parser.add_argument("--sequence-model", type=Path, help="Optional one-tier match-half sequence model")
    parser.add_argument("--hierarchical-model", type=Path, help="Optional two-tier hierarchical model")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", help="Set translate when the saved text is Whisper's direct English translation")
    parser.add_argument("--entity-aliases", type=Path, help="Optional verified JSON roster aliases; see data/entity_aliases.example.json")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.classifier_model, args.sequence_model, args.hierarchical_model)) > 1:
        parser.error("Use only one model checkpoint")

    source = args.transcript_csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Transcript CSV does not exist: {source}")
    if args.entity_aliases and not args.entity_aliases.expanduser().is_file():
        parser.error(f"Entity alias file does not exist: {args.entity_aliases}")
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
    sequence_model = None
    sequence_predictions = None
    model_path = args.sequence_model or args.hierarchical_model
    if model_path:
        if args.hierarchical_model:
            from hierarchical_model import HierarchicalMatchModel
            sequence_model = HierarchicalMatchModel.load(model_path.expanduser())
        else:
            from sequence_model import MatchHalfSequenceModel
            sequence_model = MatchHalfSequenceModel.load(model_path.expanduser())
        ordered = sorted(range(len(segments)), key=lambda i: segments[i].start)
        texts = [" ".join(s.text_english or s.text for s in segments if s.end >= segments[i].start - 15 and s.start <= segments[i].end + 15) for i in ordered]
        predictions = sequence_model.predict(texts, [segments[i].start for i in ordered])
        sequence_predictions = {key: [None] * len(segments) for key in predictions}
        for key, values in predictions.items():
            for model_index, original_index in enumerate(ordered):
                sequence_predictions[key][original_index] = values[model_index]
    entity_aliases = load_entity_aliases(args.entity_aliases.expanduser() if args.entity_aliases else None, Path(args.video_name))
    highlights = rank_segments(segments, args.min_score, args.merge_gap, classifier, sequence_predictions, entity_aliases)
    out = args.out.expanduser().resolve() if args.out else source.parent
    write_outputs(out, segments, highlights, Path(args.video_name), "saved transcript", classifier, args.task, sequence_predictions, sequence_model)
    print(f"Re-ranked {len(segments)} timestamped segments; candidates: {len(highlights)}")
    print(f"Updated results: {out}")


if __name__ == "__main__":
    main()
