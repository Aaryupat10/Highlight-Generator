"""Prepare a compact commentary/event dataset for project-owned NLP models."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from nlp_analysis import normalize_commentary


OUTPUT_FIELDS = ("commentary", "event", "match_id", "half", "anchor_seconds")
TEXT_COLUMNS = ("english_commentary_context", "english_commentary", "commentary", "text")
EVENT_COLUMNS = ("event_type", "event")
NON_EVENT_LABELS = {"not selected by model", "unclear", "unknown", "none", "nan"}


def _canonical_event(value: str) -> str:
    label = normalize_commentary(value).casefold()
    if label == "routine_sample":
        return "routine"
    return re.sub(r"[\s-]+", "_", label)


def _first_column(fieldnames: list[str] | None, candidates: tuple[str, ...]) -> str | None:
    available = set(fieldnames or [])
    return next((name for name in candidates if name in available), None)


def prepare_rows(source: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    """Keep cleaned commentary, event label, and identifiers for honest splits."""
    prepared: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_column = _first_column(reader.fieldnames, TEXT_COLUMNS)
        event_column = _first_column(reader.fieldnames, EVENT_COLUMNS)
        if text_column is None:
            raise ValueError(f"No commentary text column found; expected one of {TEXT_COLUMNS}.")
        if event_column is None:
            raise ValueError(f"No event label column found; expected one of {EVENT_COLUMNS}.")

        for row in reader:
            commentary = normalize_commentary(row.get(text_column, ""))
            raw_event = normalize_commentary(row.get(event_column, ""))
            event = _canonical_event(raw_event)
            if not commentary:
                skipped["empty_commentary"] += 1
                continue
            if not event:
                skipped["empty_event"] += 1
                continue
            if raw_event.casefold() in NON_EVENT_LABELS:
                skipped["invalid_event_label"] += 1
                continue
            prepared.append({
                "commentary": commentary,
                "event": event,
                # These are metadata for grouped evaluation and timestamp alignment;
                # the text model must not use them as predictive features.
                "match_id": (row.get("match_id") or "").strip(),
                "half": (row.get("half") or "").strip(),
                "anchor_seconds": (row.get("anchor_seconds") or row.get("start_seconds") or "").strip(),
            })
    return prepared, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Dataset CSV containing commentary and event/event_type columns")
    parser.add_argument("--out", type=Path, default=Path("data/preprocessed_event_text.csv"))
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")

    try:
        rows, skipped = prepare_rows(args.input_csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("No rows had both non-empty commentary and an event label.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["event"] for row in rows)
    print(f"Prepared {len(rows)} labeled commentary rows; skipped {sum(skipped.values())} rows.")
    print(f"Event labels: {dict(sorted(counts.items()))}")
    print(f"Skipped rows: {dict(sorted(skipped.items())) or 'none'}")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
