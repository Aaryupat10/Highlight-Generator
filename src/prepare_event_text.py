#!/usr/bin/env python3
"""Split commentary blocks into sentence rows and preserve punctuation/case."""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

TEXT_COLUMNS = ("events", "english_commentary_context", "english_commentary", "commentary", "text")
OUTPUT_FIELDS = ("match_id", "line_no", "raw_text", "clean_text", "n_exclaim", "caps_ratio", "n_caps_words")
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z][A-Z0-9]+\b")


def clean_line(text: str) -> str:
    """Clean spacing/Unicode and expand the football abbreviation 'pen'."""
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    def expand_pen(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.isupper():
            return "PENALTY"
        return "Penalty" if word[:1].isupper() else "penalty"

    return re.sub(r"\bpen\b", expand_pen, cleaned, flags=re.IGNORECASE)


def excitement_features(text: str) -> dict[str, str]:
    letters = [char for char in text if char.isalpha()]
    uppercase_count = sum(char.isupper() for char in letters)
    return {
        "n_exclaim": str(text.count("!")),
        "caps_ratio": f"{uppercase_count / len(letters):.2f}" if letters else "0.00",
        "n_caps_words": str(len(ALL_CAPS_WORD_RE.findall(text))),
    }


def prepare_rows(source: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    line_numbers: dict[str, int] = defaultdict(int)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_column = next((column for column in TEXT_COLUMNS if column in set(reader.fieldnames or [])), None)
        if text_column is None:
            raise ValueError(f"No commentary column found; expected one of {TEXT_COLUMNS}.")
        for source_row_number, row in enumerate(reader, start=1):
            # A generic `id` is often a dataset row key (the 23/24 CSV has
            # values such as 93323), not a match identifier. Never relabel it
            # as a match. Keep a traceable placeholder when no match_id exists.
            match_id = (row.get("match_id") or "").strip() or f"source_row_{source_row_number}"
            raw_block = row.get(text_column, "") or ""
            sentences = [
                sentence
                for line in raw_block.splitlines()
                for sentence in re.split(r"(?<=\.)\s+", line.strip())
                if clean_line(sentence)
            ]
            for raw_text in sentences:
                clean_text = clean_line(raw_text)
                line_no = line_numbers[match_id]
                line_numbers[match_id] += 1
                rows.append({
                    "match_id": match_id,
                    "line_no": str(line_no),
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    **excitement_features(clean_text),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing a raw commentary text column")
    parser.add_argument("--out", type=Path, default=Path("data/commentary_clean.csv"))
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    try:
        rows = prepare_rows(args.input_csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("No non-empty commentary lines were found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} commentary lines; case and punctuation preserved.")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
