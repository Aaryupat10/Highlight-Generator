"""Split football commentary at line breaks and full stops, then clean it."""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


TEXT_COLUMNS = ("events", "english_commentary_context", "english_commentary", "commentary", "text")
CLEAN_FIELDS = (
    "match_id", "line_no", "raw_text", "clean_text",
    "n_exclaim", "caps_ratio", "n_caps_words",
)
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z][A-Z0-9]+\b")


def clean_line(text: str) -> str:
    """Fix spacing and one football abbreviation; preserve case and punctuation."""
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    def expand_pen(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.isupper():
            return "PENALTY"
        if word[:1].isupper():
            return "Penalty"
        return "penalty"

    return re.sub(r"\bpen\b", expand_pen, cleaned, flags=re.IGNORECASE)


def excitement_features(text: str) -> dict[str, str]:
    letters = [char for char in text if char.isalpha()]
    uppercase_count = sum(char.isupper() for char in letters)
    caps_ratio = uppercase_count / len(letters) if letters else 0.0
    caps_words = ALL_CAPS_WORD_RE.findall(text)
    return {
        "n_exclaim": str(text.count("!")),
        "caps_ratio": f"{caps_ratio:.2f}",
        "n_caps_words": str(len(caps_words)),
    }


def _find_text_column(fieldnames: list[str] | None) -> str | None:
    available = set(fieldnames or [])
    return next((column for column in TEXT_COLUMNS if column in available), None)


def prepare_rows(source: Path) -> list[dict[str, str]]:
    """Split commentary at line breaks/full stops and prepare text and cues."""
    rows: list[dict[str, str]] = []
    line_numbers: dict[str, int] = defaultdict(int)

    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_column = _find_text_column(reader.fieldnames)
        if text_column is None:
            raise ValueError(f"No commentary column found; expected one of {TEXT_COLUMNS}.")

        for row in reader:
            match_id = (row.get("match_id") or row.get("id") or "").strip()
            raw_block = row.get(text_column, "") or ""
            match_lines = [
                sentence
                for line in raw_block.splitlines()
                for sentence in re.split(r"(?<=\.)\s+", line.strip())
                if clean_line(sentence)
            ]
            if not match_lines:
                # Keep a row for a match record even when the source has no commentary.
                rows.append({
                    "match_id": match_id,
                    "line_no": "0",
                    "raw_text": "",
                    "clean_text": "",
                    **excitement_features(""),
                })
                line_numbers[match_id] = max(line_numbers[match_id], 1)
                continue

            for raw_text in match_lines:
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


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing a commentary text column")
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

    _write_csv(args.out, CLEAN_FIELDS, rows)
    print(f"Prepared {len(rows)} commentary lines.")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
