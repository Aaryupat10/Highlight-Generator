import csv
import tempfile
import unittest
from pathlib import Path

from prepare_event_text import OUTPUT_FIELDS, prepare_rows


class PrepareEventTextTests(unittest.TestCase):
    def test_keeps_clean_text_label_and_split_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "match_id", "half", "anchor_seconds", "english_commentary_context",
                    "event_type", "source", "notes",
                ])
                writer.writeheader()
                writer.writerow({
                    "match_id": "match_a", "half": "1", "anchor_seconds": "12.5",
                    "english_commentary_context": " GOAL!!!  Smith’s shot ",
                    "event_type": "Shots on Target", "source": "private source", "notes": "discard",
                })
                writer.writerow({
                    "match_id": "match_a", "half": "1", "anchor_seconds": "15",
                    "english_commentary_context": "No label", "event_type": "", "source": "annotation",
                })
                writer.writerow({
                    "match_id": "match_a", "half": "1", "anchor_seconds": "18",
                    "english_commentary_context": "   ", "event_type": "goal", "source": "annotation",
                })
                writer.writerow({
                    "match_id": "match_a", "half": "1", "anchor_seconds": "20",
                    "english_commentary_context": "Routine commentary", "event_type": "not selected by model",
                })

            rows, skipped = prepare_rows(source)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], {
            "commentary": "GOAL!!! Smith’s shot",
            "event": "shots_on_target",
            "event_context": "asserted_or_live",
            "has_excitement_cue": "true",
            "tokens_json": '["GOAL", "!", "!", "!", "Smith’s", "shot"]',
            "punctuation_counts_json": '{"!": 3, "’": 1}',
            "exclamation_count": "3",
            "uppercase_word_count": "1",
            "match_id": "match_a",
            "half": "1",
            "anchor_seconds": "12.5",
        })
        self.assertEqual(set(rows[0]), set(OUTPUT_FIELDS))
        self.assertEqual(skipped["empty_event"], 1)
        self.assertEqual(skipped["empty_commentary"], 1)
        self.assertEqual(skipped["invalid_event_label"], 1)

    def test_normalizes_label_spelling_and_routine_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text(
                "commentary,event_type\nA save,routine_sample\nA booking,yellow card\n",
                encoding="utf-8",
            )
            rows, _ = prepare_rows(source)
        self.assertEqual([row["event"] for row in rows], ["routine", "yellow_card"])

    def test_accepts_short_commentary_and_event_column_names(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("commentary,event\nA goal for City,goal\n", encoding="utf-8")
            rows, _ = prepare_rows(source)
        self.assertEqual(rows[0]["commentary"], "A goal for City")
        self.assertEqual(rows[0]["event"], "goal")
        self.assertEqual(rows[0]["event_context"], "asserted_or_live")


if __name__ == "__main__":
    unittest.main()
