# Highlight Generator — preprocessing branch
This branch contains the commentary preprocessing stage only.

## English commentary preprocessing

The preprocessing step splits commentary at existing line breaks and sentence-ending full stops. It writes one CSV containing `match_id`, zero-based `line_no`, `raw_text`, `clean_text`, `n_exclaim`, `caps_ratio`, and `n_caps_words`. Clean text fixes extra whitespace and expands the football abbreviation `pen` to `penalty`; it preserves capitalization and punctuation. The three excitement measurements are in the same row as their commentary. No event classification is added.

The input CSV needs a commentary column (`events`, `english_commentary_context`, `english_commentary`, `commentary`, or `text`). The script uses `match_id`, or `id` if that is the available identifier. The event label is not included in this preprocessing output.

```bash
PYTHONPATH=src python3 src/prepare_event_text.py data/your_dataset.csv --out data/commentary_clean.csv
```
