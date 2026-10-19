# Highlight Generator
Detecting and ranking football highlights from live text commentary with deep NLP and hierarchical DL.

## English commentary preprocessing

`src/nlp_analysis.py` is the project-owned preprocessing module. It normalizes Unicode and whitespace while retaining capitalization and emphasis, then reports lexical event evidence and English negation, hypothetical, retrospective, live-action, and excitement cues. It uses transparent rules; it is not a trained language model and does not guarantee correct interpretation of every commentary sentence.

To prepare a dataset for event classification, use commentary as the text input and the `event`/`event_type` column as its label. The cleaner drops unrelated annotation columns, standardizes label spelling, skips empty text/labels and placeholder labels such as `not selected by model`, and adds inspectable text cues: event context, excitement wording/punctuation, exclamation count, and uppercase-word count. These are textual proxies for emphasis, not proof of an actual vocal shout. It retains match, half, and timestamp only for grouped evaluation and alignment; those metadata must not be used as model features:

```bash
PYTHONPATH=src python3 src/prepare_event_text.py data/your_dataset.csv --out data/preprocessed_event_text.csv
```

Run its focused checks with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
