# Highlight Generator
Detecting and ranking football highlights from live text commentary with deep NLP and hierarchical DL.

## English commentary preprocessing

`src/nlp_analysis.py` is the project-owned preprocessing module. It normalizes Unicode and whitespace while retaining capitalization and emphasis, then reports lexical event evidence and English negation, hypothetical, retrospective, live-action, and excitement cues. It uses transparent rules; it is not a trained language model and does not guarantee correct interpretation of every commentary sentence.

Run its focused checks with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
