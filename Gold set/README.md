# Gold set

Hand-checked lines to score the pipeline against. **Nothing here is read by `run_all.py`, `train.py` or any
other pipeline script**, so adding it does not change a running or finished training.

| file | what it holds |
|---|---|
| `gold_lines.csv` | 351 live lines from 4 matches (93323, 93336, 93341, 93343): `gold_class`, `gold_excitement` (0..1), `gold_player`. 229 lines were annotated one by one (`explicit=yes`); the other 122 are quiet stretches defaulted to `routine / 0.1` (`explicit=default`) and are only used for the excitement curve. |
| `gold_coref.csv` | 96 sampled pronoun / role mentions with the correct entity (`gold_entity`), `verdict` correct / wrong. |
| `evaluate_gold.py` | Scores saved outputs against both files. Trains nothing. |

## Status: DRAFT
Annotator column says `claude-draft`; `reviewed` is empty. Correct what you disagree with, put `yes` in
`reviewed` for lines you checked, and rerun. In the report, call it "gold annotated by Claude and reviewed by
<you>" only for what you actually reviewed.

## Annotation rule (so edits stay consistent)
* `gold_class`: routine / foul / card / near_miss / goal. A disallowed goal is `near_miss` (high excitement).
* `gold_excitement`: first line of a goal 1.0, later lines of the same goal 0.5-0.9, red card ~0.9, big saves /
  woodwork 0.6-0.8, ordinary shots 0.3-0.55, fouls / yellow cards 0.3-0.45, routine 0.05-0.3.
* `gold_player`: who the line is about (scorer, shooter, carded or fouling player). Blank when unclear.
  Matching is lenient on names ("Bowen" = "Jarrod Bowen").

## Run (after the pipeline has finished)
    python "Gold set/evaluate_gold.py"              # uses outputs/model/best.pt if it exists
    python "Gold set/evaluate_gold.py" --no-model   # only Stage 4, weak labels and coreference
Writes `outputs/gold_eval/report.json`.

## Reading the numbers
* Only matches in the model's **test** split give an honest model score. The 4 gold matches were picked from the
  first matches available, so some may fall in train/val; the report says which. Add gold lines for test matches
  (ids are in `report.json` as `model_test_ids`) for a clean number.
* Heads A/B are trained on labels derived from Stage 4, so "model vs gold" also shows how much of Stage 4's
  error the model inherits.
