# Commentary2Highlight: pipeline status and next work

Updated 21 October 2026. This status is based on the current project branch and checked files, not on the teammate's proposed pipeline being assumed complete.

## What the current project actually does

The working path is: timestamped transcript or video with usable commentary audio → timestamped ASR (video route) → optional English translation → case- and punctuation-preserving whitespace preprocessing → transparent event/discourse cues → project-fitted TF-IDF → project-trained PyTorch ranking model plus football phrase rules → nearby candidate merging and peak timestamp → CSV, JSON, and Markdown.

Whisper and optional NLLB are pretrained components. The TF-IDF representation, MLP, event rules, feature extraction, sequence model, and evaluation scripts are project code. The available ASR/translations and the learned scorer have not been proven reliable for every language, match, or commentator.

## Ten-stage comparison

| # | Stage | Current status | What is true now | Remaining work |
|---|---|---|---|---|
| 1 | Data | Partial, usable | Six EPL matches have SoccerNet-Echoes English transcripts aligned with SoccerNet event labels. A small number of local MKVs have working audio; two tested City–Newcastle halves decode to empty audio. | Verify each intended video/audio input, preserve source and timestamp provenance, and expand distinct-match coverage. The 2023–24 match-details CSV has unreliable row-to-match metadata and no usable timestamps; it is excluded from model training. |
| 2 | Preprocessing | Working baseline | Unicode/zero-width cleanup and whitespace normalization retain capitalization and punctuation. `src/prepare_event_text.py` splits raw CSV commentary at line breaks/full stops; `src/pipeline.py` keeps ASR timestamp segments intact. Both emit exclamation/capitalization cues. | Review segmentation quality on real transcripts. No stemming or lowercasing is applied to displayed text. |
| 3 | Coreference | Working conservative heuristic | Verified alias lookup is exposed through `--entity-aliases`. Pronouns resolve only to one unique player alias in the nearest prior transcript segment; ambiguous cases explicitly abstain. | Player aliases must be verified per match. A trained/evaluated coreference model still needs labeled antecedent examples. |
| 4 | Semantic roles | Working project-written frame extractor | Candidate output includes action, a unique verified player actor when explicit, team candidates, outcome cues, and an unresolved target field. The Markdown/JSON reports the frame and labels its method. | This is a compact event-frame grammar, not a trained general SRL model. Add role-labeled examples before claiming broad role accuracy or training a role tagger. |
| 5 | Embeddings | Working baseline | A project-fitted TF-IDF unigram/bigram vocabulary and IDF are trained on project examples; no pretrained text embedding is used by this classifier. | Compare against a documented linear baseline and inspect errors; current training data is small and English. |
| 6 | Hierarchical model | Implemented, experimental | `src/hierarchical_model.py` now runs local window-level BiGRU/attention inside five-minute blocks, then a second match-half BiGRU/attention tier. It returns per-window heads and temporal attention. | It is trained on sparse timestamp windows, not full continuous commentary; compare held-out results with the simpler MLP before choosing it. |
| 7 | Multi-task heads | Implemented, experimental | New checkpoint separates weak event relatedness, weak event family, and human highlight value instead of treating those targets as equivalent. | Held-out results are poor, especially event-family goal/card recall and human highlight quality. More independent human labels and retraining are required. No player or excitement prediction head is trained. |
| 8 | Attention analysis | Working diagnostic | Sequence outputs include top attended timestamps and attention mass on the 30-second pre-candidate buildup; TF-IDF reports feature ablations. Reports caution that attention is not causal evidence. | Compare attention/buildup diagnostics against independently reviewed examples before interpreting them. |
| 9 | Peak detection | Working temporal grouping | Nearby candidates merge by configurable gap. Output reports the max-score peak, candidate density, buildup count/mean score, and peak offset in the group. | Validate event timing and group boundaries against more independently reviewed matches; the selector still depends on candidate detection. |
| 10 | Output | Working | Transcript and ranked commentary are exported as CSV, JSON, and Markdown with timestamps, event family, evidence, and separate score components when the new model is used. | Validate outputs on unseen reviewed matches. Video clipping is a future extension and does not run in the current pipeline. |

## Data and evaluation separation

Current training inventory:

- Weak event task: 365 timestamped windows across six matches. These labels mean SoccerNet annotated an event near that time; they do not mean a person judged the commentary to be a highlight.
- Human highlight task: 75 labeled windows across seven matches (60 original review rows plus 15 Crystal Palace review rows), 30 highlight and 45 routine. These are the only rows used for the human-highlight head.
- There are 15 exact time/half overlaps between human-reviewed rows and weak event rows where the judgments conflict. Separate heads prevent those conflicting meanings from being silently collapsed.
- The provided 2023–24 CSV has a concrete metadata mismatch: its first row is keyed as ID `93323` with Bournemouth–West Ham metadata, while its event narrative opens with Burnley–Manchester City at Turf Moor. The preprocessing-only output therefore uses `source_row_1` instead of treating `93323` as a match ID.
- Model selection and metrics use whole-match-held-out folds with TF-IDF fitted inside each fold. The final checkpoint is then refit on all available rows and is for inference/demo, not for reporting held-out performance.

## Assistant annotation and retraining experiment (21 October 2026)

Codex completed a separate text-only draft in `data/assistant_text_annotations.csv` using the English commentary windows. It has 120 rows: 33 labeled highlight, 83 routine, and 4 unclear. It also fills event type, text-derived excitement, and only clearly inferable player/role fields. Every row is marked `assistant_text_only`; none was checked against the broadcast audio/video. The teammate queue `data/human_review_batch_01.csv` remains blank and unchanged.

The trainers accept these pseudo-labels only when explicitly requested with `--assistant-labels`; they are not used by default and are omitted from human-only held-out metrics. A fair same-epoch comparison showed that adding assistant labels reduced positive highlight F1 on the existing 75 human-reviewed examples:

| Model / training rows | Human-only held-out positive F1 | Precision | Recall |
|---|---:|---:|---:|
| TF-IDF MLP, human labels only, 120 epochs | 0.296 | 0.333 | 0.267 |
| TF-IDF MLP, plus assistant text labels, 120 epochs | 0.121 | 0.667 | 0.067 |
| Hierarchical BiGRU, human labels only, 10 epochs | 0.444 | 0.424 | 0.467 |
| Hierarchical BiGRU, plus assistant text labels, 10 epochs | 0.190 | 0.333 | 0.133 |

Therefore do not select the assistant-label checkpoints as the improved/default model. Their checkpoints and metrics are retained as an experiment. The best of these controlled comparisons is the human-only hierarchical checkpoint, but it is still experimental, trained/evaluated on a small set, and not evidence of broad match/language performance. The text-derived excitement/player/action fields in the assistant CSV are not yet separate trained model heads.

Latest human-only `src/train_pipeline_model.py --epochs 160` aggregate held-out results at the fixed 0.50 threshold (retrained 21 October):

| Target | N | Accuracy | F1 / macro F1 | Reading |
|---|---:|---:|---:|---|
| Weak event-relatedness | 365 | 0.584 | positive F1 0.596; macro F1 0.583 | Modest exploratory performance against weak timestamp alignment. |
| Human highlight | 75 | 0.520 | positive F1 0.419; macro F1 0.505 | Too weak and too small to claim a reliable highlight detector. |
| Event family | 195 | 0.774 | macro F1 0.343 | Accuracy hides rare-class failure: goal F1 0.000, card F1 0.154, shot/chance F1 0.874. |

The two-tier experimental model was retrained human-only with `src/train_hierarchical_model.py --epochs 20`. Whole-match-held-out aggregates: event relatedness F1 0.521 (macro F1 0.568); human highlight positive F1 0.400 (macro F1 0.526); event family macro F1 0.519. Goal F1 was 0.162, card F1 0.522, and shot/chance F1 0.873. This improves rare-family recognition relative to the MLP but does not establish a dependable highlight selector; per-match human results vary substantially. The MLP remains the default ranker because its highlight F1 is slightly higher; both are experimental.

Do not advertise these as universal or production-grade results. Threshold selection was fixed before evaluation; no calibration is claimed. Test folds are small and the data covers one competition/season and English translated commentary. Translation errors can change names and event wording.

## What to do next

1. Have teammates label the blinded queue in `data/human_review_batch_01.csv` (40 rows per member, spread across all 12 match-halves); use footage when available. Use `highlight`, `routine`, or `unclear`, and record actual event type separately. Do not expose or infer the hidden weak source label while judging.
2. Merge only completed human labels into training. Keep SoccerNet event-alignment rows in the weak-event file. Preserve match IDs and keep complete matches out of test folds.
3. Review false positives/negatives by match, especially goal and card examples. Add examples from several matches, languages after English translation, and ordinary commentary that mentions goals/shots retrospectively or hypothetically.
4. Build and verify per-match team/player alias lists. Until then, show player identity as “unresolved” rather than guessing. Add human labels for player mentions, pronoun links, event roles, and textual excitement before training heads for them.
5. Re-run match-held-out training and compare against rule-only and linear TF-IDF baselines. Only replace the demo checkpoint if the human-label metric improves across held-out matches and goal/card errors are addressed.
6. Once the text ranking is stable, test timestamp tolerance against manually verified video moments, then add optional FFmpeg clip creation with configurable padding.

## Reproduce the current project-trained model

From the repository root, with the project's Python environment and PyTorch installed:

```bash
python3 src/train_pipeline_model.py --epochs 160
```

This writes `models/commentary_pipeline_multitask.pt` and `models/commentary_pipeline_multitask.metrics.json`. The model uses the 8 GB Mac-friendly sparse TF-IDF → small MLP path; it does not fine-tune Whisper or NLLB.

Train the two-tier hierarchy and its match-held-out diagnostics with:

```bash
python3 src/train_hierarchical_model.py --epochs 20
```

This writes `models/commentary_hierarchical.pt` and its metrics JSON. To rerank an existing timestamped transcript use `src/rerank_transcript.py --hierarchical-model models/commentary_hierarchical.pt`; for fresh video audio use the same option with `src/pipeline.py`.

Create another blinded annotation batch with:

```bash
python3 src/create_annotation_queue.py --per-class 5
```

The queue balances weak event and routine windows for sampling only. Its source labels are omitted. Reviewers make the human judgments.

Prepare a raw commentary CSV into one row per sentence while preserving uppercase and punctuation with:

```bash
python3 src/prepare_event_text.py "/path/to/commentary.csv" --out data/commentary_clean.csv
```

This is text preprocessing only; it does not create timestamps or ground-truth highlights.
The supplied 2023–24 CSV was processed into `outputs/preprocessing_23_24/commentary_clean.csv` (47,625 sentence rows). It records 7,256 rows with exclamation marks and 6,566 with at least one uppercase token. The text is useful for a preprocessing demonstration and possible future manual annotation, but has no reliable sentence-level video timestamps or labels and is not used to train/evaluate the current event model.

## Recent operational smoke check

The current video pipeline was previously run end-to-end on the first three minutes of Crystal Palace–Manchester City using Faster-Whisper `small` with English translation. After the latest human-only retrain, the saved 81-segment transcript was reranked with the two-tier model and produced two timestamped candidates. A separate 765-segment Liverpool–Aston Villa saved transcript also reranked successfully and produced seven candidates. These are operational smoke checks, not quality tests. The family head still disagrees with phrase-rule labels on some candidates (for example, goal for a shot phrase and card for a corner); review those predictions before presenting. Output reporting now maps `shot_or_chance` to `chance_or_shot` for agreement checks and identifies sequence-model runs as model-ranked.

## Files and data safety

Video/audio extensions are ignored by Git through `.gitignore`; keep NDA-restricted match media out of commits. `data/23_24_match_details_reconciled_review.csv` is a review aid, not ground truth, and must not be used as timestamped training labels. Existing transcript files may be redistributed only if their dataset terms allow it.
