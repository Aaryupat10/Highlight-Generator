# Pipeline and 2023–24 match-details review

Reviewed 19 October 2026 against the teammate's three-page pipeline example, the supplied 2023–24 CSV, the current project code, current training data, and the files visible in `path/to/SoccerNet`. The implementation notes below include the pipeline changes made on this date.

## Decision on the supplied CSV

The CSV has 303 rows and nine fields: `id`, `Home`, `Away`, `Date`, `Stadium`, `Attendance`, `Referee`, `events`, and `summary`. It contains about 5.24 million characters in `events`, but those are untimestamped text blocks, not audio transcripts with segment times. The date range is 2023-08-12 through 2024-04-15. It has 12 empty `events` cells, 5 empty `summary` cells, and 8 empty attendance cells.

The row-to-commentary match mapping fails an essential integrity check. A scan of the opening lines found two teams for 291 rows; **zero** of those pairs match that row's `Home` and `Away` fields. For example, ID 93323 is labelled Bournemouth–West Ham but its event text opens with Burnley–Manchester City. A second, stricter check matched the opening team pair plus exact home-stadium name to a unique fixture for **189** blocks. Those are in `data/23_24_match_details_reconciled_review.csv`, with source-row provenance and the matching evidence retained. This is an inferred review copy, not ground truth. The other 114 rows were excluded because the commentary was empty, the pair was absent, or the venue could not uniquely confirm a fixture. Review the inferred associations before using their text or rosters. Do not infer timestamps or labels from them. Audit summaries are in `23_24_match_details_audit.json` and `23_24_match_details_reconciliation.json`; reproduce both checks with:

```bash
python3 src/audit_match_details.py "path/to/23_24_match_details.csv" --out outputs/23_24_match_details_audit.json
python3 src/reconcile_match_details.py "path/to/23_24_match_details.csv"
```

Only 7 rows contain any `mm:ss`-shaped text and 2 have a minute marker, so the file does not provide reliable per-line timing. **Do not use the original row IDs, claim timestamp extraction, or train/evaluate the current highlight selector from these records yet.** The 189-row review copy's inferred IDs need confirmation. The supplied file is preserved unchanged.

After checking the 189 inferred associations, this text can help build fixture-specific English team/player dictionaries and, after line-level annotation, expand event/routine training examples. It cannot teach Whisper or provide video timestamps. The currently visible SoccerNet folder has five MKVs and no 2023–24 season directory; none of those five is a 2023–24 video matched to this CSV.

## What the teammate's stages mean in this project

| Teammate stage | Current status | Practical project version |
|---|---|---|
| 0. Whisper demo input | Implemented using pretrained faster-whisper | Extract decodable audio, transcribe with segment timestamps and detected language. For non-English, use Whisper's English translation or NLLB after ASR; preserve original text. These are input tools, not models trained by the team. |
| 1. Collect and split | Partly implemented | The video pipeline writes Whisper segments. The SoccerNet-Echoes JSON files provide English timestamped segments for training alignment. The new CSV is not safe input until its match metadata is repaired; split lines only after data integrity passes. |
| 2. Preprocessing | Implemented in a small, task-specific way | `src/nlp_analysis.py` normalizes whitespace and reports negation, hypothetical, retrospective, live-action and excitement cues. The model tokenizes text, creates word and bigram features, retains negation, and adds punctuation/emphasis cues. Keep original ASR text for audit. |
| 3. Coreference | Conservative alias-based heuristic implemented | `src/entity_resolution.py` links a pronoun only when exactly one verified player alias appears in the preceding 30 seconds. Otherwise it reports unresolved. Full trained coreference is not implemented. |
| 4. SRL / event extraction | Lexical event-frame approximation implemented, not full SRL | Rules detect event type, phrase evidence, outcome cues, and optional actor/team candidates from a verified alias list. They do not reliably extract assist, score state, or full semantic roles. |
| 5. Line embeddings | Implemented as project-fitted TF-IDF | `TfidfEncoder` learns the vocabulary and IDF from training text; the MLP is initialized and trained by the project. This is a sparse feature vector, not a 768-dimensional pretrained Transformer embedding. |
| 6. Hierarchical sequence model | Match-half sequence model implemented experimentally | `src/train_sequence_model.py` trains a project-built BiGRU over ordered commentary windows with timestamps/gaps. It provides local temporal context but is not a full two-level match/half hierarchy. Leave-one-match-out eventness macro-F1 was 0.597 (accuracy 0.600); the existing TF-IDF MLP remains the stronger model in prior evaluation. |
| 7. Multi-task heads | Partly implemented | The sequence and TF-IDF models predict eventness and broad event family (`goal`, `shot_or_chance`, `card`). Event-family targets are weak SoccerNet labels and sparse; the TF-IDF head had low macro-F1. No trained excitement or player-name head exists. |
| 8. Attention heatmap | Temporal attention implemented; feature inspection retained | The sequence model exports its learned attention to nearby transcript windows for a candidate. These are model context weights, not human explanations. TF-IDF feature ablations are also available for the non-sequence model. |
| 9. Peak detection / grouping | Basic version implemented | Nearby candidates are grouped; output reports the strongest segment as a peak time, candidate count and score density. This is rule-based temporal grouping, not a learned continuous excitement curve or validated peak detector. |
| 10. Structured output | Implemented and expanded | The system exports transcript cues and ranked event records, evidence, component scores and explanations in CSV/JSON/Markdown. Timestamps originate from ASR and remain estimates. Video cutting is a future extension, not required to demonstrate the current goal. |

## The smallest coherent, team-owned model

Keep the present demonstrable pipeline and improve it in this order:

1. **Input and provenance:** verify that each video has decodable commentary audio; save source file, match, half, language and transcription/translation route with each result. Preserve original-language and English text separately. Review the 189-row reassociation copy before using any of its match metadata.
2. **Text preparation:** normalize case and whitespace but preserve negation, names, event punctuation and timestamps. Apply a match-specific team/player alias table only when its fixture has been verified.
3. **NLP candidates:** use the team's football phrase patterns for candidate proposal. Separate live events from past-match recaps, statistics, negated/hypothetical mentions and routine corner/shot mentions. Keep rule rationale in output.
4. **Team-trained DL:** vectorize the local commentary window with project-fitted unigram/bigram TF-IDF and score it using the random-initialized PyTorch MLP. Compare against the transparent rule-only and TF-IDF linear baselines. The current MLP reranks candidates; it cannot recover an event the candidate rules missed.
5. **Time and output:** merge overlapping/neighbouring candidate windows, retain the strongest timestamp, and export commentary plus start/end times in CSV/JSON/Markdown. Check every demo timestamp in the source video. Add FFmpeg clip cutting only after timing checks pass.
6. **Evaluation:** separate weak SoccerNet event-alignment results from human highlight judgments. Keep whole matches out of training folds. For manual review, label candidates and randomly sampled routine controls, resolve disagreements, and measure precision/recall/F1 at a threshold selected without the final test match. For localization, report whether a prediction falls within an explicit time tolerance of an annotated event. Report ASR/translation errors separately.

The live 2023–24 narrative CSV is **not yet part of this model's training**. The current training file remains 380 mixed examples from seven matches: 365 weak event/routine examples and 15 manually reviewed Crystal Palace examples. The previously reported TF-IDF MLP macro F1 of 0.655 is exploratory, uses mixed labels and threshold 0.5, and does not show that this CSV or a universal multi-language model works.

## What to do next

1. Complete the 14-row Crystal Palace held-out review already prepared in `data/crystal_palace_holdout_review.csv`; do not peek at its key before labels are finished.
2. Review the 189-row reassociation copy against the source text before using it to build verified team/player aliases.
3. Add another independently reviewed match and routine controls before retraining; compare the team's TF-IDF MLP to rules and a linear TF-IDF baseline on an untouched whole match.
4. Collect player/team entity labels, coreference decisions, and event-role fields before training those heads. The current data cannot support honest training of them.
5. Improve the sequence model only if more reviewed matches support it; it currently scores below the TF-IDF MLP. Excitement/player heads, a full two-level hierarchy, universal multilingual claims and video cutting need more aligned labels and review.

## Honest project description for the report

“We use pretrained Whisper and optional machine translation only to create timestamped English commentary from match audio. Our project-built downstream NLP/DL pipeline normalizes transcript text, extracts lexical event evidence and discourse cues, represents commentary with a project-fitted TF-IDF vocabulary, and uses a PyTorch MLP trained from weakly aligned and human-reviewed examples to rerank event candidates. Nearby candidate timestamps are grouped, and output includes component scores and local feature-ablation evidence. Current labels are limited and partly weak; coreference, reliable semantic roles, player identification and trained excitement heads remain future work. We verify timestamps against video and do not claim universal accuracy across matches or languages.”
