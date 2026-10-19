# Commentary2Highlight MVP

This is the first working-model milestone for the project: give it a football video with usable commentary audio, transcribe the speech with Whisper, rank likely important commentary moments, and export a timestamped highlight commentary list. It does not cut video clips yet.

The current local video inventory and audio checks are recorded in [`data/video_audio_inventory.md`](data/video_audio_inventory.md). Only three of the five currently visible MKVs have usable audio; the two Manchester City–Newcastle files have metadata audio streams but decode to empty WAV files.

The teammate's complete ten-stage proposal and the new 2023–24 match-details CSV have been audited in [`outputs/pipeline_and_dataset_review.md`](outputs/pipeline_and_dataset_review.md). Its row metadata is misaligned: 0 of 291 detectable opening team pairs match their own row. A conservative team-pair plus home-stadium check created a separate 189-row inferred review copy at `data/23_24_match_details_reconciled_review.csv`; it has no labels or reliable timestamps and is not used for training. Inspect it before relying on inferred match IDs. Reproduce the audit with `python3 src/audit_match_details.py "/path/to/23_24_match_details.csv" --out outputs/23_24_match_details_audit.json` and create the review copy with `python3 src/reconcile_match_details.py "/path/to/23_24_match_details.csv"`.

## What it does

1. Extracts mono 16 kHz audio from the supplied video using FFmpeg.
2. Runs the pretrained Whisper `base` model through `faster-whisper` on CPU with int8 quantization. Whisper supplies an estimated start and end time for each spoken segment.
3. Translates speech to English when requested, preserving original text and timestamps where the translation route supports it.
4. Applies project-written text normalization, event phrase detection, and transparent negation/hypothetical/retrospective/live-action/emphasis cues.
5. Ranks event candidates with the phrase rules and, optionally, the project-trained PyTorch classifier using surrounding commentary context.
6. Groups nearby candidate moments, reports the peak timestamp and component scores, and saves transcript and highlight commentary in CSV, JSON, and Markdown.

The NLP/DL pipeline uses pretrained Whisper for speech recognition, with either Whisper's English translation task or optional NLLB text translation. Project-written event phrase rules propose candidates; a project-trained PyTorch feed-forward network reranks them using surrounding commentary. The preferred TF-IDF model fits its vocabulary and IDF values on project data; the hashed-feature model remains available. Rule-only and linear baselines are retained. TF-IDF model explanations remove one active feature at a time and report which removals lower the candidate score; this is feature perturbation, not attention. Event evidence and discourse cues are exported for inspection. This is not a fine-tuned Transformer or the originally proposed hierarchical model. No trained coreference, semantic-role, player identity, or excitement head exists yet. Automatic language detection and translation have been tried on Spanish, Russian, and French examples. Translation can be inaccurate, so do not claim reliable performance on every language.

### English text preprocessing

`src/nlp_analysis.py` applies Unicode normalization, standardizes common apostrophes and dashes, removes zero-width characters, collapses whitespace, and keeps emphasis punctuation and capitalization. It adds event evidence and local negation, hypothetical, retrospective, live-action, and excitement cues. Candidate scoring suppresses event phrases that are clearly negated, hypothetical, or retrospective, while checking separate clauses so a later positive event is retained. These are transparent English-focused rules, not a guarantee of correct interpretation for every sentence. Original ASR and translated segment text remain separately available in pipeline output.

`src/text_classifier.py` uses the same normalization, case-folded unigram/bigram features, punctuation/emphasis indicators, and separate cue features. The TF-IDF vocabulary and IDF weights are fitted from the project training split. After changing these features, retrain and compare match-held-out metrics before replacing the deployed checkpoint.

## Requirements on the Mac

- Python 3.10–3.12 recommended.
- FFmpeg installed and available in Terminal. On macOS with Homebrew: `brew install ffmpeg`.
- First run needs internet access to download the selected Whisper model. Later runs can use its local cache.
- A video file that actually contains decodable commentary audio. The earlier project conversation reports that the SoccerNet samples tested had nearly empty audio payloads; validate any candidate video before using it.

From this folder, create an environment and install the Python package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python src/pipeline.py "/path/to/football-match.mp4" --out outputs/demo --model base --language en
```

On the 8 GB Mac, begin with `base`; use `tiny` if memory or runtime is a problem. The first model download and full-match transcription may take time. For a live presentation, prepare a short verified sample and run it before the presentation.

To process only a three-minute section while preserving timestamps relative to the full video:

```bash
python src/pipeline.py "/path/to/football-match.mp4" --start-seconds 0 --duration-seconds 180 --out outputs/demo --model base --language en
```

Outputs:

- `transcript.csv`: every Whisper segment with estimated timestamps, event category, rule/model scores, and NLP cue fields.
- `highlights.csv`: ranked candidate moments with start/end and peak timestamps, event type, score components, and commentary.
- `highlights.json`: structured event evidence, NLP cues, local model feature explanations where supported, and pipeline limitations.
- `highlights.md`: readable report for the demonstration and slides.

The ranking is returned in descending score order. Use timestamps to locate moments in the source video, then verify them manually. ASR timestamps are estimates; do not present them as frame-accurate edits.

## Run with a pre-timestamped match caption file

The project also has a no-audio path. If you have the matching `Labels-caption.json`, run:

```bash
python3 src/process_labels.py "/path/to/Labels-caption.json" --out outputs/city_newcastle
```

This reads the dataset's existing per-half video positions, caption descriptions, event labels, and `important` annotations. It exports `ranked_highlights.csv`, `.json`, and `.md`. This is an annotation-driven demonstration: the timestamps and captions come from the supplied JSON, and the priority is a transparent baseline. It is not an audio transcription or an independent model prediction. Open `1_720p.mkv` or `2_720p.mkv` as shown for each moment and check the timestamp before editing.

## Event-supervised training milestone

For six EPL matches, the project has English commentary transcripts from SoccerNet-Echoes and matching SoccerNet `Labels-v2.json` files. Build aligned 30-second commentary windows around annotated goals, shots, and cards, plus routine windows sampled away from those events:

```bash
python3 src/build_event_training.py
python3 src/train_highlight_model.py --labels data/event_aligned_context.csv --out models/commentary_event_detector.pt
```

Evaluation leaves out one complete match at a time. These are weak labels for **event-related commentary**, not subjective highlight judgments: an annotated shot is not automatically an exciting highlight. Treat the metrics as exploratory. Apply the trained event-context reranker to a saved transcript without rerunning Whisper:

```bash
python3 src/rerank_transcript.py outputs/final_demo_whisper_translation/transcript.csv --video-name "Crystal Palace vs Manchester City" --out outputs/event_model_demo --classifier-model models/commentary_event_detector.pt --min-score 0.5
```

The ranking combines transparent football-event phrase matches with the model's surrounding-context score. The model does not label ordinary commentary on its own, which avoids long false-positive stretches when the small classifier's scores cluster near 0.5.

The trained model is a genuine but small PyTorch neural network (8,192 hashed text features → 32 hidden units with ReLU/dropout → event-related score). The reviewed checkpoint has 380 mixed examples across seven matches. Its leave-one-match-out mean F1 is 0.636 versus 0.587 for the linear baseline, but the fold results vary and the labels mix SoccerNet event timestamps with a small subjective review set. This is exploratory event-related classification, not evidence that the system reliably finds subjective highlights. See `models/commentary_event_detector_reviewed.metrics.json`.

### TF-IDF model trained by the project

The course-based alternative fits its own vocabulary and TF-IDF weights from project commentary, then trains a PyTorch MLP from random initialization. It does not use pretrained text embeddings. Reproduce training and match-held-out evaluation with:

```bash
python3 src/train_tfidf_highlight.py --out models/commentary_tfidf_mlp_preprocessed.pt
```

The updated checkpoint is `models/commentary_tfidf_mlp_preprocessed.pt`. It includes normalized unigram/bigram TF-IDF features plus project-engineered negation, hypothetical, retrospective, live-action, punctuation, capitalization, and excitement cues. On the same seven match-held-out folds it reached macro F1 0.667 (weighted F1 0.671), compared with macro F1 0.655 for the earlier TF-IDF MLP. This small change is exploratory and uses mixed weak event labels plus a small manually reviewed set; it is not a validated measure of subjective highlight quality or universal English commentary performance. The earlier checkpoint remains available for comparison. Use the updated one in the pipeline with:

```bash
python3 src/pipeline.py "/path/to/match.mkv" --start-seconds 0 --duration-seconds 180 --out outputs/tfidf_preprocessed_demo --model small --language auto --translate-to-english --classifier-model models/commentary_tfidf_mlp_preprocessed.pt --min-score 0.5
```

For candidate ranking without rerunning Whisper, use `models/commentary_tfidf_mlp_preprocessed.pt` with `src/rerank_transcript.py` and a saved `transcript.csv`.

### Multi-task event-family model

The next model adds a shared text trunk with two project-trained heads: event-related vs routine, and a broad event family (`goal`, `shot_or_chance`, or `card`). The family labels come from SoccerNet event annotations and remain weak/noisy labels. Across seven match-held-out folds, event-family accuracy averaged 0.816 but macro-F1 averaged only 0.343; the accuracy is inflated by the majority shot class, and goal/card recall was poor. Treat this head as an experimental comparison, not a reliable event classifier. It does not predict player identity, excitement, or whether an event is subjectively a highlight. Train it with:

```bash
python3 src/train_multitask_model.py
```

This writes `models/commentary_multitask_tfidf.pt` and match-held-out diagnostics. Use it instead of the single-head checkpoint to add the event-family prediction to the structured output:

```bash
python3 src/pipeline.py "/path/to/match.mkv" --duration-seconds 180 --out outputs/multitask_demo --model small --language auto --task translate --classifier-model models/commentary_multitask_tfidf.pt --min-score 0.5
```

### Match-half sequence and attention model

The next experimental architecture reads ordered commentary windows within each match half, using a project-fitted TF-IDF input, a bidirectional GRU, temporal self-attention, and eventness/event-family output heads. Train and evaluate it by holding out complete matches:

```bash
python3 src/train_sequence_model.py --epochs 25
```

It writes `models/commentary_sequence_attention.pt` and `.metrics.json`. The current 25-epoch run on 380 weak/mixed windows had aggregate leave-one-match-out eventness accuracy 0.600 and macro-F1 0.597. This is an experimental single match-half sequence model, not a full match-level hierarchy; the existing TF-IDF MLP performed better in its earlier evaluation. To run it:

```bash
python3 src/pipeline.py "/path/to/match.mkv" --duration-seconds 180 --out outputs/sequence_demo --model small --language auto --task translate --sequence-model models/commentary_sequence_attention.pt --min-score 0.5
```

The output includes candidate event-family predictions and temporal attention to nearby transcript windows. Attention values are model weights, not proof that a window caused a prediction. A verified optional roster file can be supplied with `--entity-aliases data/my_match_aliases.json`; use `data/entity_aliases.example.json` as the format. Fixture team names are read from SoccerNet-style folder names where possible. Player mentions and pronoun links are reported only when an exact alias is configured; uncertain pronouns remain unresolved. The event frame is a transparent lexical template, not a full semantic-role model.

After reviewing output candidates and non-selected control moments, import the completed review CSV and retrain while keeping each whole match out of its own evaluation fold:

```bash
python3 src/build_review_training.py --reviews "/path/to/Candidate Commentary Scores.csv" --out data/event_and_manual_review_context.csv
python3 src/train_highlight_model.py --labels data/event_and_manual_review_context.csv --out models/commentary_event_detector_reviewed.pt
```

Human-review metrics are small-sample results and should be reported separately from the weak SoccerNet event-label folds. If reviewed examples are later added to training, use a different match for an untouched final test.

## Current boundaries and next steps

- Untimestamped Kaggle match narratives are useful for examples and future annotation, but they cannot train or evaluate timestamp extraction. Whisper timestamps come from the input video's audio.
- The rule-based scorer can miss paraphrased events and can over-score ordinary mentions of event words. Edit the vocabulary in `src/pipeline.py` and report the baseline as such.
- After adding translated live-goal phrases and suppressing season statistics and ordinal shot counts, the Leicester reranking now catches its three first-half annotated goals within 2–9 seconds. The original NLLB ranking missed all three; see `outputs/leicester_arsenal_refined/video_audit.md`. This is same-match debugging, not independent evaluation.
- The Russian Liverpool–Aston Villa NLLB run missed the annotated first-half goal near 00:01:03 and produced false or weak candidates elsewhere. A three-minute Whisper direct-translation transcript did include “goal from Liverpool” near 00:01:08, but this short sample does not establish full-match performance. The candidate-by-candidate video review is in `outputs/liverpool_aston_villa_test/video_audit.md`.
- The event-supervised set inherits annotation/transcript alignment noise and measures event-related commentary rather than highlight quality. The human-reviewed set is very small and covers one match. Do not claim broad reliability across matches, commentators, or languages until the system is evaluated on independently reviewed matches and languages.
- Match-player names and nicknames are not learned or normalized by the current model. Whisper often misspells them, and NLLB can translate or transliterate them incorrectly. A roster-aware alias list is future work, not a current capability.
- Future extension: use the exported timestamps plus configurable padding to cut clips with FFmpeg. Keep video cutting out of this milestone so transcription and ranking can be demonstrated independently.

## Team ownership suggestion

- Member 1: input/data handling, transcript review, normalization, integration and README.
- Member 2: event labels and annotation guideline; review and expand the football event vocabulary.
- Member 3: ranking, evaluation on held-out annotated matches, and demo outputs.
- All members: label a portion of the evaluation set and review the final report/slides.
# Train the human-supervised commentary highlight selector

The classifier is trained from the human decisions in
`data/manual_annotation_context.csv`. Each row contains about 30 seconds of
English commentary with a `human_label` of `highlight`, `routine`, or `unclear`.
The trainer excludes `unclear` rows, reports leave-one-match-out results, then
saves a final model trained on the reviewed rows. Inference builds the same
30-second context around each timestamp before scoring it.

From this project folder, with the virtual environment active:

```bash
python3 src/train_highlight_model.py
```

The model is saved to `models/commentary_highlight.pt`; the exploratory
evaluation and label counts are saved beside it as
`models/commentary_highlight.metrics.json`. The current 60-row pilot has low
match-held-out scores, so do not present it as a reliable or general model.

To use it in the video pipeline, translate non-English speech into English
before ranking. For example, run on a video from a match not in the manual
annotation set:

```bash
python3 src/pipeline.py "/path/to/match/1_224p.mkv" --start-seconds 0 --duration-seconds 180 --out outputs/model_demo --model small --language auto --task translate --translate-to-english --classifier-model models/commentary_highlight.pt --min-score 0.5
```

The classifier is a small hashed unigram/bigram linear model using PyTorch; it
does not fine-tune Whisper or NLLB. The highlight score is an uncalibrated
ranking score. Verify timestamps and candidate text against the match before
reporting results.
