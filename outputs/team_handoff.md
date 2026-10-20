# Commentary2Highlight — teammate handoff

**Purpose:** explain what is implemented, show actual outputs, identify known failures, and list the next work needed. This describes the code and runs currently in the shared project folder; it does not claim that every proposed research stage is complete.

## 1. Current system in one sentence

The project extracts audio from a football video, transcribes it with timestamped pretrained Whisper, optionally translates it to English, then uses our own NLP rules and trained PyTorch text models to rank commentary moments and export timestamped highlight commentary as Markdown, CSV, and JSON.

## 2. Proposed pipeline stages and status

Use the status wording below when presenting progress.

| Stage | Status | Implemented now | What remains |
|---|---|---|---|
| 0. Speech input / Whisper | **Completed and working** | FFmpeg audio extraction; Faster-Whisper `small` in the current demo; automatic language detection; transcript segments with estimated start/end times. | Verify each video has decodable commentary audio. ASR quality and names vary by commentator and language. |
| 1. Data collection and match-level evaluation | **Need further working** | SoccerNet-Echoes transcripts are aligned to SoccerNet event timestamps; model training evaluates with whole matches held out. | Current training set is small and partly weakly labeled. Add independently reviewed matches; deduplicate and incorporate the separate 60 human-reviewed rows only after checking overlap. |
| 2. NLP text preparation | **Completed and working** | Project code normalizes text and records negation, hypothetical/past wording, live-action, emphasis, and phrase evidence. Original and translated transcript text are retained where the selected Whisper route permits it. | Translation and ASR errors still affect the text. |
| 3. Coreference / entity linking | **Need further working** | Exact team/player aliases can be loaded. A nearby pronoun is linked only when exactly one verified player alias appeared in the prior 30 seconds; otherwise it stays unresolved. Team names can be read from a SoccerNet-style fixture folder. | No verified player roster is currently supplied for the demo. This is a cautious dictionary heuristic, not a trained coreference system. |
| 4. Event extraction / SRL | **Need further working** | Project-written rules identify event candidates and evidence. A lexical frame reports event type, candidate actor/team names, and outcome wording. | This is not full semantic-role labeling. It does not reliably extract who performed an action, assists, score state, or event outcome. |
| 5. Text representation | **Completed and working** | Our code fits its own TF-IDF vocabulary and IDF values from project commentary; a hashed unigram/bigram representation is also available. | This is a sparse, project-fitted representation, not a pretrained Transformer embedding. |
| 6. Deep-learning / sequence model | **Need further working** | Project-trained PyTorch MLPs; an experimental BiGRU with temporal self-attention over ordered match-half windows is also implemented and trained. | The BiGRU is not a full two-level match/half hierarchy. Its held-out result is below the TF-IDF MLP’s earlier result. More independent labels are needed. |
| 7. Multi-task outputs | **Need further working** | Models output event-vs-routine; the multi-task versions also predict broad event families: goal, shot/chance, card. | Family labels are weak and sparse. No trained excitement or player-identity head exists. The event-family model is weak on rare goals/cards. |
| 8. Attention / explanation | **Need further working** | Sequence output includes temporal attention weights; TF-IDF output includes local feature-removal evidence. | Attention weights are not reliable human explanations. In the demo, weights were spread thinly across many transcript windows. Do not call this a validated heatmap. |
| 9. Peak detection and grouping | **Completed and working** | Nearby candidate segments are grouped; the strongest segment is reported as a peak timestamp with score and group size. | This is score-based grouping, not a learned or independently validated peak detector. |
| 10. Structured results | **Completed and working** | Full transcript and ranked candidates are exported as Markdown, CSV, and JSON with timestamps and evidence. | No video clip is generated in this milestone. Timestamps must be checked against the video. |

## 3. Actual generated outputs to show

The latest sequence-model run processed the first **3 minutes** of Crystal Palace vs. Manchester City. Whisper detected Spanish and returned English text. It produced **81 transcript segments** and **2 ranked candidates**:

1. **00:01:23–00:01:26 — Chance or shot**, score 0.85. Text: “The high shot to the front of the Crystal Palace.” The model family output was `shot_or_chance` (0.997).
2. **00:02:50–00:02:52 — Corner or free kick**, score 0.53. Text: “who arrived there at the rematch in the corner.” The separate event-family model said `goal` (0.565), which conflicts with the rule label and illustrates that this head needs further work.

Open these files for the demonstration:

- [Ranked highlights report](sequence_crystal_palace_demo/highlights.md)
- [Full timestamped transcript](sequence_crystal_palace_demo/transcript.md)
- [Structured predictions and event frames](sequence_crystal_palace_demo/highlights.json)
- [Transcript table](sequence_crystal_palace_demo/transcript.csv)
- [Candidate table](sequence_crystal_palace_demo/highlights.csv)

The TF-IDF MLP run on the same three-minute excerpt produced one candidate at 00:01:23, score 0.72:

- [TF-IDF MLP result](updated_crystal_palace_demo/highlights.md)

The direct English translation has clear errors in player names and football phrases. Show these files as a **working prototype output**, not as a clean final highlight reel. It outputs timestamps and commentary; it does not cut or render video clips.

## 4. Models and methods used

### Pretrained input models

- **Faster-Whisper `small`**: speech-to-text, language detection, and segment timestamps. The current demo uses Whisper’s direct English translation task (`--task translate`).
- **NLLB-200 distilled 600M**: optional text translation route when preserving source-language ASR; translation quality varies. It was not used for the latest sequence demo.
- **FFmpeg**: extracts audio from the input video.

### Project-built NLP and DL

- **Football phrase rules**: propose goal, penalty/VAR, card, shot/chance, and set-piece events; apply simple suppressions and retain matched phrase evidence.
- **TF-IDF**: our code builds the training vocabulary and inverse-document-frequency values from the project’s own commentary data.
- **PyTorch MLP**: trained from random initialization on project TF-IDF or hashed text features; scores event-related commentary using local context.
- **Multi-task TF-IDF MLP**: shared text features with eventness and broad event-family outputs.
- **BiGRU + temporal self-attention**: trained from project data on timestamp-ordered commentary windows grouped by match half; outputs per-window eventness and event family.
- **Entity linking / coreference rules**: optional exact aliases and cautious short-window pronoun linking. No names are invented when there is no verified alias.
- **Lexical event frames**: event category, evidence, and outcome cues; these are not a full SRL system.

We do **not** fine-tune Whisper or NLLB. We do **not** use a pretrained Transformer embedding for the downstream ranker.

## 5. Data used and not yet used

- Current combined training file: `data/event_and_manual_review_context.csv`, **380 windows from 7 matches**: **205 event/highlight** and **175 routine**. It combines **365 weak SoccerNet event/routine windows** with **15 manually reviewed windows**.
- Weak event types are uneven: 167 shot/chance, 23 goals, and 13 cards in the mapped multi-task file.
- Separate manual file: `data/manual_annotation_context.csv`, **60 labeled rows from 6 matches** (20 highlight, 40 routine). These labels are not in the current 380-row checkpoint data. Check for duplicate windows, then merge and retrain.
- Prepared holdout review: `data/crystal_palace_holdout_review.csv`, **14 rows**, currently unlabeled. Review the match footage before using it for evaluation; keep its separate key hidden until labeling is done.
- The supplied 2023–24 match-details CSV has unreliable row-to-commentary associations and no useful line timestamps. Its 189-row inferred reassociation copy is for human review only and is not in training.
- Five local SoccerNet MKVs are inventoried: three have usable audio, and both Manchester City–Newcastle 720p halves decode to empty audio. The 2023–24 match-details file does not have a corresponding local 2023–24 video in this inventory.

See [pipeline and dataset audit](pipeline_and_dataset_review.md) and [video/audio inventory](../data/video_audio_inventory.md) for detail.

## 6. Evaluation results and how to describe them

All scores below are **exploratory whole-match-held-out results** over seven matches. The labels mix weak event alignment and human review; they do not measure subjective highlight quality on a broad independent test set.

| Model | Result | Interpretation |
|---|---:|---|
| Project TF-IDF MLP | Mean held-out F1 **0.655** | Current strongest reported baseline; still small and mixed-label. |
| TF-IDF linear baseline | Mean held-out F1 **0.553** | Simple comparison model. |
| Multi-task TF-IDF event-family head | Mean family macro-F1 **0.343**; mean accuracy **0.816** | Accuracy is inflated by the majority shot/chance class; goal/card prediction is weak. |
| BiGRU + temporal attention | Held-out eventness macro-F1 **0.597**, accuracy **0.600** | Experimental sequence stage; below the earlier TF-IDF MLP result. |

Full fold metrics are in:

- [TF-IDF MLP metrics](../models/commentary_tfidf_mlp.metrics.json)
- [Multi-task model metrics](../models/commentary_multitask_tfidf.metrics.json)
- [Sequence model metrics](../models/commentary_sequence_attention.metrics.json)

## 7. What we need next

1. Review and deduplicate the 60 manually labeled rows, merge approved labels into a clean training file, and retrain.
2. Finish the 14-row Crystal Palace holdout review without viewing its hidden candidate/control key first.
3. Label at least one more independent match with both highlights and routine controls; keep that match out of model tuning.
4. Create verified team/player alias files for the chosen fixtures, then review the name corrections and pronoun links.
5. Compare transcription/translation routes against the source audio and video; specifically correct names and football terminology.
6. Keep the TF-IDF MLP as the current primary model unless a new model beats it on held-out matches. Treat the sequence model and event-family predictions as experimental.
7. After timestamps are manually checked, optionally add padded FFmpeg clip extraction as a later extension.

## 8. Reproduce the current demo

Run from the project folder:

```bash
cd "<project-root>"
source .venv/bin/activate
python3 src/pipeline.py "path/to/SoccerNet/england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City/1_224p.mkv" \
  --duration-seconds 180 \
  --out outputs/sequence_crystal_palace_demo \
  --model small \
  --language auto \
  --task translate \
  --sequence-model models/commentary_sequence_attention.pt \
  --min-score 0.5
```

To run the current stronger TF-IDF MLP comparison, replace `--sequence-model models/commentary_sequence_attention.pt` with `--classifier-model models/commentary_tfidf_mlp.pt` and use a different `--out` folder.

Training commands:

```bash
python3 src/train_tfidf_highlight.py
python3 src/train_multitask_model.py --epochs 80
python3 src/train_sequence_model.py --epochs 25
```

## Suggested explanation to the team

“We have a working speech-to-timestamp-to-ranked-commentary prototype. Whisper and optional translation are pretrained input tools; the downstream event rules, TF-IDF vocabulary, PyTorch MLPs, and BiGRU-attention model are ours and trained from our project data. The most reliable current model is the TF-IDF MLP baseline. Our labels are limited and partly weak, so player/coreference, semantic roles, rare event classes, translation quality, and sequence attention still need review and more data. The current result is a timestamped commentary highlight list, not an automatically cut video.”
