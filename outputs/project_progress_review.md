# Commentary2Highlight: current progress and improvement plan

## What the current system does

The new teammate pipeline PDF and 2023–24 match-details CSV are reviewed in [`pipeline_and_dataset_review.md`](pipeline_and_dataset_review.md). The CSV has 303 records, but none of the 291 detectable opening team pairs matched its row's Home/Away fields. A team-pair plus exact-stadium check made a separate 189-row inferred association copy for review; the remaining 114 rows are excluded. It has no reliable timestamps or labels and is not used for training. The review maps each teammate stage to implemented, partial, and not-yet-implemented project work.

1. FFmpeg extracts audio from a video.
2. Faster-Whisper detects speech language and supplies timestamped transcript segments.
3. English can come from Whisper's direct translation task or from NLLB translating the recognized text.
4. Football phrase rules propose candidate event moments; a small PyTorch neural classifier reranks the proposed moments using nearby commentary.
5. The project saves a full transcript and ranked commentary timestamps as CSV, JSON, and Markdown.

This is a working NLP/DL prototype. It does not yet cut video clips, detect events directly from pixels, or reliably recognize all languages, players, and nicknames.

## Videos currently visible on the external drive

| Match | Audio / available route | What is available |
|---|---|---|
| Crystal Palace 0–1 Manchester City, first half | Decodable commentary; Whisper detected Spanish | Full transcript and baseline ranking in `outputs/autolang_full/`. Updated reranking is in `outputs/crystal_palace_refined/`: eight candidates, mostly shot-related wording; no matching Labels-v2 file was found locally, so these remain unverified. The checkpoint includes Crystal Palace review examples, so this is not an independent test. |
| Liverpool 3–2 Aston Villa, first half | Decodable Russian commentary | Full Russian ASR + NLLB run and audit in `outputs/liverpool_aston_villa_test/`; it misses the annotated first-half goal at 00:01:03. Full-half Whisper direct translation in `outputs/liverpool_direct_translate_full/` catches goal wording at 00:01:19 (+16 sec) but also hallucinates a goal phrase at 00:31:18 when no first-half goal is annotated there. This is a better clue for the first goal, but not reliable enough to use without review. |
| Leicester 2–5 Arsenal, first half | Decodable French commentary | Full Whisper + NLLB run in `outputs/leicester_arsenal_test/`. Refined output in `outputs/leicester_arsenal_refined/` recovers all three first-half annotated goals within 2–9 seconds. The current English negation-filter version is `outputs/leicester_arsenal_negation_refined/`; see the earlier match audit at `outputs/leicester_arsenal_refined/video_audit.md`. |
| Manchester City 6–1 Newcastle, both halves | Metadata lists Vorbis, but each ten-second WAV extraction is empty (78 bytes) | Caption-based report in `outputs/city_newcastle/`. Seven goal-caption timestamps were visually checked against the video in `outputs/city_newcastle/video_audit.md`. These are dataset captions/flags, not audio transcription or a model's prediction. |

There are **five** MKV files visible under `path/to/SoccerNet` at this time: three with usable commentary and two that decode to no audio samples. See [`data/video_audio_inventory.md`](../data/video_audio_inventory.md).

## Findings from checking the actual footage

- Liverpool's Labels-v2 annotation and video place the first goal at about **00:01:03**. The generated list instead includes a later candidate at **00:01:35–00:01:51**, which is not that goal. The earlier statement that the latter candidate was a goal was incorrect.
- Around the missed goal, Russian Whisper output rendered a word related to “goal” as `голой`; NLLB translated it as “naked face.” That removed the event clue before phrase matching.
- A diagnostic MLP trained without Liverpool scored the missed-goal text context 0.569 and the later non-goal candidate 0.608. The current binary n-gram scores are not reliable enough to run as an independent candidate generator.
- Full-half direct Whisper translation preserved “goal from Liverpool” at 00:01:19, 16 seconds after the annotated 00:01:03 goal. It also produced a false goal phrase at 00:31:18. Compared with Russian ASR → NLLB, it recovers a clue for the opening goal, but still has a timing error and a false positive.
- I added English negative-context filters for phrases such as “didn't get a shot” and “cannot shoot.” Reranking Liverpool's saved direct-translation transcript dropped two such false candidates (9 to 7); the false goal phrase remains. This is a small rules improvement, not a new model training result.
- In Leicester, the translated goal language included “he'll score,” “equalization,” and “first goal of the season.” Adding those rules and local context recovered the three first-half goals; a season-statistics false positive was removed. See `outputs/leicester_arsenal_refined/video_audit.md`.
- A diagnostic MLP trained without Liverpool scored the missed-goal text context 0.569 and the later non-goal candidate 0.608. Its current binary n-gram scores are not reliable enough to run as an independent candidate generator.
- Of the 11 Liverpool candidates, six fall within 30 seconds of a SoccerNet event timestamp. This is only approximate event alignment; it is not highlight precision, and two candidates overlap the same event sequence.
- Liverpool's candidate audit and City–Newcastle's seven goal checks are manual visual checks. The caption-based City–Newcastle output was supplied with dataset event text and important flags, so it is not an ASR or model evaluation.
- In the Crystal Palace output, the word “goal” is also triggered by past-match descriptions, and phrases such as “chance,” “shot,” “falta,” and “corner” create ordinary-commentary candidates. Phrase matching needs context and better handling of negation, past events, and ambiguous words.

## Model and evaluation status

- The current reranker is an 8,192-feature hashed n-gram feed-forward network with 32 hidden units. It produces an **uncalibrated ranking score**, not a probability.
- It is a binary event-context reranker, not a multi-class event recognizer. The phrase rules assign event names such as goal, shot, or corner; the network cannot change a missed candidate into a goal.
- The reviewed checkpoint has 380 mixed examples across seven matches. Its exploratory leave-one-match-out mean F1 is 0.636; the linear baseline is 0.587. Results vary by match, and training labels combine SoccerNet event timestamps with only a small manual review set. This measures event-related text, not whether viewers would consider a moment a highlight.
- Liverpool and Leicester are in the training data for this checkpoint, so their outputs cannot be presented as independent model tests.

## Highest-value work before submission

1. Use the full Liverpool comparison to tune the goal handling: retain the 00:01:19 goal wording as a candidate with a wider review window, while checking false live-goal claims like 00:31:18 against match timeline and local context.
2. Review the remaining Leicester candidates against the video. The revised rules recover three annotated goals, but other candidate moments still include routine commentary.
3. Add or adjust event phrase rules only after checking their false positives. Separate explicit goal language, shots, cards, corners, fouls, and retrospective mentions; avoid treating every mention of “goal,” “shot,” or “corner” as a live event.
4. For one match that is not used for training, review all predicted candidates plus a small sample of unselected routine moments. Report precision/recall against the agreed human labels, with sample size and match stated.
5. Make a match-specific spelling/alias list for player names and teams as a correction layer. Do not claim the current model learned all names or nicknames.
6. Present the language scope honestly: Spanish, French, and Russian have been tried, but the Russian NLLB run missed a verified goal. “Works on every language and match” is not supported by the current evidence.

## Latest TF-IDF model run and video review

The project now also has a team-fitted TF-IDF encoder plus a PyTorch MLP, trained from random initialization. On the same seven whole-match holdouts it has macro F1 0.655, compared with 0.636 for the hashed-feature MLP. The TF-IDF linear baseline reached 0.553. These figures come from mixed labels: 365 event-window/routine samples plus only 15 human-reviewed Crystal Palace examples. They are exploratory event-related scores and do not prove better highlight selection.

The 3-minute `outputs/tfidf_demo/` run is functional, but visual review tempers its apparent result:

- At 00:01:26, frames show a Palace attack near the penalty area, then a close-up; the exact shot claim is not clear in the sampled frames. Keep it tentative.
- At 00:02:50, the footage shows a Martin Kelly throw-in near 00:02:52, not a corner at the reported time. Treat this as a false or mistimed candidate.

Full review is in `outputs/tfidf_demo/video_audit.md`. The new model, like the prior one, is text-only; it cannot see whether the footage shows a corner, throw-in, or shot. Do not tune the model from these two output rows without adding explicit human labels and separate event-type labels.

I added a whole-match exclusion option (`--exclude-match`) and ran the saved transcripts through three checkpoints that had not trained on the target match. This prevents the trained reranker from seeing that match's labels. These runs use saved ASR/translation transcripts, so they check downstream event rules + reranking, not new ASR accuracy.

| Target match excluded from MLP training | Held-out output | What it shows |
|---|---|---|
| Crystal Palace–Man City (365 other training examples) | 7 candidates | Sampled footage around the candidates shows several attacks near goal. At 00:02:50, the reported instant is a throw-in; the wider 30-second context continues into an attack and Hart collects. The exact corner label is wrong/mistimed. No matching SoccerNet Labels-v2 is available for a quantitative event check. |
| Leicester–Arsenal (307 other examples) | 13 candidates | The three annotated first-half goals appear at +2, −9, and +3 seconds. The phrase rules had already been adjusted after inspecting Leicester, so this is not a fully untouched end-to-end test, even though its classifier checkpoint excluded Leicester. |
| Liverpool–Aston Villa (316 other examples) | 9 candidates from Russian ASR → NLLB; 8 from Whisper direct English translation | NLLB still misses the 00:01:03 goal. Direct Whisper translation finds a goal phrase at 00:01:19 (+16 sec) but also a false goal phrase at 00:31:18. The language route matters; neither is reliable without review. |

The candidate lists and checkpoints are in `outputs/heldout_*` and `models/tfidf_without_*.pt`. The separate Crystal review sheet has seven held-out candidates and seven routine controls: `data/crystal_palace_holdout_review.csv`. Reviewers should fill `human_label`, `event_type`, `annotator`, and `notes`; keep `outputs/heldout_crystal_palace/review_key.csv` aside until all labels are entered. This gives us a small human-checked evaluation before we tune thresholds or retrain on the new labels.

The currently visible five MKV files were all accounted for: Crystal Palace, Leicester, Liverpool, and the two Manchester City–Newcastle halves. Three have usable audio; two City–Newcastle videos do not. Existing visual audits cover the Leicester goal sequence, the Liverpool candidate list, and seven City–Newcastle caption-derived goals. Those audits show useful cases and real misses; they do not amount to comprehensive evaluation of every minute of every video.

## Recommended model plan for the teammate pipeline

Keep the agreed stages as a staged, explainable pipeline:

1. Timestamped Whisper transcription/English translation (pretrained input components).
2. Team-written normalization, match-specific roster/alias matching, and context-window construction.
3. NLP features implemented by the team: tokenization, negation-preserving normalization, unigram/bigram TF-IDF; optionally compare team-trained Word2Vec only after collecting a larger corpus.
4. Candidate extraction using event patterns, followed by a team-trained binary MLP for event-related vs routine context. Keep the event type and the subjective highlight label separate.
5. Temporal smoothing/peak grouping and timestamped CSV/JSON/Markdown results.
6. Manual visual verification now; automatic clip extraction can be the future extension.

Coreference, SRL, hierarchical LSTM, multi-task player/excitement heads, and attention heatmaps are not currently implemented. A self-trained LSTM would be possible, but with only 380 mixed examples it would likely overfit and would not fix errors in Whisper or translation. Do not add it just to increase the number of named models. Add sequence modeling after collecting many more labeled windows and a match-disjoint test set. The next high-value contribution is structured human review across more matches, including routine controls and missed events, then retrain and compare both models on a held-out match that is not in training.

The work is substantial progress for a student prototype: audio extraction, timestamped multilingual ASR/translation, NLP candidate selection, PyTorch reranking, structured outputs, and a second caption-based path for silent videos. The next measurable gain should be improved recall of real goals without flooding the report with routine commentary.
