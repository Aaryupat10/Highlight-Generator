# Commentary2Highlight: course-aligned model plan

## Executive summary

The project already contains a small PyTorch feed-forward network (MLP) trained on project commentary examples, using hashed word and adjacent-word features. Whisper and NLLB are pretrained speech/translation tools; they are not the project’s trained highlight model. The current MLP labels mix dataset event timestamps with a small amount of human review, so its reported score measures event-related commentary more than subjective highlight quality. Since parts of the existing code were developed with AI assistance, each teammate should inspect, understand, and make their own contributions before presenting or submitting it as original assessed work.

The strongest practical project is a transparent NLP pipeline plus a model trained by the team:

1. Use Whisper for timestamped speech and translation to English when needed.
2. Keep original text when available, preserve negation, punctuation, names, and timestamps, and normalize text consistently.
3. Extract course-level NLP features: word unigrams/bigrams, TF-IDF, and optional POS/entity/event cues.
4. Train and compare two text classifiers on reviewed examples: a classical baseline such as Naive Bayes or Logistic Regression, and a PyTorch MLP written and trained by the team.
5. Predict two clearly defined targets: event type (goal, shot/chance, card, set piece, routine) and human-reviewed highlight/routine status. Treat dataset event annotations as weak event labels, not as subjective highlight judgments.
6. Smooth nearby predictions, keep local peaks, and export English commentary with timestamps to CSV/JSON/Markdown.

This gives a defensible student-built NLP and DL contribution while keeping Whisper/translation as input preparation. It does not promise perfect output for every match or language.

## Course methods found in the supplied materials

The three OneDrive archives contain NLP Units 1–5, Deep Learning Units 1–4 and course materials; several files are duplicates. The individual labs cover the following relevant methods:

| Material | Methods covered | Good use in this project |
|---|---|---|
| NLP Lab 1; NLP Units 1–2 | Tokenization, normalization, punctuation/stopword handling, stemming/lemmatization, POS, Bag of Words, n-grams, TF-IDF | Build and explain the text-preprocessing and feature pipeline. Preserve `not`, `no`, player names, time words, and punctuation/emphasis that can signal football events. |
| NLP Lab 4 | Unigram, bigram, trigram language models and perplexity | Useful as an optional language-modeling experiment; not needed for selecting highlights. |
| NLP Labs 5–6; NLP Unit 4 | POS tagging, Named Entity Recognition, information extraction | Extract player/team/time mentions. For this project, begin with match rosters and a transparent alias dictionary; a custom learned NER model would need token-level name annotations. |
| NLP EXP7; NLP Unit 5 | Word Sense Disambiguation, Lesk/WordNet, context and ambiguity | Explain why football terms need context. A custom filter/classifier for “shot” versus “shot that did not happen” is more useful here than running a general WSD system. |
| NLP Lab 8; NLP Unit 5 | Word2Vec CBOW/Skip-gram and GloVe | Can be a small optional experiment. The lab describes using pretrained embeddings, but Word2Vec itself can be trained from scratch on your own commentary. The available corpus is small and ASR/translation contains spelling errors, so do not make it the core model. |
| NLP Lab 9; NLP Units 1, 4–5 | Text classification, Naive Bayes, Logistic Regression/SVM, train/test evaluation and metrics | Compare a classical classifier with your own neural model. Use match-level splits and report precision, recall, and F1. |
| DL Labs 2–4; DL Units 1–2 | Perceptron, feed-forward networks, activation, loss, backpropagation | Direct support for the team-written PyTorch MLP classifier. |
| DL Labs 5–6; DL Unit 3 | Bias/variance, epochs, validation, hyperparameter tuning, dropout, L1/L2, early stopping, gradient-descent variants | Tune the MLP with a validation match, control overfitting, and show training/validation loss. |
| DL Lab 10; DL course policy | LSTM and sequential data | A small LSTM over consecutive commentary windows is a possible later extension. It is not the best deadline-stage choice with only a few hundred labeled windows. |
| DL Labs 8 and DL Unit 4 | CNNs and image processing | Not needed for a commentary-text model. Add video-frame/CNN analysis only as a separate visual branch later. |

Coreference resolution is introduced in NLP Unit 1. Semantic roles are mentioned in NLP corpus material, but the supplied labs do not give a practical SRL implementation. Neither full coreference nor SRL is necessary for the current deadline.

## Review of the teammate’s 10-stage proposal

The teammate proposal is a useful research vision, but several stages are not in the working code and would be too much to implement and validate now.

| Proposed stage | Current status | Practical decision |
|---|---|---|
| 0. Whisper ASR | Working as a pretrained input component | Keep. It provides timestamped speech. It is allowed to be pretrained because the team contribution is the downstream model. |
| 1. Collect and split commentary | Partly present through SoccerNet audio/transcripts and captions | Keep, but require timestamps for evaluation. A line number in an un-timestamped Kaggle commentary file is not a video time. |
| 2. Preprocessing | Present in the project’s text pipeline, with rules being improved | Keep light, consistent cleanup. Do not strip negation, names, or useful emphasis. |
| 3. Coreference | Not implemented | Optional future stage. For this MVP, a 15–30 second context window can supply nearby player/event context without claiming pronoun resolution. |
| 4. SRL/event extraction | Not implemented as SRL | Replace with a manageable event labeler: event class plus optional POS/NER/alias cues. Use transparent templates for agent/action/outcome only where the transcript clearly supports them. |
| 5. Transformer embeddings | Not used by the project’s model | Replace with team-built BoW/TF-IDF/ngram features. If desired, compare a Word2Vec model trained from scratch as a separate experiment. |
| 6. Hierarchical sequence model | Not implemented | Start with the existing MLP over a fixed commentary context. Only add an LSTM if annotation volume and results justify it. |
| 7. Multi-task heads for excitement, event type, player | Not implemented | Reasonable future target, but first build two targets—event type and highlight/routine. Add a player head only after collecting player-span labels. |
| 8. Attention heatmap | Not implemented | Do not claim attention. For the MLP, show matched phrases or a simple feature ablation explanation instead. |
| 9. Peak detection/ranking | Nearby candidate merging exists | Keep and improve by scoring a sliding window, smoothing nearby scores, and merging one event into one timestamp range. |
| 10. Structured output | CSV/JSON/Markdown are generated | Keep. This meets the current goal of timestamped commentary. Video cutting can remain future work. |

The proposed Burnley–Manchester City example describes a raw commentary block split into line numbers and reports an approximate line number/time. Unless the supplied source has real timestamps, that data can teach text categories but cannot validate or provide video clip timestamps. Use timed SoccerNet-Echoes segments or Whisper timestamps for that part.

## What the team has already trained

- `src/train_highlight_model.py` trains the model; `src/text_classifier.py` builds the input features and loads it for inference.
- The current MLP has 8,192 hashed unigram/bigram features, one 32-unit ReLU hidden layer, dropout, and a binary output. It is initialized and trained by the project code; it is not a pretrained transformer.
- The reviewed checkpoint uses 380 rows: 365 rows weakly aligned from SoccerNet event timestamps and 15 human-reviewed Crystal Palace examples. Reported leave-one-match-out F1 is 0.636 for this mixed event-related target versus 0.587 for a linear baseline. This is exploratory and does not establish highlight quality.
- A separate 60-row human-only pilot has 20 “highlight” and 40 “routine” labels. Its match-held-out F1 ranges from 0.00 to 0.33 in the saved metrics, so the current human labels do not yet support a strong general highlight-quality claim.
- `Candidate Commentary Scores.csv` contains eight candidate rows, all labeled “Highlight”; because it has no routine negatives, it cannot train a useful highlight-vs-routine classifier by itself.

### Implemented model improvement: team-fitted TF-IDF + PyTorch MLP

I added a second, course-aligned model path using a vocabulary and inverse-document-frequency weights fitted from the project training text. It uses unigram/bigram TF-IDF, simple punctuation/negation cues, and a 32-unit PyTorch MLP initialized and trained by the project. The model and its vocabulary are saved together, so pipeline inference uses the exact same features. No pretrained text embeddings are used.

To reproduce the experiment from the project folder:

```bash
python3 src/train_tfidf_highlight.py
```

This writes `models/commentary_tfidf_mlp.pt` and `models/commentary_tfidf_mlp.metrics.json`. The script fits TF-IDF separately inside each leave-one-match-out fold, then trains the final deployable MLP on the current 380 labeled examples across seven matches.

For a fairer match-specific review, the trainer can omit a complete match from the final saved checkpoint. For example:

```bash
python3 src/train_tfidf_highlight.py --exclude-match crystal_palace_mancity --out models/tfidf_without_crystal_palace_mancity.pt
python3 src/rerank_transcript.py outputs/crystal_palace_refined/transcript.csv --video-name "Crystal Palace vs Manchester City (held out)" --out outputs/heldout_crystal_palace --classifier-model models/tfidf_without_crystal_palace_mancity.pt --min-score 0.5
```

Equivalent held-out checkpoints and outputs were generated for Crystal Palace, Leicester, and Liverpool. This excludes each target match from the learned MLP and TF-IDF vocabulary. It does not make a fully blind end-to-end test when transcript phrase rules were previously tuned on a target match.

| Model evaluated on the same seven held-out matches | Macro F1, threshold 0.5 | Example-weighted F1 |
|---|---:|---:|
| Existing hashed unigram/bigram MLP | 0.636 | 0.624 |
| New TF-IDF MLP | 0.655 | 0.654 |
| New TF-IDF linear baseline | 0.553 | 0.552 |

The TF-IDF MLP is a modest improvement on this exploratory mixed-label evaluation, and it beats its matched TF-IDF linear baseline. This is not proof of better editorial highlight quality: most examples are weakly aligned SoccerNet events, there are only seven matches, and the threshold was not independently tuned. Keep the old checkpoint as a comparison, collect more independently reviewed routine/interesting examples, and report the metric as event-related classification performance. To apply the new checkpoint in the existing video pipeline, use `--classifier-model models/commentary_tfidf_mlp.pt`.

## How to make the model meaningfully better

### 1. Define labels before training

Use two separate annotations rather than one overloaded “highlight” label:

- `event_type`: goal, shot/chance, card, corner/free kick, routine, or uncertain.
- `highlight_label`: highlight, routine, or uncertain, based on a written team guideline.

SoccerNet’s Goals/Shots/Cards can weakly supervise `event_type`. They do not tell us whether a moment is exciting enough to be a highlight. Human review must supervise `highlight_label`.

For a useful short-term training set, review both predicted candidates and ordinary control moments across multiple matches. Include false positives, missed events, and routine commentary. Label at least a few hundred windows if possible; have all three teammates agree on the same small subset and resolve disagreements. Never mark every model candidate as a highlight by default.

### 2. Train and compare models using methods from class

Use a single, fixed preprocessing pipeline, then compare:

- **NLP/ML baseline:** TF-IDF word unigram/bigram vectors plus Naive Bayes or Logistic Regression/SVM.
- **Team-built DL model:** a PyTorch MLP trained from random initialization on the same TF-IDF/ngram and event/context features. Use a small hidden layer, dropout/weight decay, and early stopping based on a validation match.

The same text/label split must be used for both. Hold out whole matches, not random lines from the same match. Report per-class precision, recall, F1, confusion matrix, number of labeled examples, and exactly which match was held out. Compare event classification separately from highlight/routine classification.

### 3. Keep a fair input path for languages

For a match with commentary in another language, run Whisper’s English translation task or the existing NLLB route, then feed English text to the team-trained classifier. Preserve the timestamp and original text whenever the selected route makes it available. Translation errors in names and football terms remain a limitation; test Spanish, French, Russian separately and report their results separately. Do not claim “all languages” from three examples.

### 4. Handle names realistically

Use the NLP NER topic plus a team-built match roster/alias table to normalize likely player and team names. Keep the ASR spelling alongside the proposed canonical spelling. This is an explicit project resource, not a model that has learned every footballer. A learned player head needs examples labeled with the player involved; do not add that output before those labels exist.

## What to say in the presentation

“Whisper and translation are pretrained tools used to prepare timestamped English commentary. Our project contribution is the downstream NLP pipeline and the classifier we train ourselves: we preprocess commentary, extract n-gram/TF-IDF and football-context features, train a PyTorch feed-forward network from our labeled examples, compare it against a classical text-classification baseline, and rank timestamped candidate events. SoccerNet event labels give weak event supervision; human annotations are needed to evaluate whether a moment is truly a highlight.”

Call the current artifact a working prototype. Do not present the proposed coreference/SRL/Transformer hierarchy/attention stages as already implemented, and do not present current same-match examples as independent test results.

## Course policy note

The supplied NLP course policy says AI tools may be used for pre-class topic background, prohibits GenAI use in in-class activity, and expects original work for assessment. Use this plan as study guidance; have the team implement and explain its own changes and check with the instructor if AI-assisted code is included in an assessed submission.
