# Commentary2Highlight
## Live prototype result

This run processed the first three minutes of a football match and converted the spoken commentary into a timestamped candidate moment.

### What the system did

1. Extracted audio from the match video.
2. Whisper detected Spanish speech and produced English text with timestamps.
3. Football event phrase matching found a possible shot.
4. A PyTorch event-context model reranked the candidate using the surrounding commentary.

### Generated candidate

| Start | End | Event | Hybrid rank score |
|---|---|---|---:|
| 00:01:23 | 00:01:26 | Chance or shot | 0.59 |

> “The high shot to the front of the Crystal Palace.”

### How to interpret this result

The system automatically selected this commentary passage and attached its estimated time in the source video. The score combines event phrase matching with the trained model's context score. It is a ranking score, **not a probability**. This is a candidate shot, not a confirmed goal or major highlight; the timestamp and translation should be checked against the match.

### Model and evaluation

The event-context model was trained using 365 commentary windows aligned with SoccerNet annotations from six matches. Match-held-out evaluation produced an exploratory average F1 of 0.59 at a 0.5 threshold. These annotations identify football events, so the evaluation measures event-related commentary rather than subjective highlight quality.

### Current project scope

The prototype generates timestamped commentary candidates and exports a transcript and ranked result. It does not yet cut video clips automatically, and results may vary with audio quality, language, translation, and commentary style.
