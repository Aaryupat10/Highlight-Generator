# Leicester vs Arsenal: goal recovery check

The full first-half run processed French commentary with Whisper small and NLLB English translation. Its first report had 13 candidates, none at the three first-half goals listed in the matching `Labels-v2.json`. It also ranked “the 10th goal for the Foxes this season” as a live goal, although that is a season statistic.

I updated only the phrase rules and reranked the saved transcript. The refined output has 14 candidates and now includes all three annotated goal moments:

| Video/caption event time | Refined candidate time | Offset | Commentary cue |
|---|---|---:|---|
| 00:12:11 | 00:12:13 | +2 sec | “he'll score” |
| 00:17:32 | 00:17:23–00:17:30 | −9 sec | NLLB rendered the equalizer commentary as “equalization” |
| 00:32:52 | 00:32:55 | +3 sec | “first goal of the season,” followed by “Arsenal to lead” in nearby commentary |

The corresponding video sequences show the scoring action/celebration. This is a useful debugging result: adding live scoring phrases and local context recovered the goals, while filtering ordinal shot counts and “10th goal this season” removed two obvious false candidates. It is not an independent model evaluation: Leicester event labels were included in the classifier's training data, and the new changes were checked on this match. Many other candidates still need review.

The exact generated report is `highlights.md` in this folder; it retains original French and English text together.
