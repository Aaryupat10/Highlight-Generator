# Manual transcript labels

Open `manual_annotation_context.csv` in Numbers or Excel. It shows about 30 seconds of surrounding commentary per row, rather than one short transcript fragment. Each teammate can label 20 rows; keep the commentary and timestamp columns unchanged.

Fill `human_label` with exactly one of:

- `highlight`: the commentary at this timestamp describes a real, notable moment in this match, such as a goal, penalty, dangerous attempt/save, red card, or another clearly exciting turning point.
- `routine`: ordinary play, a generic mention of a shot/corner/foul, a past-match recap, a hypothetical/future event, or text that is not a notable moment.
- `unclear`: the transcript is too garbled or lacks enough context to decide. These rows will be excluded from training.

Fill `event_type` with `goal`, `chance_or_save`, `card_or_penalty`, `corner_or_free_kick`, `other`, `routine_play`, or `unclear`. To collect supervision for the remaining pipeline heads, also fill:

- `excitement_label`: `low`, `medium`, `high`, or `unclear`, based on the commentator's emphasis in the text/recording (do not infer from event type alone).
- `focal_player`: a clearly named player central to the described event; otherwise leave blank. Use the name spelling visible in the commentary and add a canonical spelling in `notes` if known.
- `pronoun_link`: if a pronoun clearly refers to a player in the preceding context, write the exact link, for example `he=Kevin De Bruyne` or `his=Erling Haaland`; otherwise leave blank. Do not write only `pronoun=player`.
- `role_action`: short verb/action expressed in this window (for example `shoot`, `score`, `save`, `foul`, `cross`, `book`, `send_off`).
- `role_target`: the explicit goal/ball/player/team target if stated; otherwise leave blank.

Put your initials in `annotator`. If a segment needs nearby context, add a note; do not change its timestamp. Never guess a name or role. These extra fields are distinct human labels and should be merged into training only after review; blanks mean unknown, not negative.

The 60 rows are sampled from six matches: some are near a SoccerNet event and some are far from one. That sampling information is intentionally not shown as a label. Judge each commentary segment yourself; proximity alone does not make it a highlight.

## Blinded review batch 01

`human_review_batch_01.csv` is a blank 120-row human annotation queue balanced by match-half and sampled from timestamp-aligned event and routine windows. It assigns 40 rows to each of three teammates in the `assignee` column; assignments are distributed across all match-halves. The sampling source label is not included. Review each row using the same `highlight`, `routine`, or `unclear` definitions above. When video is available, use it to verify whether the commentary describes a real event at that timestamp. Fill `event_type` separately from the highlight decision, and enter annotator initials. Leave timestamps, match IDs, and commentary unchanged. Do not use this queue as training data until human labels are completed and checked.

`assistant_text_annotations.csv` is a separate text-only draft completed by Codex. Every row says `assistant_text_only` and warns that it was not verified against audio/video. It can be used as a weak training experiment, but it is not human ground truth and should not replace independent teammate labels. Excitement there means wording/punctuation only, not vocal intensity.

Rebuild the queue with `python3 src/create_annotation_queue.py --per-class 5`. The queue includes columns for highlight value, event family, excitement, player/coreference, and event roles. Change the random seed only if you need a different reproducible sample.

## Independent Crystal Palace review batch

For the next check, use `crystal_palace_holdout_review.csv`. It contains 14 shuffled windows from the Crystal Palace–Manchester City first half: seven generated candidates and seven control moments. The model checkpoint used to generate candidates was trained without Crystal Palace labels. Each person has 4–5 rows assigned in the `assignee` column. The separate `outputs/heldout_crystal_palace/review_key.csv` identifies candidate/control rows; keep that file aside until all three reviewers finish to reduce label bias.

For each row, open the listed video and inspect roughly 15 seconds before and after the timestamp. Label the full window:

- `human_label`: `highlight` only when a real, notable moment appears in that window; `routine` for ordinary play or an unremarkable event; `unclear` if the commentary or footage is inconclusive.
- `event_type`: `goal`, `chance_or_save`, `card_or_penalty`, `corner_or_free_kick`, `throw_in`, `foul`, `routine_play`, or `unclear`. Record what the footage shows, even when the commentary translation names a different event.
- `annotator`: your name or initials; use `notes` for translation mistakes, timing offsets, or disagreements.

Do not change the transcript, match, timestamp, or video path. This batch can measure whether candidates correspond to reviewed moments and whether controls are routine. It is still only one match and does not establish performance across all leagues or languages.
