# Manual transcript labels

Open `manual_annotation_context.csv` in Numbers or Excel. It shows about 30 seconds of surrounding commentary per row, rather than one short transcript fragment. Each teammate can label 20 rows; keep the commentary and timestamp columns unchanged.

Fill `human_label` with exactly one of:

- `highlight`: the commentary at this timestamp describes a real, notable moment in this match, such as a goal, penalty, dangerous attempt/save, red card, or another clearly exciting turning point.
- `routine`: ordinary play, a generic mention of a shot/corner/foul, a past-match recap, a hypothetical/future event, or text that is not a notable moment.
- `unclear`: the transcript is too garbled or lacks enough context to decide. These rows will be excluded from training.

Optionally fill `event_type` with `goal`, `chance_or_save`, `card_or_penalty`, `other`, or leave it blank. Put your initials in `annotator`. If a segment needs nearby context, add a note; do not change its timestamp.

The 60 rows are sampled from six matches: some are near a SoccerNet event and some are far from one. That sampling information is intentionally not shown as a label. Judge each commentary segment yourself; proximity alone does not make it a highlight.

## Independent Crystal Palace review batch

For the next check, use `crystal_palace_holdout_review.csv`. It contains 14 shuffled windows from the Crystal Palace–Manchester City first half: seven generated candidates and seven control moments. The model checkpoint used to generate candidates was trained without Crystal Palace labels. Each person has 4–5 rows assigned in the `assignee` column. The separate `outputs/heldout_crystal_palace/review_key.csv` identifies candidate/control rows; keep that file aside until all three reviewers finish to reduce label bias.

For each row, open the listed video and inspect roughly 15 seconds before and after the timestamp. Label the full window:

- `human_label`: `highlight` only when a real, notable moment appears in that window; `routine` for ordinary play or an unremarkable event; `unclear` if the commentary or footage is inconclusive.
- `event_type`: `goal`, `chance_or_save`, `card_or_penalty`, `corner_or_free_kick`, `throw_in`, `foul`, `routine_play`, or `unclear`. Record what the footage shows, even when the commentary translation names a different event.
- `annotator`: your name or initials; use `notes` for translation mistakes, timing offsets, or disagreements.

Do not change the transcript, match, timestamp, or video path. This batch can measure whether candidates correspond to reviewed moments and whether controls are routine. It is still only one match and does not establish performance across all leagues or languages.
