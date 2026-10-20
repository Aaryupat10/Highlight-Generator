# Crystal Palace vs Manchester City: TF-IDF demo visual audit

The output was checked against sampled frames from `1_224p.mkv`. This is a manual spot check, not a full replay or independent model evaluation. The model's score is a ranking score, and it does not use video frames.

| Generated timestamp | Model output | What sampled footage supports | Review |
|---|---|---|---|
| 00:01:26–00:01:28 | Chance or shot; “First chance at Joe Hart's frame” | The surrounding sequence shows Palace attacking down the right near the penalty area and an attempt/challenge, followed by a close-up. The sampled frames do not establish whether it was on target. | Plausible chance candidate. The event may have happened a few seconds before the reported timestamp. |
| 00:02:50–00:02:52 | Corner or free kick; “...at the corner kick” | At about 00:02:52, Martin Kelly takes a throw-in. The broader 00:02:50–00:03:06 context then continues into Palace attacking play, with a cross/ball into the area and Joe Hart collecting. | The exact `corner_or_free_kick` label is wrong or mistimed, but the wider attacking sequence may still be relevant. The three-minute demo ended at 00:03:00, truncating this longer passage. |

## What this check changes

- Keep the first item as a plausible chance candidate; verify the precise action before describing it as a shot on target.
- Do not call the second item a corner. Its wider context contains an attack, so judge highlight quality separately from the event-type mismatch.
- Record event type separately from highlight/routine status. The 30-second human-reviewed window for this passage was previously labeled as a highlight, but its event type should be corrected after video review.
- Improve evaluation with full event windows and timestamp tolerance. A short Whisper segment can place a phrase at the end of the action or several seconds away from it.
- The current system has no visual model. Its TF-IDF MLP cannot see the throw-in, shot, or ball location; it ranks transcript contexts after phrase rules propose candidates.
