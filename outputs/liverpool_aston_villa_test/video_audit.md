# Liverpool vs Aston Villa: candidate audit

This is a manual visual check of the 11 generated candidates, compared with the match's SoccerNet `Labels-v2.json`. SoccerNet labels mark event moments such as shots and corners; they are not subjective highlight judgments. The event-context model was trained using this match's event annotations, so this is an error analysis, not an independent test.

## Important correction

The first Liverpool goal is annotated at **00:01:03**. The footage shows the attack at about 00:01:03 and the 1–0 score and celebration shortly afterward. The generated candidate at **00:01:35–00:01:51** is later live play and does **not** capture that goal. The previous claim that candidate 11 was the goal was incorrect.

## Generated candidates

| Candidate time | Model label | Nearest annotated event | Visual check |
|---|---|---|---|
| 00:01:35–00:01:51 | Corner or free kick | Goal at 00:01:03, 33 sec earlier | Live play after the score had become 1–0; not the goal. False positive for the first-goal event. |
| 00:14:53–00:15:00 | Corner or free kick | Corner at 00:15:09, 16 sec later | Corner setup and delivery are visible; likely event-related. |
| 00:15:52–00:15:54 | Chance or shot | Corner at 00:16:31, 39 sec later | Close-ups rather than a clear shot in sampled frames; likely weak candidate. |
| 00:19:58–00:20:00 | Chance or shot | Shot off target at 00:19:53, 5 sec earlier | Attack near the penalty area; event timing agrees. |
| 00:26:21–00:26:23 | Corner or free kick | Corner at 00:27:19, 58 sec later | Sampled frames do not show a clear set piece at the candidate time; likely weak. |
| 00:34:50–00:35:00 | Chance or shot | Shot at 00:36:36, 106 sec later | No matching annotated shot near the candidate; sampled play is not clearly a highlight. |
| 00:36:54–00:36:56 | Chance or shot | Corner at 00:37:08, 14 sec later; shot at 00:36:36, 18 sec earlier | Attacking play near goal is visible; likely part of the nearby event sequence. |
| 00:37:12–00:37:16 | Corner or free kick | Corner at 00:37:08, 4 sec earlier | Corner sequence is visible and timing agrees. |
| 00:38:14–00:38:16 | Chance or shot | Nearest listed event is corner at 00:37:28, 46 sec earlier | Sampled frames show a player down and play continuing; no clear shot at the candidate time. Likely weak. |
| 00:40:27–00:40:29 | Chance or shot | Shot on target at 00:40:18, 9 sec earlier | Attack toward goal and goalkeeper involvement are visible; timing agrees. |
| 00:42:34–00:42:36 | Chance or shot | Shot on target at 00:42:32, 2 sec earlier | Attack toward goal and goalkeeper collecting the ball are visible; timing agrees. |

The nearest-event comparisons above are approximate and do not prove the commentary describes that event. They are useful for finding likely hits and misses for human review.

By a simple ±30-second match to any listed event, **6 of the 11 candidates** are near an annotated event; two candidates are near the same 37-minute event sequence. This is an event-alignment check, not highlight precision. The annotated 00:01:03 goal is still missed.

## What this shows

- The output includes several candidates close to annotated shots or corners, but also weak candidates.
- The first-half goal is a major miss: the report does not rank the actual goal moment.
- The Russian transcript and English translation around the opening goal are garbled, which likely prevented event phrase matching.
- Near the missed goal, Whisper produced the Russian phrase `с голой Леверпуля`; NLLB rendered it as “with the naked face of Liverpool.” The intended goal wording was lost before candidate matching.
- I also scored the transcript contexts with a diagnostic MLP trained without Liverpool. The context around the missed goal scored 0.569, while the later non-goal candidate scored 0.608. The binary n-gram model does not separate these well enough to safely propose candidates on its own.
- The classifier reranks only moments proposed by phrase matching. It cannot recover an event that candidate generation misses.
- Improve multilingual event recall and transcript quality first; then tune the reranker using human-reviewed, match-held-out examples. Do not report these same-match comparisons as independent model accuracy.
