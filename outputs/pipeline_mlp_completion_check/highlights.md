# Ranked commentary highlights

Source: `Crystal Palace vs Manchester City (saved transcript)`

Timestamps are estimated from ASR and should be checked before video editing.
Hybrid ranking combines event phrase evidence with separate SoccerNet event-alignment labels and human-reviewed highlight labels. Components and final rank are exploratory scores, not calibrated probabilities.

## 1. Chance Or Shot — 00:01:23 to 00:01:26
Hybrid rank score: 0.76
Peak: 00:01:23 | Candidate segments: 1 | 30-second buildup segments: 10

Event frame (project-written patterns): action=shoot; actor=unresolved; outcomes=none detected.

Event relatedness: 0.54 | Supervised highlight score: 0.91

Event text evidence: shot

Emphasis cues: 0 exclamation marks; caps ratio 0.08; all-caps words 0; heuristic excitement 0.05 (not a trained head).

Experimental model family: shot_or_chance (0.68); disagrees with the rule-based event label `chance_or_shot`. Treat this head as experimental.

Model feature evidence (removing these words lowered this candidate's score): a very, high shot, he asks, the high, the crystal, to stop

The high shot to the front of the Crystal Palace.

## 2. Corner Or Free Kick — 00:02:50 to 00:02:52
Hybrid rank score: 0.53
Peak: 00:02:50 | Candidate segments: 1 | 30-second buildup segments: 14

Event frame (project-written patterns): action=corner; actor=unresolved; outcomes=none detected.

Event relatedness: 0.42 | Supervised highlight score: 0.64

Event text evidence: corner

Emphasis cues: 0 exclamation marks; caps ratio 0.00; all-caps words 0; heuristic excitement 0.00 (not a trained head).

Experimental model family: shot_or_chance (0.47); disagrees with the rule-based event label `corner_or_free_kick`. Treat this head as experimental.

Model feature evidence (removing these words lowered this candidate's score): pallas, no time, corner, the corner, power, perfect

who arrived there at the rematch in the corner.
