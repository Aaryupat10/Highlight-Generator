# Ranked commentary highlights

Source: `2015-09-12 Crystal Palace 0 - 1 Manchester City`

Timestamps are estimated from ASR and should be checked before video editing.
Hybrid ranking combines event rules, SoccerNet weak-event supervision, existing human review labels, and explicitly marked assistant text-only pseudo-labels in a project-trained two-tier hierarchy. Scores are exploratory, not calibrated probabilities; assistant annotations are not human ground truth.

## 1. Chance Or Shot — 00:01:23 to 00:01:26
Hybrid rank score: 0.81
Peak: 00:01:23 | Candidate segments: 1 | 30-second buildup segments: 10

Event frame (project-written patterns): action=shoot; actor=unresolved; outcomes=none detected.

Event relatedness: 0.84 | Supervised highlight score: 0.85

Event text evidence: shot

Emphasis cues: 0 exclamation marks; caps ratio 0.08; all-caps words 0; heuristic excitement 0.05 (not a trained head).

Experimental model family: goal (0.58); disagrees with the rule-based event label `chance_or_shot`. Treat this head as experimental.

Top temporal context attended by the sequence model: 00:01:28 (0.01), 00:01:37 (0.01), 00:01:26 (0.01)

Attention mass on the 30-second pre-candidate buildup: 0.12 (model attention is a diagnostic, not causal evidence).

The high shot to the front of the Crystal Palace.
