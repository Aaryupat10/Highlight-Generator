# Event extraction configs

`extract_events.py` contains no football. Every word it looks for comes from a JSON file in this
folder, `football.json` by default. To run the same step on another sport, copy the file, edit
it, and pass it with `--config`:

    python "Event extraction/extract_events.py" --match-id 93323 --config "Event extraction/configs/basketball.json"

## What each key does

| key | what it holds |
|---|---|
| `sport` | the name, printed nowhere important |
| `classes` | highlight classes, highest priority first (football: goal, card, near_miss, foul, routine). The last one is the "nothing special" class. Stage 7 trains its event-type head on these. |
| `events` | the event types (see below) |
| `suppress` | `{"goal": ["shot"]}` means a shot in the same sentence as a goal is the same moment, so only the goal is kept |
| `outcome_from_event` | `{"save": "saved"}` means a shot followed by a save in the sentence gets the outcome `saved` |
| `methods` | how it was done: label + regex (header, left foot, volley ...) |
| `outcomes` | what happened: label + regex + optional `class` + `for` (the event types it applies to). `class: near_miss` lifts an `upgradable` event (a shot that went wide) to that class |
| `sources` | where it came from: label + regex + `for`. A group named `who` (or `who2`) captures the player who passed or crossed |
| `modality` | words just before a trigger that say it did not really happen: `negated`, `appeal`, `unrealised`, `attempted`, `hypothetical`. First match wins |
| `context_markers` | regexes for lines that report stats, previews or past games. Events on those lines are kept but their class is set to the last class |
| `live_markers` | case-sensitive regexes that keep a line live even when a context word appears (the all-caps lead of a live call) |
| `prematch_end` | the first line matching this marks the start of play. Everything before it is `prematch` |
| `recap_start` / `recap_end` | after "HALF-TIME" the next `max_lines` lines summarise the past (`recap`) until a line matches `recap_end` |
| `clause_breakers`, `non_subject_prepositions`, `new_subject_markers`, `max_gap_words`, `possessive_pronouns` | how far a player name can be from the trigger and still be its subject |
| `method_skip_events` | event types that have no "method" (a card, a substitution) |

## An event type

    {"name": "yellow_card", "class": "card", "weight": 80, "merge": true,
     "triggers": ["\\byellow card\\b", "\\bbooked\\b"],
     "exclude":  ["\\balready\\b"],
     "patterns": ["shows?\\s+(?P<agent>@P)\\s+a yellow"],
     "agent": ["subject", "prep:for", "object"], "other": [],
     "team": ["agent_team", "subject_team", "any_team"]}

- `triggers` are the words that make the event. `exclude` drops a trigger when the 25 characters
  before or 15 after it match (for "bottom corner" and "penalty area").
- `weight` decides which trigger wins when two overlap, and which event is the primary one of a line
  when two have the same class.
- `patterns` are optional and entity-aware. `@P` stands for a player mention, `@T` a team, `@PS` a
  possessive player ("his" becomes "Haaland's"), `@E` either. Groups named `agent` and `other` fill
  those roles and win over the strategies below.
- `agent`, `other` and `team` are strategies, tried in order:
  - `subject` nearest player before the trigger in the same clause
  - `prep:by`, `prep:for`, `prep:from` ... a player right after that word ("saved by Pope")
  - `object` the first player just after the trigger
  - `any` the first player in the sentence not already used
  - `headline` the first player of the next sentence, when the trigger's own sentence is only "GOAL!"
  - `agent_team`, `subject_team`, `for_team`, `any_team` (team only)
- `merge: true` collapses "GOAL!" and "DIAZ SCORES!" into one goal.
- `upgradable: true` lets an outcome with a `class` (saved, wide) lift the event to that class.
- `unrealised_type` turns "almost scored" / "tries to curl it in" into this other type (a chance).
- `appeal_class` is the class when it was only claimed ("Forest want a penalty").
- `default_outcome` is used when no outcome word is found.

Test a change on one match, read `outputs/events/events_<id>.txt`, and look at the lines it got wrong.
