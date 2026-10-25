# Sport configs for coreference resolution

`name_book.py` and `resolve.py` contain no sport-specific words or patterns. Everything
that depends on the sport is read from one JSON file here. Football is the default.

```bash
# football (default, same as --config "Coref resolution/configs/football.json")
python "Coref resolution/name_book.py" data/23_24_commentary_clean.csv
python "Coref resolution/resolve.py" --match-number 1 --stage C

# another sport: copy football.json, edit it, and pass it to both scripts
python "Coref resolution/name_book.py" data/cricket_clean.csv --config "Coref resolution/configs/cricket.json"
python "Coref resolution/resolve.py" --match-number 1 --stage C --config "Coref resolution/configs/cricket.json"
```

## Which file does what

| File | Role |
|---|---|
| `name_book.py` | Builds each match's name book from the lineup lines |
| `run_coref.py` | Stage A (baseline): runs the pretrained model, saves its clusters, names each cluster after its most common proper name. Also holds the helpers B and C reuse |
| `resolve.py` | Stages B and C: name book matching and voting (B), then the sport rules (C), driven by the sport config |
| `sport_config.py` | Loads the sport config |

Keys starting with an underscore are comments. Regexes are Python regexes written as JSON
strings, so every backslash is doubled (`\\s`, not `\s`).

## `name_book` section (how rosters are read out of the commentary)

| Key | What it does |
|---|---|
| `lineup_regex` | Matches a lineup line. Needs named groups `team` and `body`; a `formation` group is optional. `null` = no lineup lines in this sport. |
| `subs_regex` | Matches a substitutes line (named groups `team` optional, `body`). `null` = none. |
| `header_suffix_regex` | Suffix to strip from a team header ("ARSENAL STARTING XI" becomes "ARSENAL"). |
| `first_lineup_is_home` | `true` if the first lineup in the text is the home side. `false` leaves the side unknown, so the hosts/visitors rule does nothing. |
| `roles_from_formation` | `true` derives GK/DEF/MID/FWD from a formation like 4-2-3-1. `false` leaves roles empty. |
| `surname_particles` | Words that stay attached to a surname ("de", "van", "al"). |
| `distinctive_last_words` | Last words that work as a short team name ("City", "United"). |
| `generic_first_words` | First words too generic to be a short name ("West", "Real"). |
| `first_word_abbreviations` | `{"Manchester": "Man"}` makes "Man City" an alias of "Manchester City". |
| `teams_file` | Team list (relative to this folder): how each team is written in the lineup header, plus nicknames. |

## `resolver` section (words and rules used by stages B and C)

| Key | What it does |
|---|---|
| `player_pronouns`, `team_pronouns` | Pronouns resolved to a player or to a team. |
| `possessive_pronouns` | Pronouns that get `'s` when replaced ("his" becomes "Haaland's"). |
| `reflexive_pronouns` | Pronouns left alone ("himself"). |
| `roles` | Role word to the roster roles it can mean, e.g. `"striker": ["FWD"]`. `{}` turns the role rule off. |
| `home_words`, `away_words` | Phrases meaning the home or away side ("the hosts", "the visitors"). Empty lists turn the side rule off. |
| `scorer_line_regex` | A line after which the next player pronoun is the scorer (`^GOAL\\W` in football). `null` turns the goal rule off. |
| `history_lines` | How many lines back the role and recency rules look. |
| `ambiguous_lines` | How many lines back to look to settle an ambiguous name like "Gomes". |
| `default_rules` | Stage C rules used when `--rules` is not given (from `side`, `goal`, `role`, `recency`). |

## Adding a sport

1. Copy `football.json` to `<sport>.json` and set `sport`.
2. Set `lineup_regex` and `subs_regex` to `null` if the commentary has no lineup lines.
3. Replace `roles`, `home_words`, `away_words` and `scorer_line_regex` with that sport's wording.
4. Add a teams file if the sport has team nicknames, or set `teams_file` to `null`.
5. Run both scripts with `--config`.

## Known limits

- Without lineup lines the name book is empty, and stages B and C then cannot name a
  cluster that has no name book entry. Those mentions come out unresolved.
- Women's sports and tennis need she/her added to `player_pronouns` (and `herself` to
  `reflexive_pronouns`). Leave "her" out of `possessive_pronouns` unless you handle that it
  is also an object pronoun ("passes to her").
- The preprocessing step (`Preprocessing/prepare_event_text.py`) still has football's
  `pen` to `penalty` expansion hard-coded. It is not covered by these configs.
