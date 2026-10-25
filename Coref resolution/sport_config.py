"""Per-sport settings for the coreference step.

Everything that is specific to one sport (how lineups are written, role words,
the "scorer line" rule, pronouns, team naming) lives in a JSON file under
configs/. name_book.py and resolve.py read it through load_config(), so
supporting another sport means writing a new JSON file, not changing code.

    python "Coref resolution/name_book.py" --config "Coref resolution/configs/football.json"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "football.json"

# matches 's, ’s, or a bare ' after a mention ("the hosts'", "the striker's")
POSS = r"(?:'s|’s|')?"


@dataclass
class SportConfig:
    sport: str
    path: Path

    # --- name book (name_book.py) ---
    lineup_re: re.Pattern | None          # "TEAM (4-2-3-1): player, player, ..."
    subs_re: re.Pattern | None            # "SUBS: player, player, ..."
    header_suffix_re: re.Pattern | None   # "ARSENAL STARTING XI" -> "ARSENAL"
    first_lineup_is_home: bool
    roles_from_formation: bool            # derive GK/DEF/MID/FWD from the formation
    surname_particles: frozenset          # "de", "van", ... stay attached to a surname
    distinctive_last_words: frozenset     # "City", "United" work as a short team name
    generic_first_words: frozenset        # "West" is too generic to be a short team name
    first_word_abbreviations: dict        # {"Manchester": "Man"} gives "Man City"
    teams_file: Path | None               # team header keys and nicknames

    # --- resolver (resolve.py) ---
    player_pronouns: frozenset
    team_pronouns: frozenset
    possessive_pronouns: frozenset        # these get "'s" when replaced ("his" -> "Haaland's")
    reflexive_pronouns: frozenset         # left alone ("himself")
    roles: dict                          # role word -> set of roster roles it can mean
    home_words: frozenset
    away_words: frozenset
    scorer_line_re: re.Pattern | None     # line after which he/him/his is the scorer
    history_lines: int
    ambiguous_lines: int
    default_rules: str

    # --- compiled from the values above ---
    role_re: re.Pattern | None = None
    side_re: re.Pattern | None = None
    find_pronoun_re: re.Pattern | None = None
    find_phrase_re: re.Pattern | None = None


def _regex(value: str | None, flags: int = 0) -> re.Pattern | None:
    return re.compile(value, flags) if value else None


def _alternation(words) -> str:
    return "|".join(re.escape(w) for w in words)


def load_config(path: Path | str | None = None) -> SportConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"sport config not found: {path}")
    nb, rs = raw.get("name_book", {}), raw.get("resolver", {})

    teams_file = nb.get("teams_file")
    cfg = SportConfig(
        sport=raw.get("sport", path.stem),
        path=path,
        lineup_re=_regex(nb.get("lineup_regex")),
        subs_re=_regex(nb.get("subs_regex")),
        header_suffix_re=_regex(nb.get("header_suffix_regex")),
        first_lineup_is_home=bool(nb.get("first_lineup_is_home", True)),
        roles_from_formation=bool(nb.get("roles_from_formation", False)),
        surname_particles=frozenset(nb.get("surname_particles", [])),
        distinctive_last_words=frozenset(nb.get("distinctive_last_words", [])),
        generic_first_words=frozenset(nb.get("generic_first_words", [])),
        first_word_abbreviations=dict(nb.get("first_word_abbreviations", {})),
        teams_file=(path.parent / teams_file) if teams_file else None,
        player_pronouns=frozenset(rs.get("player_pronouns", [])),
        team_pronouns=frozenset(rs.get("team_pronouns", [])),
        possessive_pronouns=frozenset(rs.get("possessive_pronouns", [])),
        reflexive_pronouns=frozenset(rs.get("reflexive_pronouns", [])),
        roles={word: frozenset(r) for word, r in rs.get("roles", {}).items()},
        home_words=frozenset(rs.get("home_words", [])),
        away_words=frozenset(rs.get("away_words", [])),
        scorer_line_re=_regex(rs.get("scorer_line_regex")),
        history_lines=int(rs.get("history_lines", 6)),
        ambiguous_lines=int(rs.get("ambiguous_lines", 10)),
        default_rules=rs.get("default_rules", "side,goal,role"),
    )

    role_words = _alternation(cfg.roles)                     # keeps the order of the JSON file
    side_words = _alternation(sorted(cfg.home_words | cfg.away_words, key=lambda w: (-len(w), w)))
    pronouns = _alternation(sorted(cfg.player_pronouns | cfg.team_pronouns, key=lambda w: (-len(w), w)))

    if role_words:
        cfg.role_re = re.compile(rf"^(?:the|The)\s+(?:(\S+)\s+)?({role_words}){POSS}$")
    if side_words:
        cfg.side_re = re.compile(rf"^(?:the|The|their|Their)\s+({side_words}){POSS}$")
    if pronouns:
        cfg.find_pronoun_re = re.compile(rf"\b(?:{pronouns})\b", re.I)

    # the same phrases, searched inside a line (finds mentions the model skipped)
    phrases = []
    if side_words:
        phrases.append(rf"\b(?:[Tt]he|[Tt]heir)\s+(?:{side_words}){POSS}(?!\w)")
    if role_words:
        phrases.append(rf"\b[Tt]he\s+(?:[A-Z][\w-]*\s+)?(?:{role_words}){POSS}(?!\w)")
    if phrases:
        cfg.find_phrase_re = re.compile("|".join(phrases))
    return cfg
