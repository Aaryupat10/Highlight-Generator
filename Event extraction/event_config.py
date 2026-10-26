"""Per-sport settings for the event extraction step (Stage 4).

Everything that is specific to one sport (which words mean a goal, a card, a shot, what
counts as a method or an outcome, which lines are match context rather than live play)
lives in a JSON file under configs/. extract_events.py reads it through load_config(), so
supporting another sport means writing a new JSON file, not changing code.

Regexes in the file may use these placeholders, which stand for an entity mention that the
coreference stage found (the extractor swaps every mention for a short token first):

    @P   a player / person mention          @T   a team mention
    @PS  a possessive player mention ("his" -> "Haaland's")      @E   either of @P / @T

so "(?P<agent>@P) is booked" means "a player, call him the agent, is booked".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "football.json"


def expand(pattern: str) -> str:
    # one pass, longest placeholder first, so the text a placeholder expands to is never expanded again
    table = {"@PS": r"@P\d+'s", "@P": r"@P\d+", "@T": r"@T\d+", "@E": r"@[PT]\d+"}
    return re.sub(r"@PS|@P|@T|@E", lambda m: table[m.group(0)], pattern)


def rx(pattern: str | None, flags: int = re.I) -> re.Pattern | None:
    return re.compile(expand(pattern), flags) if pattern else None


def rx_list(patterns: list[str] | None) -> list[re.Pattern]:
    return [rx(p) for p in (patterns or [])]


@dataclass
class EventType:
    name: str
    cls: str                       # highlight class: goal / card / near_miss / foul / routine
    weight: int                    # higher wins when two triggers overlap or compete for "primary"
    triggers: list[re.Pattern]
    exclude: list[re.Pattern]      # drop the trigger when its surroundings match (e.g. "bottom corner")
    patterns: list[re.Pattern]     # entity-aware patterns with named groups agent / other
    agent: list[str]               # strategies tried in order: subject, by, headline, object, any
    other: list[str]
    team: list[str]
    upgradable: bool               # an outcome like "saved" can lift the class to near_miss
    unrealised_type: str | None    # becomes this type when it only almost happened ("almost scored" -> chance)
    default_outcome: str | None
    merge: bool                    # same-type events in one line collapse into the best one
    appeal_class: str | None = None   # class when it was only claimed ("want a penalty")


@dataclass
class EventConfig:
    sport: str
    path: Path
    classes: list[str]                       # highest priority first
    possessive_pronouns: frozenset
    max_gap_words: int
    short_sentence_words: int
    clause_breakers: re.Pattern
    non_subject_prepositions: re.Pattern     # "to", "for", "by" ... right before a mention
    context_markers: list[re.Pattern]
    live_markers: list[re.Pattern]
    modality: list[tuple[str, re.Pattern]]   # (label, regex on the text before the trigger)
    events: list[EventType]
    suppress: dict                           # {"goal_disallowed": ["goal"]}
    outcome_from_event: dict                 # {"save": "saved"}
    methods: list[tuple[str, re.Pattern]]
    outcomes: list[dict]                     # {label, regex, class}
    sources: list[dict]                      # {label, regex, for}
    new_subject: re.Pattern | None = None    # ", the defender ..." between a mention and a trigger: a new subject
    method_skip: frozenset = frozenset()     # event types that have no "method"
    recap_start: list[tuple[re.Pattern, int]] = field(default_factory=list)   # ("HALF-TIME", max lines) what follows summarises the past
    recap_end: list[re.Pattern] = field(default_factory=list)     # "... back underway"
    prematch_end: list[re.Pattern] = field(default_factory=list)  # the first line that says play has started
    prematch_search_lines: int = 60
    by_name: dict = field(default_factory=dict)


def load_config(path: Path | str = DEFAULT_CONFIG) -> EventConfig:
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    classes = d["classes"]
    events = []
    for e in d["events"]:
        if e["class"] not in classes:
            raise ValueError(f"event {e['name']}: class {e['class']!r} not in classes {classes}")
        events.append(EventType(
            name=e["name"], cls=e["class"], weight=e.get("weight", 1),
            triggers=rx_list(e["triggers"]), exclude=rx_list(e.get("exclude")),
            patterns=rx_list(e.get("patterns")),
            agent=e.get("agent", ["subject", "by"]), other=e.get("other", []),
            team=e.get("team", ["agent_team", "subject_team", "any_team"]),
            upgradable=e.get("upgradable", False), unrealised_type=e.get("unrealised_type"),
            default_outcome=e.get("default_outcome"), merge=e.get("merge", False),
            appeal_class=e.get("appeal_class")))
    cfg = EventConfig(
        sport=d["sport"], path=path, classes=classes,
        possessive_pronouns=frozenset(w.lower() for w in d.get("possessive_pronouns", [])),
        max_gap_words=d.get("max_gap_words", 8),
        short_sentence_words=d.get("short_sentence_words", 3),
        clause_breakers=rx(d.get("clause_breakers", r"\b(?:but|though|although|however|while|whereas)\b|[;:—]|\s-\s")),
        non_subject_prepositions=rx(r"\b(?:%s)\s+(?:the\s+|a\s+|an\s+)?$" % "|".join(d.get("non_subject_prepositions", ["to", "for", "by", "from"]))),
        context_markers=rx_list(d.get("context_markers")),
        live_markers=[rx(p, 0) for p in d.get("live_markers", [])],     # case-sensitive on purpose
        modality=[(m["label"], rx(m["regex"])) for m in d.get("modality", [])],
        events=events, suppress=d.get("suppress", {}),
        outcome_from_event=d.get("outcome_from_event", {}),
        methods=[(m["label"], rx(m["regex"])) for m in d.get("methods", [])],
        outcomes=[{"label": o["label"], "regex": rx(o["regex"]), "class": o.get("class"),
                   "for": set(o.get("for", ["*"]))} for o in d.get("outcomes", [])],
        sources=[{"label": s["label"], "regex": rx(s["regex"]), "for": set(s.get("for", ["*"]))}
                 for s in d.get("sources", [])],
        new_subject=rx(d.get("new_subject_markers")),
        method_skip=frozenset(d.get("method_skip_events", [])),
        recap_start=[(rx(r["regex"], 0), r.get("max_lines", 12)) for r in d.get("recap_start", [])],
        recap_end=rx_list(d.get("recap_end")),
        prematch_end=rx_list(d.get("prematch_end")),
        prematch_search_lines=d.get("prematch_search_lines", 60))
    cfg.by_name = {e.name: e for e in events}
    return cfg
