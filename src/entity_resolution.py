"""Dictionary-based team/player mentions and cautious local coreference."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def match_team_names_from_match_path(video: Path) -> dict[str, list[str]]:
    """Get exact full club names from a SoccerNet-style fixture folder name."""
    name = video.parent.name
    score = re.search(r"\s(\d{1,2})\s-\s(\d{1,2})\s", name)
    if not score:
        return {}
    home = re.sub(r"^\d{4}-\d{2}-\d{2}\s+-\s+\d{1,2}-\d{2}\s+", "", name[:score.start()]).strip()
    away = name[score.end():].strip()
    return {team: [team] for team in (home, away) if team}


def load_entity_aliases(path: Path | None, video: Path) -> dict[str, dict[str, str]]:
    """Combine exact team names in the fixture path with optional verified aliases JSON."""
    entities: dict[str, dict[str, str]] = {"team": {}, "player": {}}
    for canonical, aliases in match_team_names_from_match_path(video).items():
        for alias in aliases:
            entities["team"][alias] = canonical
    if path is None:
        return entities
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for kind in ("teams", "players"):
        entity_type = "team" if kind == "teams" else "player"
        for canonical, aliases in (data.get(kind) or {}).items():
            for alias in [canonical, *(aliases or [])]:
                clean = str(alias).strip()
                if clean:
                    entities[entity_type][clean] = str(canonical).strip()
    return entities


def find_mentions(text: str, aliases: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    matches = []
    for kind, table in aliases.items():
        for alias, canonical in table.items():
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
            if pattern.search(text):
                matches.append({"type": kind, "canonical": canonical, "surface": alias})
    # Prefer the longest verified alias, avoiding a short alias eclipsing a full name.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for mention in sorted(matches, key=lambda item: len(item["surface"]), reverse=True):
        key = (mention["type"], mention["canonical"])
        unique.setdefault(key, mention)
    return list(unique.values())


def add_entity_and_coreference_fields(frame: dict[str, Any], text: str, previous_texts: list[str], aliases: dict[str, dict[str, str]]) -> dict[str, Any]:
    mentions = find_mentions(text, aliases)
    # Resolve only against the nearest prior player mention. If two or more
    # distinct players were mentioned in the same nearest context, abstain.
    prior_people: list[str] = []
    for prior in reversed(previous_texts):
        names = list(dict.fromkeys(mention["canonical"] for mention in find_mentions(prior, aliases) if mention["type"] == "player"))
        if names:
            prior_people = names
            break
    pronouns = re.findall(r"\b(he|him|his|she|her|hers|they|them|their)\b", text, re.I)
    antecedent = prior_people[0] if len(prior_people) == 1 and pronouns else None
    low = text.casefold()
    # Small, inspectable event-frame grammar. Roles are deliberately returned
    # as candidates and never as a claim of full statistical SRL.
    actions = (
        ("score", r"\b(?:scores?|scored|scoring|finds? the net|puts? .*? ahead|levels? the score)\b"),
        ("shoot", r"\b(?:shoots?|shot|shooting|fires?|strikes?|heads?)\b"),
        ("save", r"\b(?:saves?|saved|stops?|blocked|parried)\b"),
        ("foul", r"\b(?:fouls?|fouled|brings? down|tackles?)\b"),
        ("book", r"\b(?:booked|cautioned|shown a yellow|yellow card)\b"),
        ("send_off", r"\b(?:sent off|red card|shown red|dismissed)\b"),
        ("cross", r"\b(?:cross(?:es|ed|ing)?|delivery|delivers?)\b"),
        ("corner", r"\b(?:corner|corner kick)\b"),
        ("penalty", r"\b(?:penalty|spot kick)\b"),
    )
    action = next((name for name, pattern in actions if re.search(pattern, low)), "play")
    outcomes = [name for name, pattern in (
        ("goal", r"\b(?:goal|scores?|scored|finds? the net)\b"),
        ("saved", r"\b(?:saved|save|stopped|parried)\b"),
        ("blocked", r"\b(?:blocked|cleared)\b"),
        ("missed", r"\b(?:wide|over the bar|off the post|off the crossbar)\b"),
        ("card", r"\b(?:yellow card|red card|booked|sent off)\b"),
        ("penalty_awarded", r"\b(?:penalty awarded|penalty given)\b"),
    ) if re.search(pattern, low)]
    player_candidates = [item["canonical"] for item in mentions if item["type"] == "player"]
    team_candidates = [item["canonical"] for item in mentions if item["type"] == "team"]
    # Agent attribution only when exactly one verified player is named. The
    # patient/target stays unresolved because broadcast syntax is ambiguous.
    actor = player_candidates[0] if len(set(player_candidates)) == 1 else None
    frame.update({
        "team_mentions": team_candidates,
        "player_mentions": player_candidates,
        "entity_mentions": mentions,
        "coreference": {"pronouns": pronouns, "antecedent_candidate": antecedent, "method": "nearest unique verified player mention within supplied context" if antecedent else "abstained: no unique nearest verified player antecedent", "confidence": "low" if antecedent else "none"},
        "event_frame": {"action_type": action, "actor": actor, "actor_candidates": player_candidates, "team_candidates": team_candidates, "patient_or_target": None, "outcome_cues": outcomes, "method": "project-written event-frame patterns; arguments are conservative candidates, not trained SRL"},
    })
    return frame
