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
    prior_people = {mention["canonical"] for prior in previous_texts for mention in find_mentions(prior, aliases) if mention["type"] == "player"}
    pronouns = re.findall(r"\b(he|him|his|she|her|hers|they|them|their)\b", text, re.I)
    antecedent = next(iter(prior_people)) if len(prior_people) == 1 and pronouns else None
    outcome_patterns = (
        "saved", "blocked", "cleared", "over the bar", "off the post", "wide", "scores", "scored", "finds the net",
    )
    low = text.casefold()
    outcomes = [phrase for phrase in outcome_patterns if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", low)]
    action = frame.get("event_type", "routine")
    frame.update({
        "team_mentions": [item["canonical"] for item in mentions if item["type"] == "team"],
        "player_mentions": [item["canonical"] for item in mentions if item["type"] == "player"],
        "entity_mentions": mentions,
        "coreference": {"pronouns": pronouns, "antecedent_candidate": antecedent, "method": "verified-alias, short-window heuristic" if antecedent else "unresolved", "confidence": "low" if antecedent else "none"},
        "event_frame": {"action_type": action, "actor_candidates": [item["canonical"] for item in mentions if item["type"] == "player"], "team_candidates": [item["canonical"] for item in mentions if item["type"] == "team"], "outcome_cues": outcomes, "method": "lexical frame; candidate arguments only"},
    })
    return frame
