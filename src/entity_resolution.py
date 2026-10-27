"""Dictionary-based team/player mentions and cautious local coreference."""
from __future__ import annotations

import json
import re
from functools import lru_cache
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


def load_entity_aliases(path: Path | None, video: Path) -> dict[str, dict[str, Any]]:
    """Combine exact team names in the fixture path with optional verified aliases JSON."""
    entities: dict[str, dict[str, Any]] = {"team": {}, "player": {}, "ambiguous": {}}
    alias_candidates: dict[str, set[tuple[str, str]]] = {}
    for canonical, aliases in match_team_names_from_match_path(video).items():
        for alias in aliases:
            alias_candidates.setdefault(alias.casefold(), set()).add(("team", canonical))
    if path is not None:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for kind in ("teams", "players"):
            entity_type = "team" if kind == "teams" else "player"
            for canonical, aliases in (data.get(kind) or {}).items():
                for alias in [canonical, *(aliases or [])]:
                    clean = str(alias).strip()
                    if clean:
                        alias_candidates.setdefault(clean.casefold(), set()).add((entity_type, str(canonical).strip()))
    display_spellings: dict[str, str] = {}
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for kind in ("teams", "players"):
            for canonical, aliases in (data.get(kind) or {}).items():
                for alias in [canonical, *(aliases or [])]:
                    clean = str(alias).strip()
                    if clean:
                        display_spellings.setdefault(clean.casefold(), clean)
    for folded, candidates in alias_candidates.items():
        alias = display_spellings.get(folded, folded)
        if len(candidates) == 1:
            kind, canonical = next(iter(candidates))
            entities[kind][alias] = canonical
        else:
            entities["ambiguous"][alias] = [
                {"type": kind, "canonical": canonical} for kind, canonical in sorted(candidates)
            ]
    return entities


def _alias_signature(aliases: dict[str, dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    rows: set[tuple[str, str, str]] = set()
    for kind in ("team", "player"):
        for alias, canonical in (aliases.get(kind) or {}).items():
            clean = str(alias).strip()
            if clean:
                rows.add((clean.casefold(), kind, str(canonical).strip()))
    for alias, candidates in (aliases.get("ambiguous") or {}).items():
        clean = str(alias).strip()
        for candidate in candidates:
            rows.add((clean.casefold(), str(candidate["type"]), str(candidate["canonical"]).strip()))
    return tuple(sorted(rows))


@lru_cache(maxsize=64)
def _compiled_alias_matcher(signature: tuple[tuple[str, str, str], ...]):
    targets: dict[str, set[tuple[str, str]]] = {}
    spellings: dict[str, str] = {}
    for folded, kind, canonical in signature:
        targets.setdefault(folded, set()).add((kind, canonical))
        spellings.setdefault(folded, folded)
    alternatives = sorted(spellings.values(), key=lambda value: (-len(value), value.casefold()))
    if not alternatives:
        return None, targets
    pattern = re.compile(r"(?<![\w'’])(" + "|".join(re.escape(value) for value in alternatives) + r")(?!\w)", re.I)
    return pattern, targets


def find_mentions(text: str, aliases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Find non-overlapping, verified aliases and retain ambiguous matches.

    Alias collisions (for example two players sharing a surname) are reported as
    ambiguous instead of silently resolving to whichever JSON key was loaded last.
    Longer names win over contained short aliases such as ``City`` in ``Manchester City``.
    """
    if not text:
        return []
    pattern, targets = _compiled_alias_matcher(_alias_signature(aliases))
    if pattern is None:
        return []

    found: list[dict[str, Any]] = []
    occupied_until = -1
    for match in pattern.finditer(text):
        if match.start() < occupied_until:
            continue
        surface = match.group(1)
        candidates = sorted(targets.get(surface.casefold(), set()))
        if not candidates:
            continue
        occupied_until = match.end()
        item: dict[str, Any] = {
            "type": candidates[0][0] if len({kind for kind, _ in candidates}) == 1 else "ambiguous",
            "surface": surface,
            "start": match.start(),
            "end": match.end(),
        }
        if len(candidates) == 1:
            item["canonical"] = candidates[0][1]
        else:
            item["candidates"] = [{"type": kind, "canonical": canonical} for kind, canonical in candidates]
        found.append(item)
    return found


def add_entity_and_coreference_fields(frame: dict[str, Any], text: str, previous_texts: list[str], aliases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mentions = find_mentions(text, aliases)
    unique_current = {
        kind: list(dict.fromkeys(item["canonical"] for item in mentions if item["type"] == kind and item.get("canonical")))
        for kind in ("player", "team")
    }
    # Resolve only when there is one unambiguous verified alias before the
    # pronoun in the current segment, or one in the nearest prior segment.
    # A name that occurs later in the same segment is not assumed to be its
    # antecedent; multiple candidates cause an explicit abstention.
    def prior_unique(kind: str) -> str | None:
        for prior in reversed(previous_texts):
            prior_entities = list(dict.fromkeys(
                item["canonical"] for item in find_mentions(prior, aliases)
                if item["type"] == kind and item.get("canonical")
            ))
            if prior_entities:
                return prior_entities[0] if len(prior_entities) == 1 else None
        return None
    pronoun_matches = list(re.finditer(r"\b(he|him|his|she|her|hers|they|them|their|theirs)\b", text, re.I))
    pronouns = [match.group(0) for match in pronoun_matches]
    links = []
    for match in pronoun_matches:
        word = match.group(0).casefold()
        kind = "team" if word in {"they", "them", "their", "theirs"} else "player"
        before = list(dict.fromkeys(
            item["canonical"] for item in mentions
            if item["type"] == kind and item.get("canonical") and item["end"] <= match.start()
        ))
        if len(before) == 1:
            antecedent = before[0]
        elif before:
            antecedent = None
        elif unique_current[kind]:
            # There is a current-segment name but it appears after this pronoun;
            # avoid guessing whether it is a later mention or antecedent.
            antecedent = None
        else:
            antecedent = prior_unique(kind)
        links.append({"text": match.group(0), "start": match.start(), "end": match.end(),
                      "entity_type": kind, "antecedent_candidate": antecedent,
                      "status": "candidate" if antecedent else "unresolved"})
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
    player_candidates = list(dict.fromkeys(item["canonical"] for item in mentions if item["type"] == "player" and item.get("canonical")))
    team_candidates = list(dict.fromkeys(item["canonical"] for item in mentions if item["type"] == "team" and item.get("canonical")))
    # A name in an event sentence is only a participant candidate: it may be
    # the shooter, goalkeeper, or recipient. Keep the semantic agent unset
    # until syntax or a human annotation supports that role.
    actor_candidate = player_candidates[0] if len(player_candidates) == 1 else None
    frame.update({
        "team_mentions": team_candidates,
        "player_mentions": player_candidates,
        "entity_mentions": mentions,
        "coreference": {
            "pronouns": pronouns,
            "links": links,
            "method": "nearest unique verified player/team alias in current or preceding transcript segment; otherwise abstain",
            "scope_seconds": 30,
            "confidence": "heuristic candidate; manually verify",
        },
        "event_frame": {"action_type": action, "actor": None, "actor_candidate": actor_candidate, "actor_candidates": player_candidates, "team_candidates": team_candidates, "patient_or_target": None, "outcome_cues": outcomes, "method": "project-written event-frame patterns; participant names are candidates, not assigned semantic roles; not trained SRL"},
    })
    return frame
