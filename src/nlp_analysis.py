"""Project-owned normalization and cue analysis for football commentary.

The analysis is deliberately transparent and rule-based. It does not claim
trained named-entity recognition, coreference, or semantic-role labeling.
"""
from __future__ import annotations

import re
from typing import Any

EVENT_TERMS: dict[str, tuple[str, ...]] = {
    "goal": (
        "goal", "scores", "score", "scored", "scoring", "netted", "finds the net",
        "back of the net", "equalizer", "equaliser", "levels the score",
    ),
    "penalty_or_var": ("penalty", "spot kick", "var check", "var review", "penalty decision"),
    "red_card": ("red card", "sent off", "shown red", "dismissed"),
    "yellow_card": ("yellow card", "booked", "cautioned"),
    "chance_or_shot": (
        "chance", "shot", "shoot", "shoots", "header", "save", "saved", "misses",
        "over the bar", "off the post", "hits the post", "hits the bar", "blocked",
    ),
    "corner_or_free_kick": ("corner", "free kick", "set piece"),
}

NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|without|did\s+not|didn't|does\s+not|doesn't|"
    r"do\s+not|don't|is\s+not|isn't|was\s+not|wasn't|has\s+not|hasn't|"
    r"have\s+not|haven't|had\s+not|hadn't|cannot|can't|could\s+not|couldn't|"
    r"would\s+not|wouldn't|will\s+not|won't|should\s+not|shouldn't|"
    r"failed\s+to|unable\s+to)\b",
    re.I,
)
HYPOTHETICAL_RE = re.compile(
    r"\b(?:if|could|might|may|would|should|possibly|perhaps|potentially|"
    r"expected\s+to|expect\s+to|looking\s+to|trying\s+to)\b",
    re.I,
)
RETROSPECTIVE_RE = re.compile(
    r"\b(?:last\s+(?:match|game|week|season|round|day|year)|previous\s+(?:match|game|season)|"
    r"earlier\s+this\s+season|in\s+(?:his|her|their)\s+career|years?\s+ago|"
    r"career\s+total|this\s+season)\b",
    re.I,
)
LIVE_RE = re.compile(
    r"\b(?:now|right\s+now|he's\s+through|she's\s+through|they're\s+through|"
    r"cross(?:es|ed|ing)?|shoots?|scores?|scored|saved|save|header|cleared|blocked|"
    r"hits?\s+the\s+(?:post|bar)|goal\s+for|goal\s+by|finds?\s+the\s+net)\b",
    re.I,
)
EXCITEMENT_RE = re.compile(
    r"!|\b(?:what\s+a|brilliant|stunning|incredible|dramatic|crucial|wonderful|"
    r"sensational|unbelievable)\b",
    re.I,
)

_SENTENCE_RE = re.compile(
    r"(?<=[.!?;])\s+|\s*,\s*(?=(?:but|however|instead|yet|now|rather|and\s+then)\b)|"
    r"\s+\b(?:but|however)\b\s+|\s*[\n]+\s*",
    re.I,
)
_EVENT_RESULT_NEGATION_RE = re.compile(
    r"\b(?:ruled\s+out|disallowed|not\s+given|chalked\s+off|called\s+back|"
    r"offside|foul\s+given|goal\s+doesn't\s+stand|goal\s+will\s+not\s+stand)\b",
    re.I,
)
_CURRENT_OVERRIDE_RE = re.compile(r"\b(?:now|just|this\s+time|right\s+now|here\s+he\s+is)\b", re.I)


def normalize_commentary(text: str) -> str:
    """Collapse whitespace only; preserve the original words and punctuation."""
    return re.sub(r"\s+", " ", text or "").strip()


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)


def _nearby_negation(clause: str, start: int, end: int) -> bool:
    before = clause[max(0, start - 55):start]
    after = clause[end:min(len(clause), end + 45)]
    # "not only" is emphatic, not a negated event.
    before = re.sub(r"\bnot\s+only\b", "", before, flags=re.I)
    near_before = re.search(
        r"\b(?:no|not|never|neither|without|did\s+not|didn't|does\s+not|doesn't|"
        r"do\s+not|don't|is\s+not|isn't|was\s+not|wasn't|has\s+not|hasn't|"
        r"have\s+not|haven't|had\s+not|hadn't|cannot|can't|could\s+not|couldn't|"
        r"would\s+not|wouldn't|will\s+not|won't|should\s+not|shouldn't|"
        r"failed\s+to|unable\s+to)\b(?:[\s\w'-]+){0,4}$",
        before,
        re.I,
    )
    return bool(near_before or _EVENT_RESULT_NEGATION_RE.search(after))


def _nearby_hypothetical(clause: str, start: int) -> bool:
    before = clause[max(0, start - 55):start]
    # Reject negative contractions as hypothetical modals ("couldn't believe he scored").
    before = re.sub(r"\b(?:couldn't|wouldn't|shouldn't|can't|won't)\b", "", before, flags=re.I)
    return bool(re.search(
        r"\b(?:if|could|might|may|would|should|possibly|perhaps|potentially|"
        r"expected\s+to|expect\s+to|looking\s+to|trying\s+to)\b(?:[\s\w'-]+){0,4}$",
        before,
        re.I,
    ))


def _has_negation(text: str) -> bool:
    text = re.sub(r"\bnot\s+only\b", "", text, flags=re.I)
    return bool(NEGATION_RE.search(text))


def _classify_event_mentions(text: str, terms: tuple[str, ...]) -> tuple[list[str], str]:
    evidence: list[str] = []
    statuses: list[str] = []
    for clause in _SENTENCE_RE.split(text):
        clause_hits: list[tuple[int, int, str]] = []
        for term in terms:
            for match in _term_pattern(term).finditer(clause):
                evidence.append(term)
                clause_hits.append((match.start(), match.end(), term))
        if not clause_hits:
            continue
        clause_statuses = []
        for start, end, _ in clause_hits:
            if _nearby_negation(clause, start, end):
                clause_statuses.append("negated")
            elif _nearby_hypothetical(clause, start):
                clause_statuses.append("hypothetical")
            elif RETROSPECTIVE_RE.search(clause) and not _CURRENT_OVERRIDE_RE.search(clause):
                clause_statuses.append("retrospective")
            else:
                clause_statuses.append("asserted_or_live")
        # A clearly asserted event in a clause outweighs a separate negated or
        # hypothetical mention in that same transcript segment.
        if "asserted_or_live" in clause_statuses:
            statuses.append("asserted_or_live")
        else:
            statuses.extend(clause_statuses)
    if not evidence:
        return [], "no_event_phrase"
    if "asserted_or_live" in statuses:
        status = "asserted_or_live"
    elif statuses and all(value == "negated" for value in statuses):
        status = "negated"
    elif statuses and all(value == "retrospective" for value in statuses):
        status = "retrospective"
    elif statuses and all(value == "hypothetical" for value in statuses):
        status = "hypothetical"
    else:
        status = "mixed_or_uncertain"
    return list(dict.fromkeys(evidence)), status


def analyze_event(text: str, event: str = "routine") -> dict[str, Any]:
    """Return event evidence and discourse cues for one commentary span.

    ``event_context`` describes the lexical mention, not verified match ground
    truth. ``asserted_or_live`` means it is not locally negated, hypothetical,
    or clearly retrospective; it does not guarantee that a highlight occurred.
    """
    text = normalize_commentary(text)
    terms = EVENT_TERMS.get(event, ())
    evidence, event_context = _classify_event_mentions(text, terms) if terms else ([], "no_event_phrase")
    return {
        "event_type": event,
        "event_evidence": evidence,
        "event_context": event_context,
        "has_negation": _has_negation(text),
        "has_hypothetical_language": bool(HYPOTHETICAL_RE.search(text)),
        "has_retrospective_language": bool(RETROSPECTIVE_RE.search(text)),
        "has_live_action_language": bool(LIVE_RE.search(text)),
        "has_excitement_cue": bool(EXCITEMENT_RE.search(text)),
        # No identities are assigned without a verified roster/alias file.
        "player_mentions": [],
        "team_mentions": [],
    }
