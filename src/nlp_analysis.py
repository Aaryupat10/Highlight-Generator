"""Lightweight, project-owned NLP analysis for football commentary.

These are transparent lexical features and event frames, not pretrained NER,
coreference, or semantic-role models. They are designed to be reviewable and
to avoid inventing player identities from noisy ASR/translation text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

EVENT_TERMS: dict[str, tuple[str, ...]] = {
    "goal": ("goal", "scores", "scored", "finds the net", "back of the net", "equalizer", "equaliser"),
    "penalty_or_var": ("penalty", "spot kick", "var check", "var review"),
    "red_card": ("red card", "sent off", "shown red"),
    "yellow_card": ("yellow card", "booked", "cautioned"),
    "chance_or_shot": ("chance", "shot", "shoot", "header", "save", "saved", "over the bar", "off the post"),
    "corner_or_free_kick": ("corner", "free kick"),
}

NEGATION_RE = re.compile(r"\b(no|not|never|without|didn't|doesn't|can't|cannot|couldn't|won't|failed to|unable to)\b", re.I)
HYPOTHETICAL_RE = re.compile(r"\b(if|could|might|may|would|should|would have|could have|chance to|looking to)\b", re.I)
RETROSPECTIVE_RE = re.compile(r"\b(last (?:match|game|week|season|round|day)|previous|earlier this season|in his career|this season)\b", re.I)
LIVE_RE = re.compile(r"\b(now|here|there|he's through|cross(?:es|ed|ing)?|shoots?|scores?|scored|saved|save|header|cleared|blocked|hits? the (?:post|bar)|goal)\b", re.I)
EXCITEMENT_RE = re.compile(r"!|\b(what a|brilliant|stunning|incredible|incredible save|dramatic|crucial|wonderful)\b", re.I)


def normalize_commentary(text: str) -> str:
    """Normalize Unicode and whitespace while preserving case and punctuation."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def excitement_features(text: str) -> dict[str, float | int]:
    """Measure visible punctuation and capitalization without changing text."""
    letters = [char for char in text if char.isalpha()]
    caps_words = re.findall(r"\b[A-Z][A-Z0-9]+\b", text)
    return {
        "n_exclaim": text.count("!"),
        "caps_ratio": round(sum(char.isupper() for char in letters) / len(letters), 4) if letters else 0.0,
        "n_caps_words": len(caps_words),
    }


def analyze_event(text: str, event: str = "routine") -> dict[str, Any]:
    """Return transparent event evidence and discourse cues for one transcript span."""
    text = normalize_commentary(text)
    low = text.casefold()
    terms = EVENT_TERMS.get(event, ())
    matched = [term for term in terms if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", low)]
    excitement = excitement_features(text)
    lexical_emphasis = bool(re.search(r"\b(what a|brilliant|stunning|incredible|dramatic|crucial|wonderful)\b", low))
    punctuation_strength = min(1.0, excitement["n_exclaim"] / 3.0)
    capitalization_strength = min(1.0, excitement["caps_ratio"] / 0.75)
    excitement_score = max(punctuation_strength * 0.75, capitalization_strength * 0.5, 0.6 if lexical_emphasis else 0.0)
    return {
        "event_type": event,
        "event_evidence": matched,
        "has_negation": bool(NEGATION_RE.search(text)),
        "has_hypothetical_language": bool(HYPOTHETICAL_RE.search(text)),
        "has_retrospective_language": bool(RETROSPECTIVE_RE.search(text)),
        "has_live_action_language": bool(LIVE_RE.search(text)),
        "has_excitement_cue": bool(EXCITEMENT_RE.search(text)),
        **excitement,
        "excitement_score_heuristic": round(excitement_score, 3),
        # Entity roles are deliberately left null: ASR text and translated
        # commentary alone are not reliable enough to assign a player or team.
        "player_mentions": [],
        "team_mentions": [],
    }
