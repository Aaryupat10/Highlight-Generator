#!/usr/bin/env python3
"""Timestamped football commentary extraction and highlight ranking MVP."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nlp_analysis import analyze_event, excitement_features, normalize_commentary
from entity_resolution import add_entity_and_coreference_fields, load_entity_aliases


@dataclass
class Segment:
    start: float
    end: float
    text: str
    text_english: str = ""
    score: float = 0.0
    event: str = "routine"
    reasons: list[str] | None = None
    rule_score: float = 0.0
    model_score: float | None = None
    event_score: float | None = None


EVENT_PATTERNS: list[tuple[str, float, tuple[str, ...]]] = [
    # Prefer live scoring language over generic mentions (for example, season statistics).
    ("goal", 1.0, (r"\bgoal for\b", r"\bgoal by\b", r"\bgoal from\b", r"\bwhat a goal\b", r"\bit'?s a goal\b", r"\bthat'?s a goal\b", r"\bscores!", r"\bscored!", r"\bhas scored\b", r"\bfinds the net\b", r"\bin the back of the net\b", r"\b(?:he|she|they)\s+(?:will|would|is going to) score\b", r"\b(?:he|she|they)['’]ll score\b", r"\b(?:equalizer|equaliser|equalization|equalisation)\b", r"\blevels? the score\b", r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) goal of (?:the|this) season\b", r"\bmarcó\b", r"\banota\b", r"\bse adelanta\b", r"\bempata\b")),
    ("penalty_or_var", 0.88, (r"\bpenalty (?:awarded|given|saved|scored|converted|decision|kick)\b", r"\bvar check\b", r"\bvar review\b", r"\bspot[- ]kick\b", r"\bpenal(?:ti)? (?:señalado|pitado|convertido|parado)\b", r"\bdesde los once metros\b")),
    ("red_card", 0.86, (r"\bsent off\b", r"\bred card\b", r"\bshown red\b", r"\btarjeta roja\b", r"\bexpulsad[oa]\b", r"\bexpulsión\b")),
    ("chance_or_shot", 0.68, (r"\bchance\b", r"\bshoots?\b", r"\bshot\b", r"\bheader\b", r"\bsaved\b", r"\bsave\b", r"\bover the bar\b", r"\boff the post\b", r"\bocasión\b", r"\bdisparo\b", r"\bremate\b", r"\btiro\b", r"\bchut\w*\b", r"\bcabece\w*\b", r"\bparad[ao]\b", r"\bal larguero\b", r"\bal poste\b")),
    ("yellow_card", 0.58, (r"\byellow card\b", r"\bbooked\b", r"\bcautioned\b", r"\btarjeta amarilla\b", r"\bamonestad[oa]\b")),
    ("corner_or_free_kick", 0.42, (r"\bcorner\b", r"\bfree[- ]kick\b", r"\bcórner\b", r"\bcorner\b", r"\bfalta\b", r"\blibre directo\b")),
]
EXCITEMENT = (r"!{1,}", r"\bwhat a save\b", r"\bstunning\b", r"\bbrilliant\b", r"\bdramatic\b", r"\bcrucial\b", r"\bnearly\b", r"\bqué parad[ao]\b", r"\bimpresionante\b", r"\bincreíble\b", r"\bclarísima\b")


def run_ffmpeg(video: Path, wav: Path, start_seconds: float = 0.0, duration_seconds: float | None = None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Install it first (for example: brew install ffmpeg).")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds > 0:
        cmd += ["-ss", str(start_seconds)]
    cmd += ["-i", str(video)]
    if duration_seconds is not None:
        cmd += ["-t", str(duration_seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Could not extract audio from {video}:\n{proc.stderr.strip()}")
    if not wav.exists() or wav.stat().st_size < 1000:
        raise RuntimeError("The extracted audio is empty or unusually small. Verify that this video contains decodable audio.")


def transcribe(wav: Path, model_size: str, language: str | None, task: str = "transcribe") -> tuple[list[Segment], str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Missing faster-whisper. Install project requirements before running the pipeline.") from exc
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav), language=language, task=task, vad_filter=True, beam_size=3)
    result = [Segment(float(s.start), float(s.end), normalize_commentary(s.text)) for s in segments if (s.text or "").strip()]
    return result, (info.language or language or "")


# Whisper language codes mapped to NLLB-200 source language codes.
NLLB_LANGUAGES = {
    "af": "afr_Latn", "ar": "arb_Arab", "az": "aze_Latn", "be": "bel_Cyrl",
    "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gl": "glg_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl",
    "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl",
    "mr": "mar_Deva", "ms": "zsm_Latn", "mt": "mlt_Latn", "my": "mya_Mymr",
    "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl",
    "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
    "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn", "vi": "vie_Latn", "yi": "ydd_Hebr", "zh": "zho_Hans",
}


def translate_segments(segments: list[Segment], detected_language: str, source_lang_override: str = "auto") -> bool:
    """Translate transcript text to English in small batches; timestamps remain attached."""
    if detected_language == "en":
        for segment in segments:
            segment.text_english = segment.text
        return True
    source_lang = source_lang_override if source_lang_override != "auto" else NLLB_LANGUAGES.get(detected_language)
    if not source_lang:
        print(f"No NLLB language mapping for Whisper code '{detected_language}'; keeping original transcript and using available rules.")
        return False
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        print(f"Translation packages are missing ({exc}); keeping original transcript and using available rules.")
        return False

    model_id = "facebook/nllb-200-distilled-600M"
    print(f"Translating transcript to English from detected language '{detected_language}' ({source_lang})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=source_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.eval()
        english_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        batch_size = 8
        for offset in range(0, len(segments), batch_size):
            batch = segments[offset:offset + batch_size]
            encoded = tokenizer([s.text for s in batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.inference_mode():
                generated = model.generate(**encoded, forced_bos_token_id=english_id, max_length=160)
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for segment, translated in zip(batch, translations):
                segment.text_english = translated
            if offset and offset % 160 == 0:
                print(f"Translated {min(offset + batch_size, len(segments))}/{len(segments)} transcript segments...")
        del model
        return True
    except Exception as exc:
        print(f"Translation failed ({exc}); continuing with the original transcript and available rules.")
        return False


def score_segment(text: str, local_context: str = "") -> tuple[float, str, list[str]]:
    low = text.casefold()
    nearby = f"{low} {local_context.casefold()}"
    matches: list[tuple[str, float, str]] = []
    for event, score, patterns in EVENT_PATTERNS:
        if any(re.search(p, low) for p in patterns):
            matches.append((event, score, event.replace("_", " ")))
    event, base, reason = max(matches, key=lambda x: x[1]) if matches else ("routine", 0.0, "")
    # Suppress common non-event mentions that keyword matching otherwise mistakes
    # for live chances, cards, or set pieces.
    if event == "chance_or_shot" and re.search(
        r"\bno (?:real |clear |great |big )?chances?\b|\bnot the result of chance\b|\bfruit of (?:the )?chance\b|\bchance to (?:play|sign|join)\b|\b(?:never|not|without) (?:had |having )?(?:any |great |big )?chances?\b",
        low,
    ):
        return 0.0, "routine", []
    if event == "chance_or_shot" and re.search(
        r"\b(?:didn't|did not|doesn't|does not|can't|cannot|can not|couldn't|could not|won't|will not|shouldn't|should not)\s+(?:(?:get|take|find|manage|have|make)\s+)?(?:a\s+)?(?:shot|chance|shoot)\b|\b(?:failed|unable|impossible) to (?:shoot|get a shot)\b|\bno (?:real |clear )?(?:shot|chance)\b",
        low,
    ):
        return 0.0, "routine", []
    if event == "chance_or_shot" and re.search(
        r"\b(?:can|could|may|might|will|should) (?:look for|search for|try|attempt|take|have) (?:a )?(?:shot|chance)\b|\bcan dare (?:with|to) (?:the )?shot\b",
        low,
    ):
        return 0.0, "routine", []
    if event == "chance_or_shot" and re.search(r"\b\d+(?:st|nd|rd|th)\s+shots?\b", low):
        return 0.0, "routine", []
    if event == "goal" and re.search(r"\b\d+(?:st|nd|rd|th) goal\b.*\bthis season\b", low):
        return 0.0, "routine", []
    if event == "goal" and re.search(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) goal of (?:the|this) season\b", low):
        live_goal_context = r"\b(?:he|she|they)['’]ll score\b|\b(?:he|she|they)\s+(?:will|would|is going to) score\b|\bgoal from\b|\bgoal by\b|\b(?:equalizer|equaliser|equalization|equalisation)\b|\b(?:[a-z][a-z&'-]*\s+){0,3}to lead\b|\blevels? the score\b|\b(?:scores?|scored)\s+\d"
        if not re.search(live_goal_context, nearby):
            return 0.0, "routine", []
    if event == "yellow_card" and re.search(r"\b(signing|transfer|fichajes?|transfer window)\b", low):
        return 0.0, "routine", []
    if event == "corner_or_free_kick" and re.search(r"\b(?:like|as) a corner\b", low):
        return 0.0, "routine", []
    text_cues = excitement_features(text)
    caps_emphasis = text_cues["caps_ratio"] >= 0.55 and text_cues["n_caps_words"] >= 2
    excitement = any(re.search(p, low) for p in EXCITEMENT) or caps_emphasis
    score = min(1.0, base + (0.12 if excitement and base else 0.0))
    emphasis_reason = "commentary capitalization emphasis" if caps_emphasis else "commentary emphasis"
    reasons = ([reason] if reason else []) + ([emphasis_reason] if excitement else [])
    return score, event, reasons


def rank_segments(segments: list[Segment], min_score: float = 0.35, merge_gap: float = 12.0, classifier: Any = None, sequence_predictions: dict[str, Any] | None = None, entity_aliases: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    selected: list[Segment] = []
    for i, seg in enumerate(segments):
        scoring_text = seg.text_english or seg.text
        rule_score, rule_event, rule_reasons = score_segment(scoring_text, classifier_context(segments, i, radius_seconds=6.0))
        context = classifier_context(segments, i)
        seg.event_score = classifier.event_probability(context) if classifier is not None and hasattr(classifier, "event_probability") else None
        seg.model_score = classifier.probability(context) if classifier is not None else None
        if sequence_predictions is not None:
            if rule_event == "routine":
                seg.score, seg.event, seg.reasons = 0.0, "routine", []
                seg.rule_score = 0.0
            else:
                if "highlight" in sequence_predictions:
                    model_score = float(sequence_predictions["highlight"][i])
                    seg.event_score = float(sequence_predictions["eventness"][i])
                    seg.score = round(0.25 * rule_score + 0.25 * seg.event_score + 0.50 * model_score, 6)
                else:
                    model_score = float(sequence_predictions["eventness"][i])
                    seg.score = round(0.45 * rule_score + 0.55 * model_score, 6)
                seg.event = rule_event
                sequence_name = "project-trained two-tier hierarchical model" if "highlight" in sequence_predictions else "project-trained match-half sequence model"
                seg.reasons = ["football event phrase", sequence_name] + rule_reasons
                seg.rule_score, seg.model_score = rule_score, model_score
        elif classifier is not None:
            # Event phrases propose candidate moments; the trained model
            # adjusts their ranking using surrounding commentary. This avoids
            # near-0.5 model priors labeling long routine passages as highlights.
            if rule_event == "routine":
                seg.score, seg.event, seg.reasons = 0.0, "routine", []
                seg.rule_score = 0.0
            else:
                model_score = float(seg.model_score or 0.0)
                if seg.event_score is not None:
                    # Human-highlight and weak event-relatedness are different
                    # trained targets; retain each score and expose the blend.
                    seg.score = round(0.25 * rule_score + 0.25 * seg.event_score + 0.50 * model_score, 6)
                else:
                    seg.score = round(0.45 * rule_score + 0.55 * model_score, 6)
                seg.event = rule_event
                seg.reasons = ["football event phrase", "project-trained commentary model"] + rule_reasons
                seg.rule_score, seg.model_score = rule_score, model_score
        else:
            seg.score, seg.event, seg.reasons = rule_score, rule_event, rule_reasons
            seg.rule_score, seg.model_score = rule_score, None
        # A past-match recap is not an event happening at this timestamp.
        if seg.event == "goal" and re.search(r"\b(last match|last day|last round|previous match|last jornada|última jornada|último partido)\b", scoring_text, re.I):
            seg.score, seg.event, seg.reasons = 0.0, "routine", []
        if seg.score >= min_score:
            selected.append(seg)
    selected.sort(key=lambda x: x.start)
    groups: list[list[Segment]] = []
    for seg in selected:
        if groups and seg.start - groups[-1][-1].end <= merge_gap:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    results = []
    for group in groups:
        peak = max(group, key=lambda x: x.score)
        frame = analyze_event(peak.text_english or peak.text, peak.event)
        peak_text = " ".join(s.text_english or s.text for s in group)
        previous_texts = [(s.text_english or s.text) for s in segments if s.start < peak.start and s.end >= peak.start - 30.0]
        frame = add_entity_and_coreference_fields(frame, peak_text, previous_texts, entity_aliases or {"team": {}, "player": {}})
        frame["event_evidence"] = sorted({term for s in group for term in analyze_event(s.text_english or s.text, s.event)["event_evidence"]})
        frame["peak_timestamp"] = timestamp(peak.start)
        frame["peak_seconds"] = round(peak.start, 2)
        frame["candidate_segment_count"] = len(group)
        frame["temporal_density"] = round(sum(s.score for s in group) / max(1, group[-1].end - group[0].start), 4)
        buildup = [s for s in segments if group[0].start - 30.0 <= s.start < group[0].start]
        frame["buildup_window_seconds"] = 30.0
        frame["buildup_segment_count"] = len(buildup)
        frame["buildup_mean_rank_score"] = round(sum(s.score for s in buildup) / len(buildup), 4) if buildup else 0.0
        frame["peak_position_in_candidate_seconds"] = round(peak.start - group[0].start, 2)
        frame["event_relatedness_score"] = round(peak.event_score, 3) if peak.event_score is not None else None
        has_supervised_head = (classifier is not None and getattr(classifier, "model_type", "") == "pipeline_multitask_tfidf_mlp_v1") or (sequence_predictions is not None and "highlight" in sequence_predictions)
        frame["supervised_highlight_score"] = round(peak.model_score, 3) if has_supervised_head and peak.model_score is not None else None
        frame["ranker"] = "phrase_rules_plus_project_trained_model" if (classifier is not None or sequence_predictions is not None) else "phrase_rules"
        peak_index = next(i for i, segment in enumerate(segments) if segment is peak)
        if sequence_predictions is not None:
            family_prediction = sequence_predictions["event_types"][peak_index]
            attention_row = sequence_predictions["attention"][peak_index]
            top_context = sorted(enumerate(attention_row), key=lambda item: item[1], reverse=True)
            frame["temporal_attention_context"] = [{"timestamp": timestamp(segments[j].start), "weight": round(float(weight), 4)} for j, weight in top_context[:3] if j != peak_index]
            frame["attention_buildup_mass"] = round(sum(float(weight) for j, weight in enumerate(attention_row) if group[0].start - 30.0 <= segments[j].start < peak.start), 4)
            frame["predicted_event_family"] = family_prediction
            rule_family = {
                "goal": "goal",
                "chance_or_shot": "shot_or_chance",
                "yellow_card": "card",
                "red_card": "card",
            }.get(peak.event)
            # The model has no penalty/set-piece family, so report agreement
            # only when both labels belong to the same defined taxonomy.
            frame["event_family_agrees_with_rule"] = (
                family_prediction.get("label") == rule_family
                if isinstance(family_prediction, dict) and rule_family
                else None
            )
        elif classifier is not None and hasattr(classifier, "predict_event_type"):
            family_prediction = classifier.predict_event_type(classifier_context(segments, peak_index))
            frame["predicted_event_family"] = family_prediction
            frame["temporal_attention_context"] = []
        else:
            frame["predicted_event_family"] = None
            frame["temporal_attention_context"] = []
            frame["attention_buildup_mass"] = None
            frame["event_family_agrees_with_rule"] = None
        if sequence_predictions is None and classifier is not None and hasattr(classifier, "explain"):
            frame["model_evidence"] = classifier.explain(classifier_context(segments, peak_index), top_n=6)
        else:
            frame["model_evidence"] = []
        results.append({
            "start_seconds": round(max(0.0, group[0].start), 2),
            "end_seconds": round(group[-1].end, 2),
            "start_timestamp": timestamp(group[0].start),
            "end_timestamp": timestamp(group[-1].end),
            "event": peak.event,
            "score": round(peak.score, 3),
            "rule_score": round(peak.rule_score, 3),
            "model_score": round(peak.model_score, 3) if peak.model_score is not None else None,
            "supervised_highlight_score": round(peak.model_score, 3) if has_supervised_head and peak.model_score is not None else None,
            "event_score": round(peak.event_score, 3) if peak.event_score is not None else None,
            "n_exclaim": frame["n_exclaim"],
            "caps_ratio": frame["caps_ratio"],
            "n_caps_words": frame["n_caps_words"],
            "excitement_score_heuristic": frame["excitement_score_heuristic"],
            "event_frame": frame,
            "commentary": " ".join(s.text for s in group),
            "commentary_english": " ".join(s.text_english for s in group) if all(s.text_english for s in group) else "",
            "reasons": sorted({r for s in group for r in (s.reasons or [])}),
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def classifier_context(segments: list[Segment], index: int, radius_seconds: float = 15.0) -> str:
    """Build a 30-second text window like those used in manual annotation."""
    center = (segments[index].start + segments[index].end) / 2
    lower, upper = center - radius_seconds, center + radius_seconds
    left = index
    while left > 0 and segments[left - 1].end >= lower:
        left -= 1
    right = index + 1
    while right < len(segments) and segments[right].start <= upper:
        right += 1
    return " ".join((s.text_english or s.text) for s in segments[left:right])


def write_outputs(out_dir: Path, segments: list[Segment], highlights: list[dict[str, Any]], video: Path, model: str, classifier: Any = None, task: str = "transcribe", sequence_predictions: dict[str, Any] | None = None, sequence_model: Any = None, entity_aliases: dict[str, dict[str, Any]] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "transcript.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_seconds", "end_seconds", "start_timestamp", "end_timestamp", "text", "text_english", "event", "n_exclaim", "caps_ratio", "n_caps_words", "rule_score", "event_relatedness_score", "supervised_highlight_score", "model_score", "score", "nlp_cues_json"])
        writer.writeheader()
        for i, s in enumerate(segments):
            analysis_text = s.text_english or s.text
            frame = analyze_event(analysis_text, s.event)
            previous_texts = []
            prior_index = i - 1
            while prior_index >= 0 and segments[prior_index].end >= s.start - 30.0:
                prior = segments[prior_index]
                previous_texts.append(prior.text_english or prior.text)
                prior_index -= 1
            previous_texts.reverse()
            frame = add_entity_and_coreference_fields(
                frame, analysis_text, previous_texts,
                entity_aliases or {"team": {}, "player": {}, "ambiguous": {}},
            )
            excitement = excitement_features(s.text)
            frame["source_text_cues"] = excitement
            frame["ranking_text_cues"] = excitement_features(s.text_english or s.text)
            writer.writerow({
                "start_seconds": round(s.start, 2), "end_seconds": round(s.end, 2),
                "start_timestamp": timestamp(s.start), "end_timestamp": timestamp(s.end),
                "text": s.text, "text_english": s.text_english, "event": s.event,
                **excitement,
                "rule_score": round(s.rule_score, 3),
                "event_relatedness_score": round(s.event_score, 3) if s.event_score is not None else "",
                "supervised_highlight_score": round(s.model_score, 3) if ((classifier is not None and getattr(classifier, "model_type", "") == "pipeline_multitask_tfidf_mlp_v1") or (sequence_predictions is not None and "highlight" in sequence_predictions)) and s.model_score is not None else "",
                "model_score": round(s.model_score, 3) if s.model_score is not None else "",
                "score": round(s.score, 3), "nlp_cues_json": json.dumps(frame, ensure_ascii=False),
            })
    translated = any(s.text_english and s.text_english != s.text for s in segments)
    if task == "translate":
        note = "Whisper direct English translation of the detected speech. Original-language wording is not retained in this mode; timestamps are estimates."
    else:
        note = "English translations are shown with original speech below each segment. Timestamps are estimates." if translated else "Speech recognition output. Timestamps are estimates."
    transcript_lines = [f"# Full timestamped transcript: {video.name}", "", note, ""]
    transcript_lines.extend(f"[{timestamp(s.start)}–{timestamp(s.end)}] {s.text_english or s.text}" + (f"\n\nOriginal: {s.text}" if s.text_english and s.text_english != s.text else "") for s in segments)
    (out_dir / "transcript.md").write_text("\n\n".join(transcript_lines) + "\n", encoding="utf-8")
    model_supervision = (getattr(sequence_model, "supervision", None)
                         or getattr(classifier, "supervision", None)
                         or "transparent event rules only")
    pseudo_label_note = (
        " Assistant text-only pseudo-labels, if supplied for an experiment, are not human ground truth."
        if "assistant text-only" in model_supervision.casefold()
        else ""
    )
    if sequence_predictions is not None and "highlight" in sequence_predictions:
        score_note = f"Hybrid ranking combines event rules with {model_supervision}. Scores are exploratory and are not calibrated probabilities.{pseudo_label_note}"
        deep_learning_stage = "project-trained two-tier hierarchical BiGRU with local window attention, match-half attention, and separate eventness/family/highlight heads"
    elif sequence_predictions is not None:
        score_note = f"Hybrid ranking: event phrase evidence and a project-trained match-half sequence score ({getattr(classifier, 'supervision', 'weak labels')}); scores are not calibrated probabilities or guaranteed highlight judgments."
        deep_learning_stage = "project-trained BiGRU with temporal self-attention and eventness/event-family heads; exploratory weak supervision"
    elif classifier is not None and getattr(classifier, "model_type", "") == "pipeline_multitask_tfidf_mlp_v1":
        score_note = f"Hybrid ranking combines event phrase evidence with {model_supervision}. Components and final rank are exploratory scores, not calibrated probabilities.{pseudo_label_note}"
        deep_learning_stage = "project-trained TF-IDF/PyTorch MLP with separately supervised eventness, event-family, and human highlight-value heads"
    elif classifier is not None:
        score_note = f"Hybrid ranking: football event phrase matching plus a trained event-context model ({getattr(classifier, 'supervision', 'unknown supervision')}); score is not calibrated."
        deep_learning_stage = "project-trained PyTorch text classifier; label meaning depends on the checkpoint's training data"
    else:
        score_note = "Rule-match score, not a calibrated probability or measured confidence."
        deep_learning_stage = "not used; phrase rules only"
    pipeline_stages = {
        "speech_to_text": "pretrained Faster-Whisper; language auto-detection or user override",
        "translation": "optional pretrained NLLB translation to English; timestamps retained",
        "nlp_preprocessing": "project-written Unicode/whitespace cleanup preserving case and punctuation; exclamation/caps features plus negation/hypothetical/retrospective/live-action cues",
        "coreference": "project-written nearest-unique verified player/team alias candidate within a 30-second transcript window; ambiguous cases abstain and links require manual verification",
        "event_extraction": "project-written event rules and event-frame patterns emit action, conservative actor candidate, team candidates, and outcome cues; target remains unresolved; not trained SRL",
        "text_representation": "project-fitted TF-IDF unigram/bigram features when TF-IDF checkpoint is selected; otherwise hashed word features",
        "deep_learning": deep_learning_stage,
        "model_supervision": model_supervision,
        "temporal_selection": "score-based nearby-event grouping with peak, candidate density, and preceding 30-second buildup statistics",
        "explainability": "temporal attention/context mass for the sequence model; TF-IDF feature ablation for the MLP; neither is causal evidence",
        "limitations": ["Coreference and event-role extraction use conservative dictionary/phrase heuristics, not trained coreference or SRL models", "Only fixture team names are auto-read from SoccerNet-style paths; player names require a verified aliases JSON", "Excitement labels in the assistant review set are text-only proxies, not judgments of vocal intensity", "Assistant text-only annotations are pseudo-labels and are not human ground truth", "Event-family head is weak-label trained and remains experimental", "No claim of universal accuracy across languages or competitions"],
    }
    (out_dir / "highlights.json").write_text(json.dumps({"source_video": video.name, "asr_model": model, "timestamp_note": "Whisper segment timestamps are estimates; verify against the video before editing.", "score_note": score_note, "pipeline_stages": pipeline_stages, "highlights": highlights}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "highlights.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["start_timestamp", "end_timestamp", "event", "score", "rule_score", "event_score", "supervised_highlight_score", "n_exclaim", "caps_ratio", "n_caps_words", "excitement_score_heuristic", "model_score", "commentary", "commentary_english", "reasons"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for h in highlights:
            writer.writerow({**{k: h.get(k, "") for k in fields if k != "reasons"}, "reasons": ", ".join(h["reasons"])})
    md = ["# Ranked commentary highlights", "", f"Source: `{video.name}`", "", "Timestamps are estimated from ASR and should be checked before video editing.", score_note, ""]
    if not highlights:
        md.append("No candidate moments passed the current event threshold. Inspect `transcript.csv` and adjust the threshold if needed.")
    for i, h in enumerate(highlights, 1):
        rank_label = "Hybrid rank score" if classifier is not None or sequence_predictions is not None else "Rule score"
        md += [f"## {i}. {h['event'].replace('_', ' ').title()} — {h['start_timestamp']} to {h['end_timestamp']}", f"{rank_label}: {h['score']:.2f}", f"Peak: {h['event_frame']['peak_timestamp']} | Candidate segments: {h['event_frame']['candidate_segment_count']} | 30-second buildup segments: {h['event_frame']['buildup_segment_count']}", ""]
        role_frame = h["event_frame"].get("event_frame", {})
        md.append(f"Event frame (project-written patterns): action={role_frame.get('action_type', 'unknown')}; named participant candidate={role_frame.get('actor_candidate') or 'none'}; semantic actor={role_frame.get('actor') or 'unassigned'}; outcomes={', '.join(role_frame.get('outcome_cues', [])) or 'none detected'}.")
        coref = h["event_frame"].get("coreference", {})
        if coref.get("pronouns"):
            links = coref.get("links", [])
            rendered_links = ", ".join(
                f"{link['text']} → {link['antecedent_candidate']}" if link.get("antecedent_candidate")
                else f"{link['text']} → unresolved"
                for link in links
            )
            md.append(f"Coreference candidates: {rendered_links or ', '.join(coref['pronouns']) + ' → unresolved'} ({coref.get('method', '')}).")
        md.append("")
        if h["event_frame"].get("event_relatedness_score") is not None:
            md.append(f"Event relatedness: {h['event_frame']['event_relatedness_score']:.2f} | Supervised highlight score: {h['event_frame']['supervised_highlight_score']:.2f}")
            md.append("")
        evidence = h["event_frame"].get("event_evidence", [])
        if evidence:
            md += [f"Event text evidence: {', '.join(evidence)}", ""]
        md.append(f"Emphasis cues: {h['n_exclaim']} exclamation marks; caps ratio {h['caps_ratio']:.2f}; all-caps words {h['n_caps_words']}; heuristic excitement {h['excitement_score_heuristic']:.2f} (not a trained head).")
        md.append("")
        family = h["event_frame"].get("predicted_event_family")
        if family:
            agrees = h["event_frame"].get("event_family_agrees_with_rule")
            status = "agrees with" if agrees else "disagrees with"
            md += [f"Experimental model family: {family['label']} ({family['score']:.2f}); {status} the rule-based event label `{h['event']}`. Treat this head as experimental.", ""]
        attention_context = h["event_frame"].get("temporal_attention_context", [])
        if attention_context:
            md += ["Top temporal context attended by the sequence model: " + ", ".join(f"{item['timestamp']} ({item['weight']:.2f})" for item in attention_context), ""]
            md += [f"Attention mass on the 30-second pre-candidate buildup: {h['event_frame'].get('attention_buildup_mass', 0):.2f} (model attention is a diagnostic, not causal evidence).", ""]
        model_evidence = h["event_frame"].get("model_evidence", [])
        if model_evidence:
            md += ["Model feature evidence (removing these words lowered this candidate's score): " + ", ".join(item["feature"] for item in model_evidence), ""]
        if h.get("commentary_english") and h["commentary_english"] != h["commentary"]:
            md += [f"**English:** {h['commentary_english']}", "", f"Original: {h['commentary']}", ""]
        else:
            md += [h.get("commentary_english") or h["commentary"], ""]
    (out_dir / "highlights.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a football video and export ranked commentary highlights with timestamps.")
    parser.add_argument("video", type=Path, help="Path to a video with usable commentary audio")
    parser.add_argument("--out", type=Path, default=Path("outputs/run"), help="Output folder (default: outputs/run)")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default="base", help="Whisper model; base is a practical starting point for an 8 GB Mac")
    parser.add_argument("--language", default="auto", help="Spoken language code; default auto-detects the language")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", help="Whisper speech task; use transcribe to retain the original language")
    parser.add_argument("--translate-to-english", "--translate-highlights", dest="translate_to_english", action="store_true", help="Translate the full transcript to English before ranking; requires torch and transformers")
    parser.add_argument("--source-lang", default="auto", help="Optional NLLB language code override; normally inferred from Whisper language detection")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum rule score for candidate events")
    parser.add_argument("--classifier-model", type=Path, help="Optional trained PyTorch text model from src/train_highlight_model.py; requires English text")
    parser.add_argument("--sequence-model", type=Path, help="Optional trained match-half BiGRU/temporal-attention model from src/train_sequence_model.py; use instead of --classifier-model")
    parser.add_argument("--hierarchical-model", type=Path, help="Optional project-trained two-tier model from src/train_hierarchical_model.py; use instead of --classifier-model or --sequence-model")
    parser.add_argument("--entity-aliases", type=Path, help="Optional verified JSON roster aliases with teams and players; see data/entity_aliases.example.json")
    parser.add_argument("--merge-gap", type=float, default=12.0, help="Merge candidate commentary within this many seconds")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start at this offset in the source video; output times remain relative to the full video")
    parser.add_argument("--duration-seconds", type=float, help="Optional duration to process; useful for a short demo sample")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"Video file does not exist: {video}")
    if args.start_seconds < 0 or (args.duration_seconds is not None and args.duration_seconds <= 0):
        parser.error("Start must be non-negative and duration must be positive.")
    if sum(bool(value) for value in (args.sequence_model, args.hierarchical_model, args.classifier_model)) > 1:
        parser.error("Use only one of --classifier-model, --sequence-model, or --hierarchical-model.")
    language = None if args.language.casefold() == "auto" else args.language
    with tempfile.TemporaryDirectory(prefix="c2h-") as temp:
        wav = Path(temp) / "audio.wav"
        print("Extracting mono audio...")
        run_ffmpeg(video, wav, args.start_seconds, args.duration_seconds)
        print(f"Transcribing with Whisper {args.model} on CPU (int8)...")
        segments, detected_language = transcribe(wav, args.model, language, args.task)
    for segment in segments:
        segment.start += args.start_seconds
        segment.end += args.start_seconds
    if not segments:
        raise SystemExit("No speech was transcribed. Check audio, language, and input video.")
    print(f"Detected speech language: {detected_language or 'unknown'}")
    if args.translate_to_english:
        if args.task == "translate":
            for segment in segments:
                segment.text_english = segment.text
        else:
            translate_segments(segments, detected_language, args.source_lang)
    elif args.task == "translate":
        # Whisper's direct translate task already returns English. Mark it as
        # English for structured outputs without pretending the source text is retained.
        for segment in segments:
            segment.text_english = segment.text
    classifier = None
    sequence_model = None
    if args.classifier_model:
        try:
            from text_classifier import CommentaryClassifier
            classifier = CommentaryClassifier.load(args.classifier_model.expanduser())
        except Exception as exc:
            raise SystemExit(f"Could not load highlight classifier: {exc}") from exc
        print(f"Ranking with trained event-context model: {args.classifier_model}")
    sequence_predictions = None
    if args.sequence_model:
        try:
            from sequence_model import MatchHalfSequenceModel
            sequence_model = MatchHalfSequenceModel.load(args.sequence_model.expanduser())
        except Exception as exc:
            raise SystemExit(f"Could not load match-half sequence model: {exc}") from exc
        ordered = sorted(range(len(segments)), key=lambda i: segments[i].start)
        sequence_texts = [classifier_context(segments, i) for i in ordered]
        sequence_times = [segments[i].start for i in ordered]
        print(f"Ranking with match-half attention sequence model: {args.sequence_model}")
        predictions = sequence_model.predict(sequence_texts, sequence_times)
        # Map sorted model rows back to the original transcript segment order.
        sequence_predictions = {key: [None] * len(segments) for key in predictions}
        for key, values in predictions.items():
            for model_index, original_index in enumerate(ordered):
                sequence_predictions[key][original_index] = values[model_index]
    if args.hierarchical_model:
        try:
            from hierarchical_model import HierarchicalMatchModel
            sequence_model = HierarchicalMatchModel.load(args.hierarchical_model.expanduser())
        except Exception as exc:
            raise SystemExit(f"Could not load hierarchical model: {exc}") from exc
        ordered = sorted(range(len(segments)), key=lambda i: segments[i].start)
        sequence_texts = [classifier_context(segments, i) for i in ordered]
        sequence_times = [segments[i].start for i in ordered]
        print(f"Ranking with project-trained two-tier hierarchical model: {args.hierarchical_model}")
        predictions = sequence_model.predict(sequence_texts, sequence_times)
        sequence_predictions = {key: [None] * len(segments) for key in predictions}
        for key, values in predictions.items():
            for model_index, original_index in enumerate(ordered):
                sequence_predictions[key][original_index] = values[model_index]
    if args.entity_aliases and not args.entity_aliases.is_file():
        parser.error(f"Entity alias file does not exist: {args.entity_aliases}")
    entity_aliases = load_entity_aliases(args.entity_aliases.expanduser() if args.entity_aliases else None, video)
    highlights = rank_segments(segments, args.min_score, args.merge_gap, classifier, sequence_predictions, entity_aliases)
    write_outputs(args.out.expanduser().resolve(), segments, highlights, video, args.model, classifier, args.task, sequence_predictions, sequence_model, entity_aliases)
    print(f"Transcript segments: {len(segments)}")
    print(f"Ranked highlight candidates: {len(highlights)}")
    print(f"Results saved in: {args.out.expanduser().resolve()}")
    for h in highlights[:10]:
        preview = h.get("commentary_english") or h["commentary"]
        print(f"{h['start_timestamp']}  {h['event']:<20} {h['score']:.2f}  {preview[:140]}")


if __name__ == "__main__":
    main()
