"""
Stage 10 - Structured output

Turns peaks (Stage 9) + the per-line predictions + Stage 4's event details into records like

  [line 57, event: GOAL, player: Erling Haaland, method: header from cross, excitement: 0.97,
   buildup_length: 4 lines, narrative_context: "sustained late pressure"]

Where each field comes from (say this in the report - only some of it is learned):
  excitement         Head A score at the peak line (or the weak label when --source weak)
  event              Head B's most likely class at the peak line (Stage 4's class for --source weak)
  player             Stage 4's agent on the peak line, else the nearest line of the moment that has
                     one; falls back to Head C only when Stage 4 found nobody and Head C is not '<other>'
  method             Stage 4's method / source slots on or next to the peak line ("header from cross")
  buildup_length     lines before the peak that stay above the span floor (Stage 9)
  narrative_context  a RULE TEMPLATE over Stage 4 classes in the build-up and the match phase.
                     It is not generated text and not learned.
  time               a minute field if Stage 4's rows have one (minute/time/timestamp/clock), else
                     the line number. The Kaggle commentary has no timestamps, so expect line numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Multitask heads"))
from labels import agent_name  # noqa: E402
from peak_detection import build_span, find_peaks  # noqa: E402

TIME_KEYS = ("minute", "time", "timestamp", "clock")

NARRATIVE_DEFAULT = {
    "chance_classes": ["near_miss"],        # classes that read as "the team is creating chances"
    "heat_classes": ["foul", "card"],       # classes that read as "a scrappy / heated spell"
    "early_below": 0.15, "late_above": 0.85,
    "sudden_max_buildup": 1,                # <= this many build-up lines -> "sudden moment"
}


def slot_text(v, keys=("entity", "label", "text", "name")) -> str | None:
    """Stage 4 slots are strings, dicts like {'label': 'header', 'text': 'nods', ...}, or lists of dicts
    (an empty list means 'nothing found')."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            t = slot_text(x, keys)
            if t and t not in parts:
                parts.append(t)
        return ", ".join(parts) or None
    if isinstance(v, dict):
        for k in keys:
            if v.get(k):
                return str(v[k]).strip() or None
        return None
    s = str(v).strip()
    return s if s and s.lower() not in {"none", "null", "nan"} else None


def load_events(events_dir: str | Path | None, match_id: str) -> dict[int, dict]:
    if not events_dir:
        return {}
    p = Path(events_dir) / f"events_{match_id}.jsonl"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["line_no"]: r for r in map(json.loads, filter(str.strip, f))}


def _by_distance(peak: int, start: int, end: int):
    return sorted(range(start, end + 1), key=lambda i: (abs(i - peak), i))


def describe_method(rows: dict[int, dict], peak: int, start: int, end: int) -> str | None:
    """'header from cross' from the closest primary event in the moment that has a method/source."""
    for i in _by_distance(peak, max(start, peak - 2), min(end, peak + 2)):
        for e in (rows.get(i) or {}).get("events") or []:
            m, s = slot_text(e.get("method"), ("text", "label")), slot_text(e.get("source"))
            if m or s:
                return f"{m} from {s}" if (m and s) else (m or f"from {s}")
    return None


def pick_player(meta: list[dict], event: str, peak: int, start: int, end: int,
                model_player: str | None, max_dist: int = 2) -> tuple[str | None, str]:
    """The peak line's own Stage 4 agent, else the agent of a line of the SAME class at most max_dist
    lines away (e.g. 'Haaland scores' right after 'GOAL!!!'). A player from a different kind of line is
    never borrowed: better no name than the wrong one."""
    order = [i for i in _by_distance(peak, start, end) if abs(i - peak) <= max_dist]
    for i in order:
        a = agent_name(meta[i].get("primary_agent"))
        if a and (i == peak or meta[i].get("primary_class") == event):
            return a, "stage4"
    if model_player and model_player != "<other>":
        return model_player, "head_c"
    return None, "none"


def narrative_context(meta: list[dict], valid: np.ndarray, peak: int, start: int,
                      cfg: dict | None = None) -> str:
    n = {**NARRATIVE_DEFAULT, **(cfg or {})}
    classes = [m.get("primary_class") for m in meta[start:peak]]
    chances = sum(c in n["chance_classes"] for c in classes)
    heat = sum(c in n["heat_classes"] for c in classes)
    length = peak - start
    live_before = int(valid[:peak + 1].sum())
    frac = live_before / max(int(valid.sum()), 1)
    pw = "early " if frac < n["early_below"] else ("late " if frac > n["late_above"] else "")
    if chances >= 2 or (chances >= 1 and length >= 3):
        return f"sustained {pw}pressure"
    if chances == 1:
        return f"{pw}moment after a near miss".strip()
    if heat >= 2:
        return f"{pw}moment after a scrappy spell".strip()
    if length <= n["sudden_max_buildup"]:
        return f"sudden {pw}moment with little build-up".replace("  ", " ")
    return f"gradual {pw}build-up".replace("  ", " ")


def build_highlights(match_id: str, scores: np.ndarray, valid: np.ndarray, meta: list[dict],
                     rows: dict[int, dict], classes: list[str],
                     event_prob: np.ndarray | None = None, agent_prob: np.ndarray | None = None,
                     agent_vocab: list[str] | None = None, narrative_cfg: dict | None = None,
                     threshold=0.5, min_distance=5, top_k=12, smooth_k=1,
                     span_floor=0.25, max_before=10, max_after=3, with_text=False) -> list[dict]:
    peaks = find_peaks(scores, valid, threshold, min_distance, top_k, smooth_k)
    peak_set = set(peaks)
    # `rows` is keyed by the commentary's line_no, which has gaps; everything else here works on
    # positions in the match. Map once so a peak at position p reads ITS OWN line, not line_no == p.
    rows = {i: rows.get(meta[i].get("line_no", i)) or {} for i in range(len(meta))}
    out = []
    for rank, p in enumerate(peaks, 1):
        start, end = build_span(scores, valid, p, peak_set - {p}, span_floor, max_before, max_after)
        s4_class = meta[p].get("primary_class")
        if event_prob is not None:
            k = int(event_prob[p].argmax())
            event, event_p = classes[k], float(event_prob[p, k])
        else:
            event, event_p = s4_class or "routine", None
        model_player = None
        if agent_prob is not None and agent_vocab:
            model_player = agent_vocab[int(agent_prob[p].argmax())]
        player, player_src = pick_player(meta, event, p, start, end, model_player)
        row = rows.get(p) or {}
        t = next((row[k] for k in TIME_KEYS if row.get(k) not in (None, "")), None)
        rec = {
            "rank": rank, "match_id": match_id, "line_no": meta[p].get("line_no", p),
            "start_line": meta[start].get("line_no", start), "end_line": meta[end].get("line_no", end),
            "time": t, "event": event, "event_prob": event_p, "event_stage4": s4_class,
            "player": player, "player_source": player_src, "player_model": model_player,
            "method": describe_method(rows, p, start, end),
            "excitement": round(float(scores[p]), 4),
            "buildup_length": p - start,
            "narrative_context": narrative_context(meta, valid, p, start, narrative_cfg),
            "text": row.get("text", ""),
        }
        if with_text:
            rec["span_text"] = [(rows.get(i) or {}).get("text", "") for i in range(start, end + 1)]
        out.append(rec)
    return out


def render_line(r: dict) -> str:
    when = r["time"] if r.get("time") not in (None, "") else f"line {r['line_no']}"
    parts = [str(when), f"event: {str(r['event']).replace('_', ' ').upper()}"]
    if r.get("player"):
        parts.append(f"player: {r['player']}")
    if r.get("method"):
        parts.append(f"method: {r['method']}")
    n = r["buildup_length"]
    parts += [f"excitement: {r['excitement']:.2f}",
              f"buildup_length: {n} line{'s' if n != 1 else ''}",
              f'narrative_context: "{r["narrative_context"]}"']
    return "[" + ", ".join(parts) + "]"


CSV_FIELDS = ["rank", "match_id", "line_no", "start_line", "end_line", "time", "event", "event_prob",
              "event_stage4", "player", "player_source", "player_model", "method", "excitement",
              "buildup_length", "narrative_context", "text"]


def write_match(out_dir: str | Path, match_id: str, records: list[dict], params: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"highlights_{match_id}.json").write_text(
        json.dumps({"match_id": match_id, "params": params, "highlights": records},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / f"highlights_{match_id}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    lines = [f"Match {match_id}: {len(records)} highlights", "", "Ranked by excitement:"]
    for r in records:
        lines += [f"{r['rank']:2d}. {render_line(r)}", f"      \"{r['text']}\""]
    lines += ["", "In match order:"]
    for r in sorted(records, key=lambda r: r["line_no"]):
        lines.append(f"    {render_line(r)}")
    (out / f"highlights_{match_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(out_dir: str | Path, all_records: list[dict]) -> None:
    with open(Path(out_dir) / "highlights_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_records)
