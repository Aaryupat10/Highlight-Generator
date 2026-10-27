"""
Stage 7 labels - training targets for the three heads, built from Stage 4 output.

Nothing here is hand-written ground truth. Everything is derived from Stage 4's rule-based
primary_class / primary_agent, so the numbers in your report measure agreement with Stage 4 unless
you add hand-corrected lines (the "gold" column below).

  Head A  excitement  0..1   weak score from the line's event class, spread to nearby lines
  Head B  event type         Stage 4 primary_class  (goal / card / near_miss / foul / routine)
  Head C  central player     Stage 4 primary_agent  (vocabulary built from the training matches)

Only lines with line_kind == "live" carry targets; prematch / context / recap lines are ignored
by every loss and metric (target -100 / not in the live mask).

How the excitement score is made (all numbers live in DEFAULT_CONFIG, or your own --config JSON):
  1. each live line gets class_scores[primary_class]   goal 1.0, near_miss 0.7, card 0.5, foul 0.3, routine 0.05
  2. a class repeated on consecutive live lines is damped (repeat_decay ** n): Stage 4 often emits
     several goal events for one goal, and only the first should be the peak
  3. build-up and aftermath: a line at distance d before an event gets score * pre_decay ** d
     (up to pre_lines lines), after it score * post_decay ** d (up to post_lines lines). The final
     value is the max of its own score and anything spread onto it.
A different sport only needs a different config JSON (class names and scores).

CLI - write one CSV per match you can open in Excel, and hand-correct it:
    python "Multitask heads/labels.py" --emb-dir outputs/embeddings --all --events-dir outputs/events
Fill the gold_excitement column for lines you checked (0..1). train.py then uses those values in
place of the weak score and reports a separate metric on them. Existing CSVs are never overwritten
unless you pass --overwrite, so your hand edits are safe.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

IGNORE = -100

DEFAULT_CONFIG = {
    "classes": ["routine", "foul", "card", "near_miss", "goal"],
    "class_scores": {"goal": 1.0, "near_miss": 0.7, "card": 0.5, "foul": 0.3, "routine": 0.05},
    "live_kinds": ["live"],
    "pre_lines": 3, "pre_decay": 0.6,
    "post_lines": 2, "post_decay": 0.5,
    "repeat_decay": 0.8,
    "agent_min_count": 3, "max_agents": 300,
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def agent_name(v) -> str | None:
    """Stage 4 may store the agent as a string or as {'entity': ..., 'text': ...}."""
    if v is None:
        return None
    if isinstance(v, dict):
        for k in ("entity", "label", "text", "name"):
            if v.get(k):
                return str(v[k]).strip() or None
        return None
    s = str(v).strip()
    return s if s and s.lower() not in {"none", "null", "nan"} else None


def read_meta(emb_dir: str | Path, match_id: str) -> list[dict]:
    with open(Path(emb_dir) / f"emb_{match_id}_meta.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def excitement_scores(meta: list[dict], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (excitement[T] float32 in 0..1, live[T] bool)."""
    T = len(meta)
    live = np.array([m.get("line_kind") in cfg["live_kinds"] for m in meta], dtype=bool)
    scores = cfg["class_scores"]
    floor = scores.get("routine", 0.0)

    base = np.zeros(T, dtype=np.float32)
    prev, run = None, 0
    for i, m in enumerate(meta):
        if not live[i]:
            prev, run = None, 0
            continue
        c = m.get("primary_class") or "routine"
        s = float(scores.get(c, 0.0))
        run = run + 1 if (c == prev and s > floor) else 0
        base[i] = s * cfg["repeat_decay"] ** run
        prev = c

    exc = base.copy()
    for i in np.flatnonzero(live & (base > floor)):
        for d in range(1, cfg["pre_lines"] + 1):
            j = i - d
            if j >= 0 and live[j]:
                exc[j] = max(exc[j], base[i] * cfg["pre_decay"] ** d)
        for d in range(1, cfg["post_lines"] + 1):
            j = i + d
            if j < T and live[j]:
                exc[j] = max(exc[j], base[i] * cfg["post_decay"] ** d)
    exc[~live] = 0.0
    return np.clip(exc, 0.0, 1.0), live


def build_agent_vocab(metas: list[list[dict]], cfg: dict) -> list[str]:
    """Index 0 is '<other>' (rare or unseen players). Built from TRAINING matches only."""
    counts: dict[str, int] = {}
    for meta in metas:
        for m in meta:
            if m.get("line_kind") in cfg["live_kinds"]:
                a = agent_name(m.get("primary_agent"))
                if a:
                    counts[a] = counts.get(a, 0) + 1
    keep = [a for a, n in sorted(counts.items(), key=lambda kv: -kv[1])
            if n >= cfg["agent_min_count"]][: cfg["max_agents"]]
    return ["<other>"] + keep


def make_labels(meta: list[dict], cfg: dict, agent_vocab: list[str] | None = None,
                gold: dict[int, float] | None = None) -> dict:
    """gold: {line_no: excitement} hand-checked values that replace the weak score."""
    exc, live = excitement_scores(meta, cfg)
    classes = {c: i for i, c in enumerate(cfg["classes"])}
    event = np.full(len(meta), IGNORE, dtype=np.int64)
    agent = np.full(len(meta), IGNORE, dtype=np.int64)
    a_idx = {a: i for i, a in enumerate(agent_vocab or [])}
    gold_mask = np.zeros(len(meta), dtype=bool)
    for i, m in enumerate(meta):
        if not live[i]:
            continue
        c = m.get("primary_class") or "routine"
        if c in classes:
            event[i] = classes[c]
        a = agent_name(m.get("primary_agent"))
        if a and agent_vocab:
            agent[i] = a_idx.get(a, 0)
        if gold and m.get("line_no") in gold:
            exc[i] = float(gold[m["line_no"]])
            gold_mask[i] = True
    return {"excitement": exc.astype(np.float32), "live": live, "event": event,
            "agent": agent, "gold": gold_mask}


def load_gold(labels_dir: str | Path) -> dict[str, dict[int, float]]:
    """Read gold_excitement cells you filled in labels_<id>.csv. {match_id: {line_no: value}}"""
    out: dict[str, dict[int, float]] = {}
    for p in sorted(Path(labels_dir).glob("labels_*.csv")):
        mid = p.stem[len("labels_"):]
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                g = (row.get("gold_excitement") or "").strip()
                if g:
                    v = float(g)
                    if not 0.0 <= v <= 1.0:
                        raise ValueError(f"{p.name} line {row['line_no']}: gold_excitement {v} not in 0..1")
                    out.setdefault(mid, {})[int(row["line_no"])] = v
    return out


# ------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description="Write weak labels to CSV for inspection / hand-correction")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--match-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--out-dir", default="outputs/labels")
    ap.add_argument("--events-dir", default=None, help="outputs/events: adds the line text to the CSV")
    ap.add_argument("--config", default=None, help="JSON overriding DEFAULT_CONFIG (other sports)")
    ap.add_argument("--overwrite", action="store_true", help="replaces CSVs, including your gold edits")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ids = ([a.match_id] if a.match_id else
           sorted(p.stem[len("emb_"):-len("_meta")] for p in Path(a.emb_dir).glob("emb_*_meta.jsonl")))
    if not ids:
        raise SystemExit(f"no emb_*_meta.jsonl in {a.emb_dir}")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    for mid in ids:
        dst = Path(a.out_dir) / f"labels_{mid}.csv"
        if dst.exists() and not a.overwrite:
            print(f"[skip] {mid}: {dst.name} exists (use --overwrite; this would erase gold edits)")
            continue
        meta = read_meta(a.emb_dir, mid)
        exc, live = excitement_scores(meta, cfg)
        text = {}
        if a.events_dir:
            ev = Path(a.events_dir) / f"events_{mid}.jsonl"
            if ev.exists():
                with open(ev, encoding="utf-8") as f:
                    text = {r["line_no"]: r.get("text", "") for r in map(json.loads, filter(str.strip, f))}
        with open(dst, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_no", "line_kind", "primary_class", "primary_agent",
                        "weak_excitement", "gold_excitement", "text"])
            for i, m in enumerate(meta):
                w.writerow([m.get("line_no"), m.get("line_kind"), m.get("primary_class"),
                            agent_name(m.get("primary_agent")) or "", f"{exc[i]:.3f}", "",
                            text.get(m.get("line_no"), "")])
        top = int((exc >= 0.9).sum())
        print(f"[ok] {mid}: {len(meta)} lines, {int(live.sum())} live, {top} lines >= 0.9 -> {dst}")


if __name__ == "__main__":
    main()
