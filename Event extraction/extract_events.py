"""Stage 4: semantic role labelling / event extraction.

Turns every commentary line into structured events:

    {type, class, action, agent, other, team, source, method, outcome, modality, trigger}

    agent    who performs it ("Haaland" in "Haaland fires a shot")
    action   the trigger words ("fires")
    other    the second person involved (the keeper who saved, the player fouled, the player replaced)
    source   where it came from (an assist, a cross, a set piece)
    method   how (header, left foot, volley, long range ...)
    outcome  what happened (saved, wide, hit the post, scored ...)

It reads the Stage 3 output (resolved_C_<match>.jsonl): mentions already point to the right
player or team, so "he" and "the striker" arrive as names. Each mention is swapped for a short
token (@P3 / @T1) and the rules in the sport config look for triggers and fill the roles
around them. Nothing here knows about a particular sport: the words, the event types and the
patterns come from configs/<sport>.json (see configs/README.md).

Run from the repo root:
    python "Event extraction/extract_events.py" --match-id 93323
    python "Event extraction/extract_events.py" --all
    python "Event extraction/extract_events.py" --match-id 93323 --bio      # also writes tagger training data
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from event_config import DEFAULT_CONFIG, EventConfig, EventType, load_config

TOKEN = re.compile(r"@([PT])(\d+)('s)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"'”]?\s+")
WORD = re.compile(r"\w+(?:[-']\w+)*")


# --------------------------------------------------------------------------
# step 1: swap every mention for a token, keeping a map back to the original text
# --------------------------------------------------------------------------

class Line:
    """One commentary line with its mentions masked as @P0, @T1 ..."""

    def __init__(self, rec: dict, book: dict | None, cfg: EventConfig):
        self.text = rec["clean_text"]
        self.ents: list[dict] = []
        out: list[str] = []
        self.o_start: list[int] = []        # for each masked character: where it starts / ends in the original
        self.o_end: list[int] = []
        pos = 0

        def literal(chunk: str, at: int) -> None:
            out.append(chunk)
            self.o_start.extend(range(at, at + len(chunk)))
            self.o_end.extend(range(at + 1, at + len(chunk) + 1))

        for m in sorted((m for m in rec.get("mentions", []) if m.get("entity")),
                        key=lambda m: (m["start"], -(m["end"] - m["start"]))):
            if m["start"] < pos:
                continue                    # overlaps one we already took
            literal(self.text[pos:m["start"]], pos)
            kind = entity_kind(m, book)
            tok = f"@{kind}{len(self.ents)}"
            if kind == "P" and m["text"].lower() in cfg.possessive_pronouns:
                tok += "'s"                 # "his shot" must not read like "Haaland shot"
            self.ents.append({"id": len(self.ents), "kind": kind, "entity": m["entity"],
                              "text": m["text"], "span": [m["start"], m["end"]]})
            out.append(tok)
            self.o_start.extend([m["start"]] * len(tok))
            self.o_end.extend([m["end"]] * len(tok))
            pos = m["end"]
        literal(self.text[pos:], pos)
        self.masked = "".join(out)

    def orig(self, ms: int, me: int) -> list[int]:
        return [self.o_start[ms], self.o_end[me - 1]]

    def sentences(self) -> list[tuple[int, int]]:
        spans, last = [], 0
        for m in SENT_SPLIT.finditer(self.masked):
            spans.append((last, m.start() + 1))
            last = m.end()
        spans.append((last, len(self.masked)))
        return [(a, b) for a, b in spans if self.masked[a:b].strip()]

    def tokens_in(self, lo: int, hi: int) -> list[dict]:
        found = []
        for m in TOKEN.finditer(self.masked, lo, hi):
            e = self.ents[int(m.group(2))]
            found.append({**e, "ms": m.start(), "me": m.end(), "poss": bool(m.group(3))})
        return found


def entity_kind(m: dict, book: dict | None) -> str:
    e = m["entity"]
    if book:
        if e in book["player_team"]:
            return "P"
        if e in book["team_names"]:
            return "T"
    return "T" if m.get("kind") == "side" else "P"


def load_books(path: Path) -> dict:
    if not path.exists():
        return {}
    books = {}
    for mid, b in json.loads(path.read_text(encoding="utf-8")).items():
        books[mid] = {"player_team": {p["name"]: p["team"] for p in b["players"]},
                      "team_names": {t["name"] for t in b["teams"]}}
    return books


# --------------------------------------------------------------------------
# step 2: find triggers
# --------------------------------------------------------------------------

def find_triggers(line: Line, lo: int, hi: int, cfg: EventConfig) -> list[tuple[EventType, int, int]]:
    s = line.masked
    hits = []
    for ev in cfg.events:
        for rx in ev.triggers:
            for m in rx.finditer(s, lo, hi):
                around = s[max(lo, m.start() - 25):min(hi, m.end() + 15)]
                if any(x.search(around) for x in ev.exclude):
                    continue
                hits.append((ev, m.start(), m.end()))
    # overlapping triggers: the heavier event wins
    hits.sort(key=lambda h: (-h[0].weight, h[1]))
    kept: list[tuple[EventType, int, int]] = []
    for h in hits:
        if not any(h[1] < k[2] and k[1] < h[2] for k in kept):
            kept.append(h)
    kept.sort(key=lambda h: h[1])
    # one event type can suppress another in the same sentence (a disallowed goal is not a goal)
    present = {h[0].name for h in kept}
    return [h for h in kept
            if not any(h[0].name in cfg.suppress.get(p, []) for p in present)]


# --------------------------------------------------------------------------
# step 3: fill the roles
# --------------------------------------------------------------------------

def clause_start(line: Line, lo: int, ts: int, cfg: EventConfig) -> int:
    start = lo
    for m in cfg.clause_breakers.finditer(line.masked, lo, ts):
        start = m.end()
    return start


def pick(line: Line, strategy: str, lo: int, hi: int, ts: int, te: int, cfg: EventConfig,
         taken: set, kind: str, next_sent: tuple[int, int] | None) -> dict | None:
    """Return the token chosen by one strategy, or None."""
    toks = [t for t in line.tokens_in(lo, hi) if t["kind"] == kind and t["id"] not in taken]
    s = line.masked

    def words_between(a: int, b: int) -> int:
        return len(WORD.findall(s[a:b]))

    if strategy == "subject":
        # nearest mention before the trigger, in the same clause, not the object of a preposition
        cs = clause_start(line, lo, ts, cfg)
        for t in reversed([t for t in toks if t["me"] <= ts and t["ms"] >= cs]):
            gap = words_between(t["me"], ts)
            if gap > cfg.max_gap_words or cfg.clause_breakers.search(s, t["me"], ts):
                continue
            if cfg.new_subject and cfg.new_subject.search(s, t["me"], ts):
                continue                    # "he ..., a defender is well-placed to block": new subject
            if t["poss"] and gap > 2:
                continue                    # "Iwobi's block ... the effort": Iwobi belongs to the block
            if cfg.non_subject_prepositions.search(s[max(lo, t["ms"] - 14):t["ms"]]) \
               and not (kind == "T" and gap == 0) and not re.match(r"\s*to\b", s[t["me"]:ts]):
                continue                    # (a team right before the trigger is fine: "a Newcastle corner")
            return t
        return None
    if strategy.startswith("prep:"):
        # a mention right after "<word> ": "saved by Pope", "offside against Havertz", "straight at Neto"
        word = strategy[5:]
        for t in toks:
            if t["ms"] >= te and re.search(r"\b%s\s+(?:the\s+)?$" % re.escape(word), s[max(te, t["ms"] - len(word) - 6):t["ms"]], re.I) \
               and words_between(te, t["ms"]) <= cfg.max_gap_words and not cfg.clause_breakers.search(s, te, t["ms"]):
                return t
        return None
    if strategy == "object":
        for t in toks:
            if t["ms"] >= te and words_between(te, t["ms"]) <= 3 and not cfg.clause_breakers.search(s, te, t["ms"]):
                return t
        return None
    if strategy == "for":
        for t in toks:                      # "a corner for Chelsea", "Chelsea's corner"
            if t["ms"] >= te and re.search(r"\bfor\s+$", s[max(te, t["ms"] - 6):t["ms"]], re.I):
                return t
            if t["me"] <= ts and t["poss"] and words_between(t["me"], ts) <= 1:
                return t
        return None
    if strategy == "headline":
        # "GOAL!" on its own: the player is named in the next sentence ("DIAZ SCORES!")
        if next_sent and words_between(lo, hi) <= cfg.short_sentence_words:
            nxt = [t for t in line.tokens_in(*next_sent) if t["kind"] == kind and t["id"] not in taken]
            return nxt[0] if nxt else None
        return None
    if strategy == "any":
        return toks[0] if toks else None
    if strategy == "nearest":
        if not toks:
            return None
        return min(toks, key=lambda t: min(abs(t["ms"] - te), abs(ts - t["me"])))
    return None


def team_of(ent: dict | None, book: dict | None) -> str | None:
    if not ent:
        return None
    if ent["kind"] == "T":
        return ent["entity"]
    return (book or {}).get("player_team", {}).get(ent["entity"])


def role_dict(line: Line, tok: dict | None, via: str) -> dict | None:
    if not tok:
        return None
    return {"entity": tok["entity"], "text": tok["text"], "span": tok["span"], "via": via}


def modality_of(line: Line, lo: int, hi: int, ts: int, cfg: EventConfig) -> str:
    cs = clause_start(line, lo, ts, cfg)
    prefix = line.masked[cs:ts]
    for label, rx in cfg.modality:
        if rx.search(prefix):
            return label
    if line.masked[lo:hi].rstrip().endswith("?"):
        return "question"
    return "realised"


def build_event(line: Line, ev: EventType, ts: int, te: int, si: int, nxt_trig: int | None,
                prev_end: int, sents: list, book: dict | None, cfg: EventConfig) -> dict:
    lo, hi = sents[si]
    next_sent = sents[si + 1] if si + 1 < len(sents) else None
    s = line.masked
    modality = modality_of(line, lo, hi, ts, cfg)
    if modality in ("unrealised", "attempted") and ev.unrealised_type:
        ev, modality = cfg.by_name[ev.unrealised_type], "realised"      # "tries to curl it in" is a chance, not a goal
    roles: dict[str, dict | None] = {"agent": None, "other": None}
    taken: set = set()

    # 1. entity-aware patterns with named groups, when one covers the trigger
    for rx in ev.patterns:
        for m in rx.finditer(s, lo, hi):
            if m.start() <= ts and te <= m.end():
                for role in ("agent", "other"):
                    if role in rx.groupindex and m.group(role):
                        t = TOKEN.search(m.group(role))
                        ent = line.ents[int(t.group(2))]
                        roles[role] = {"entity": ent["entity"], "text": ent["text"],
                                       "span": ent["span"], "via": "pattern"}
                        taken.add(ent["id"])
                break
        if roles["agent"]:
            break

    # 2. otherwise the strategies listed for the event type
    passive = bool(re.match(r"\s*by\b", s[te:te + 6]))      # "his shot is blocked by ...": the subject is not the doer
    for role, strategies in (("agent", ev.agent), ("other", ev.other)):
        for st in ([] if roles[role] else strategies):
            if passive and st == "subject" and role == "agent":
                continue
            tok = pick(line, st, lo, hi, ts, te, cfg, taken, "P", next_sent)
            if tok:
                roles[role] = role_dict(line, tok, st)
                taken.add(tok["id"])
                break

    # team
    team = None
    for st in ev.team:
        if st == "agent_team":
            team = team_of(roles["agent"] and {"kind": "P", "entity": roles["agent"]["entity"]}, book)
        elif st == "subject_team":
            tok = pick(line, "subject", lo, hi, ts, te, cfg, set(), "T", next_sent)
            team = tok["entity"] if tok else None
        elif st == "for_team":
            tok = pick(line, "for", lo, hi, ts, te, cfg, set(), "T", next_sent)
            team = tok["entity"] if tok else None
        elif st == "any_team":
            tok = pick(line, "any", lo, hi, ts, te, cfg, set(), "T", next_sent)
            team = tok["entity"] if tok else None
        if team:
            break

    # method / outcome / source are read from this event's slice of the sentence
    mslice = (max(lo, prev_end), min(hi, nxt_trig) if nxt_trig else hi)
    methods = []
    for label, rx in ([] if ev.name in cfg.method_skip else cfg.methods):
        m = rx.search(s, *mslice)
        if m:
            methods.append({"label": label, "text": m.group(0), "span": line.orig(m.start(), m.end())})
    oslice = (ts, min(hi, nxt_trig) if nxt_trig else hi)
    outcome = None
    for o in cfg.outcomes:
        if "*" not in o["for"] and ev.name not in o["for"]:
            continue
        m = o["regex"].search(s, *oslice)
        if m:
            outcome = {"label": o["label"], "text": m.group(0), "span": line.orig(m.start(), m.end()),
                       "class": o["class"]}
            break
    if outcome is None and ev.default_outcome:
        outcome = {"label": ev.default_outcome, "text": None, "span": None, "class": None}
    source = None
    for sc in cfg.sources:
        if "*" not in sc["for"] and ev.name not in sc["for"]:
            continue
        m = sc["regex"].search(s, lo, hi)
        if m:
            who = None
            for g in sc["regex"].groupindex:
                if g.startswith("who") and m.group(g):
                    t = TOKEN.search(m.group(g))
                    if t:
                        who = line.ents[int(t.group(2))]["entity"]
                        break
            source = {"label": sc["label"], "who": who, "text": re.sub(r"@[PT]\d+('s)?",
                      lambda x: line.ents[int(TOKEN.match(x.group(0)).group(2))]["text"], m.group(0)),
                      "span": line.orig(m.start(), m.end())}
            break

    return {
        "type": ev.name,
        "class": ev.cls,
        "action": re.sub(r"@[PT]\d+('s)?", lambda x: line.ents[int(TOKEN.match(x.group(0)).group(2))]["text"],
                         s[ts:te]),
        "trigger_span": line.orig(ts, te),
        "sentence": si,
        "agent": roles["agent"], "other": roles["other"], "team": team,
        "source": source, "method": methods, "outcome": outcome,
        "modality": modality,
        "_ts": ts, "_te": te,
    }


# --------------------------------------------------------------------------
# step 4: one line -> events
# --------------------------------------------------------------------------

def line_kind(text: str, cfg: EventConfig) -> str:
    if any(r.search(text) for r in cfg.live_markers):
        return "live"
    if any(r.search(text) for r in cfg.context_markers):
        return "context"
    return "live"


def finish_class(e: dict, cfg: EventConfig, live: bool) -> str:
    ev = cfg.by_name[e["type"]]
    cls = ev.cls
    if e["outcome"] and e["outcome"]["class"] and ev.upgradable:
        cls = e["outcome"]["class"]
    if e["modality"] != "realised":
        cls = cfg.classes[-1]
        if e["modality"] == "attempted":
            cls = ev.cls                    # "tries to curl it in": still a real attempt
            if e["outcome"] and e["outcome"]["class"] and ev.upgradable:
                cls = e["outcome"]["class"]
        elif e["modality"] == "appeal" and ev.appeal_class:
            cls = ev.appeal_class
    if not live:
        cls = cfg.classes[-1]               # a stat or a recap is not a live moment
    return cls


def extract_match(recs: list[dict], book: dict | None, cfg: EventConfig) -> list[dict]:
    """Lines in order. After a half-time / full-time call, lines that summarise the past are marked recap."""
    # everything before the line that says play has started is team news and preview, not live play
    start = next((i for i, r in enumerate(recs[:cfg.prematch_search_lines])
                  if any(x.search(r["clean_text"]) for x in cfg.prematch_end)), None)
    results, recap_left = [], 0
    for i, rec in enumerate(recs):
        text = rec["clean_text"]
        if start is not None and i < start:
            results.append(extract_line(rec, book, cfg, kind="prematch"))
            continue
        if recap_left and any(r.search(text) for r in cfg.recap_end):
            recap_left = 0                      # play has restarted
        results.append(extract_line(rec, book, cfg, kind="recap" if recap_left else None))
        recap_left = max(recap_left - 1, 0)
        for rx, max_lines in cfg.recap_start:   # what follows "HALF-TIME" summarises the past
            if rx.search(text):
                recap_left = max_lines
    return results


def extract_line(rec: dict, book: dict | None, cfg: EventConfig, kind: str | None = None) -> dict:
    line = Line(rec, book, cfg)
    kind = kind or line_kind(line.text, cfg)
    sents = line.sentences()
    events: list[dict] = []
    for si, (lo, hi) in enumerate(sents):
        hits = find_triggers(line, lo, hi, cfg)
        for i, (ev, ts, te) in enumerate(hits):
            nxt = hits[i + 1][1] if i + 1 < len(hits) else None
            prev_end = hits[i - 1][2] if i else lo
            events.append(build_event(line, ev, ts, te, si, nxt, prev_end, sents, book, cfg))

    # an earlier shot is "saved" / "blocked" when a later event in the sentence says so
    for i, e in enumerate(events):
        if cfg.by_name[e["type"]].upgradable and e["outcome"] is None:
            for later in events[i + 1:]:
                if later["sentence"] == e["sentence"] and later["type"] in cfg.outcome_from_event:
                    lab = cfg.outcome_from_event[later["type"]]
                    o = next((o for o in cfg.outcomes if o["label"] == lab), None)
                    e["outcome"] = {"label": lab, "text": None, "span": None, "class": o and o["class"]}
                    break

    # "GOAL!" and "DIAZ SCORES!" are one goal: of the events of a merge type keep one per player,
    # and keep the player-less one only when there is no player at all
    keep, seen = [], set()
    for e in events:
        if not cfg.by_name[e["type"]].merge:
            keep.append(e)
            continue
        who = e["agent"]["entity"] if e["agent"] else None
        has_named = any(o["type"] == e["type"] and o["agent"] for o in events)
        if (who is None and has_named) or (e["type"], who) in seen:
            continue
        seen.add((e["type"], who))
        keep.append(e)
    events = keep

    live = kind == "live"          # prematch, context and recap lines describe other times, not the moment
    for e in events:
        e["class"] = finish_class(e, cfg, live)
    rank = {c: i for i, c in enumerate(cfg.classes)}
    best = min(events, key=lambda e: (rank[e["class"]], -cfg.by_name[e["type"]].weight, e["_ts"]), default=None)
    for e in events:
        e["primary"] = e is best
        e.pop("_ts"), e.pop("_te")

    return {
        "match_id": rec["match_id"], "line_no": rec["line_no"], "text": rec["clean_text"],
        "line_kind": kind, "n_events": len(events), "events": events,
        "primary_type": best["type"] if best else "none",
        "primary_class": best["class"] if best else cfg.classes[-1],
        "primary_agent": (best["agent"] or {}).get("entity") if best else None,
        "primary_team": best["team"] if best else None,
    }


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def label(role: dict | None) -> str:
    return role["entity"] if role else "-"


def write_outputs(match_id, results: list[dict], out_dir: Path, bio: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"events_{match_id}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = ["match_id", "line_no", "line_kind", "primary", "type", "class", "action", "agent", "other", "team",
            "source", "method", "outcome", "modality", "text"]
    with (out_dir / f"events_{match_id}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            for e in r["events"]:
                w.writerow([r["match_id"], r["line_no"], r["line_kind"], int(e["primary"]), e["type"], e["class"],
                            e["action"], label(e["agent"]), label(e["other"]), e["team"] or "",
                            (e["source"] or {}).get("label", ""), "|".join(m["label"] for m in e["method"]),
                            (e["outcome"] or {}).get("label", ""), e["modality"], r["text"]])

    with (out_dir / f"events_{match_id}.txt").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['line_no']}] ({r['line_kind']}) {r['text']}\n")
            for e in r["events"]:
                bits = [f"agent={label(e['agent'])}", f"team={e['team'] or '-'}"]
                if e["other"]:
                    bits.append(f"other={label(e['other'])}")
                if e["source"]:
                    bits.append(f"source={e['source']['label']}" + (f"({e['source']['who']})" if e["source"]["who"] else ""))
                if e["method"]:
                    bits.append("method=" + "+".join(m["label"] for m in e["method"]))
                if e["outcome"]:
                    bits.append(f"outcome={e['outcome']['label']}")
                if e["modality"] != "realised":
                    bits.append(f"modality={e['modality']}")
                f.write(f"    {'*' if e['primary'] else ' '} {e['class']:<9} {e['type']:<16} {e['action']!r}  " + "  ".join(bits) + "\n")
            f.write("\n")

    if bio:
        write_bio(match_id, results, out_dir)


def write_bio(match_id, results: list[dict], out_dir: Path) -> None:
    """Silver training data for a token tagger: one record per event, BIO tags over its sentence."""
    n = 0
    with (out_dir / f"silver_bio_{match_id}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            if r["line_kind"] != "live":
                continue                    # previews, stats and recaps are not training examples
            text = r["text"]
            sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*", text)]
            for e in r["events"]:
                if e["modality"] != "realised":
                    continue
                spans = [("ACTION", e["trigger_span"])]
                if e["agent"]:
                    spans.append(("AGENT", e["agent"]["span"]))
                if e["other"]:
                    spans.append(("OTHER", e["other"]["span"]))
                if e["source"]:
                    spans.append(("SOURCE", e["source"]["span"]))
                spans += [("METHOD", m["span"]) for m in e["method"]]
                if e["outcome"] and e["outcome"]["span"]:
                    spans.append(("OUTCOME", e["outcome"]["span"]))
                # the sentence that holds the trigger
                a, b = next(((a, b) for a, b in sents if a <= e["trigger_span"][0] < b), (0, len(text)))
                toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\w+(?:[-']\w+)*|[^\w\s]", text)
                        if a <= m.start() < b]
                tags = ["O"] * len(toks)
                for role, (s, t) in spans:
                    first = True
                    for i, (_, ts, te) in enumerate(toks):
                        if ts < t and te > s and tags[i] == "O":
                            tags[i] = ("B-" if first else "I-") + role
                            first = False
                f.write(json.dumps({"match_id": r["match_id"], "line_no": r["line_no"], "event_type": e["type"],
                                    "tokens": [t[0] for t in toks], "tags": tags}, ensure_ascii=False) + "\n")
                n += 1
    print(f"  silver BIO records: {n}")


def summarise(results: list[dict]) -> None:
    n = len(results)
    live = [r for r in results if r["line_kind"] == "live"]
    print(f"lines: {n}   live: {len(live)}   context (stats / previews): "
          f"{sum(r['line_kind'] == 'context' for r in results)}   recap (after half / full time): "
          f"{sum(r['line_kind'] == 'recap' for r in results)}   prematch: "
          f"{sum(r['line_kind'] == 'prematch' for r in results)}")
    print(f"lines with at least one event: {sum(bool(r['n_events']) for r in results)}")
    print("primary class (live lines):", dict(Counter(r["primary_class"] for r in live).most_common()))
    types = Counter(e["type"] for r in results for e in r["events"])
    print("event types:", dict(types.most_common()))
    evs = [e for r in live for e in r["events"] if e["modality"] == "realised"]
    for t in sorted({e["type"] for e in evs}):
        sub = [e for e in evs if e["type"] == t]
        fill = lambda k: sum(bool(e[k]) for e in sub) / len(sub)
        print(f"  {t:<16} n={len(sub):<4} agent {fill('agent'):4.0%}  team {fill('team'):4.0%}  "
              f"other {fill('other'):4.0%}  outcome {fill('outcome'):4.0%}  method {fill('method'):4.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured events from the Stage 3 output.")
    ap.add_argument("--match-id", type=int)
    ap.add_argument("--all", action="store_true", help="every resolved_<stage>_*.jsonl in --coref-dir")
    ap.add_argument("--input", type=Path, help="a resolved_*.jsonl file, instead of --match-id")
    ap.add_argument("--stage", choices=["A", "B", "C"], default="C", help="which coref stage output to read")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sport config (JSON), see configs/README.md")
    ap.add_argument("--coref-dir", type=Path, default=Path("outputs/coref"))
    ap.add_argument("--name-books", type=Path, default=Path("outputs/name_book/name_books.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/events"))
    ap.add_argument("--bio", action="store_true", help="also write silver BIO tags for training a token tagger")
    args = ap.parse_args()

    if sum(x is not None and x is not False for x in (args.match_id, args.input, args.all or None)) != 1:
        ap.error("give exactly one of --match-id, --input or --all")
    if args.input:
        files = [args.input]
    elif args.all:
        files = sorted(args.coref_dir.glob(f"resolved_{args.stage}_*.jsonl"))
    else:
        files = [args.coref_dir / f"resolved_{args.stage}_{args.match_id}.jsonl"]
    missing = [f for f in files if not f.exists()]
    if missing or not files:
        raise SystemExit(f"not found: {missing or args.coref_dir}; run the coref step first")

    cfg = load_config(args.config)
    books = load_books(args.name_books)
    all_results = []
    for f in files:
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
        if not recs:
            continue
        mid = recs[0]["match_id"]
        book = books.get(str(mid))
        results = extract_match(recs, book, cfg)
        write_outputs(mid, results, args.out_dir, args.bio)
        all_results += results
        if len(files) == 1:
            summarise(results)
    if len(files) > 1:
        print(f"matches: {len(files)}")
        summarise(all_results)
    print(f"saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
