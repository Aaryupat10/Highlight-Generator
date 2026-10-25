"""Stages B and C of the coreference step: the name book and the sport rules.

Stage A (the pretrained model alone) is run_coref.py, which also holds the helpers used here.

B  the model's clusters plus the match name book: aliases, nicknames, ambiguous names,
   cluster voting, and pronouns / role phrases the model left out of every cluster
C  B plus the sport rules: hosts/visitors, scorer lines ("GOAL!!!" in football), role phrases, recency

The sport's words and patterns (pronouns, role words, home/away words, scorer-line pattern,
look-back windows) are read from a config file, configs/football.json by default.
Use --config to point at another sport's file. See configs/README.md.

Needs the baseline clusters from run_coref.py, so run that first for the same match:
    python "Coref resolution/run_coref.py" --match-number 1
    python "Coref resolution/resolve.py" --match-number 1 --stage C
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_coref import (Book, classify, drop_nested, load_commentary, pick_match_id, scan_names,
                       write_resolved)
from sport_config import DEFAULT_CONFIG, SportConfig, load_config

ALL_RULES = {"side", "goal", "role", "recency"}     # rule names; the words they use come from the config


# --------------------------------------------------------------------------
# stage B: name book
# --------------------------------------------------------------------------

def resolve_clusters(records, book: Book, line_idx: dict, cfg: SportConfig) -> dict:
    """Return {(line_no, start, end): {"entity", "source", "info", ...}} for every target mention."""
    assigned: dict = {}
    targets: dict = {}
    for rec in records:
        members = [m for m in rec["mentions"] if m["line_no"] in line_idx]
        members.sort(key=lambda m: (line_idx[m["line_no"]], m["start"]))
        pos = lambda m: (line_idx[m["line_no"]], m["start"])
        key = lambda m: (m["line_no"], m["start"], m["end"])

        # entities this cluster already points to: earlier answers and plain names
        anchors = []
        for m in members:
            e = assigned.get(key(m)) or book.canonical_of(m["text"])
            if e:
                anchors.append((pos(m), e))
        entities = {e for _, e in anchors}

        for m in members:
            info = classify(m["text"], cfg)
            if not m["in_block"] or info is None:
                continue
            if len(entities) == 1:
                e = next(iter(entities))
            elif entities:
                before = [x for p, x in anchors if p < pos(m)]
                e = before[-1] if before else None          # mixed cluster: nearest earlier name
            else:
                e = None
            targets[key(m)] = {"line_no": m["line_no"], "start": m["start"], "end": m["end"],
                               "text": m["text"], "info": info, "entity": e,
                               "source": "cluster"}
            if e:
                assigned[key(m)] = e
    return targets


def add_missed_targets(lines, targets: dict, cfg: SportConfig) -> None:
    # pronouns and role phrases the model left out because they were in no cluster
    for line_no, text in lines:
        have = {(t["start"], t["end"]) for t in targets.values() if t["line_no"] == line_no}
        for regex in (cfg.find_pronoun_re, cfg.find_phrase_re):
            if regex is None:
                continue
            for m in regex.finditer(text):
                info = classify(m.group(0), cfg)
                if info is None or (m.start(), m.end()) in have:
                    continue
                targets[(line_no, m.start(), m.end())] = {
                    "line_no": line_no, "start": m.start(), "end": m.end(), "text": m.group(0),
                    "info": info, "entity": None, "source": "no cluster"}


# --------------------------------------------------------------------------
# stage C: sport rules (the words and patterns they use come from the config)
# --------------------------------------------------------------------------

def recent_entity(names, book: Book, idx, start, window, kind=None, roles=None, team=None) -> str | None:
    """Most recent name before (idx, start), within `window` lines, that fits the kind / role / team asked for."""
    for nm in reversed(names):
        if (nm["line_idx"], nm["start"]) >= (idx, start):
            continue
        if idx - nm["line_idx"] > window:
            break
        e = nm["entity"]
        if e is None or (kind and book.kind_of(e) != kind):
            continue
        if roles or team:
            p = book.players.get(e)
            if p is None or (roles and p["role"] not in roles) or (team and p["team"] != team):
                continue
        return e
    return None


def find_scorers(lines, names, book: Book, cfg: SportConfig) -> dict:
    """{line index: player} for every scorer line ("GOAL!!! HAALAND ..." in football)."""
    scorers = {}
    for i, (_, text) in enumerate(lines):
        if cfg.scorer_line_re and cfg.scorer_line_re.match(text):
            first = next((nm["entity"] for nm in names if nm["line_idx"] == i and nm["entity"]
                          and book.kind_of(nm["entity"]) == "player"), None)
            if first:
                scorers[i] = first
    return scorers


def apply_rules(t: dict, text: str, idx: int, names, scorers: dict, book: Book, rules: set,
                cfg: SportConfig) -> tuple[str | None, str]:
    info, entity, source = t["info"], t["entity"], t["source"]

    if "side" in rules and info["kind"] == "side":
        team = book.home if info["side"] == "home" else book.away
        if team:
            return team, "rule:side"

    if "goal" in rules and info["kind"] == "pronoun" and info["slot"] == "player":
        # in a scorer line ("GOAL!!! HAALAND ..." in football), and in the two lines after it,
        # the player pronoun is the scorer,
        # unless another player has already been named earlier in the same line
        scorer = next((scorers[i] for i in (idx, idx - 1, idx - 2) if i in scorers), None)
        earlier = {nm["entity"] for nm in names if nm["line_idx"] == idx and nm["start"] < t["start"]
                   and nm["entity"] and book.kind_of(nm["entity"]) == "player"}
        if scorer and earlier <= {scorer}:
            return scorer, "rule:goal"

    if "role" in rules and info["kind"] == "role":
        team = book.canonical_of(info["modifier"]) if info["modifier"] else None
        team = team if team in book.teams else None
        p = book.players.get(entity) if entity else None
        fits = p and p["role"] in info["roles"] and (team is None or p["team"] == team)
        if fits:
            return entity, source
        found = recent_entity(names, book, idx, t["start"], cfg.history_lines,
                              kind="player", roles=info["roles"], team=team)
        return found, "rule:role" if found else "unresolved"

    if "recency" in rules and info["kind"] == "pronoun":
        if entity is None or book.kind_of(entity) != info["slot"]:
            found = recent_entity(names, book, idx, t["start"], cfg.history_lines, kind=info["slot"])
            if found:
                return found, "rule:recency"

    return entity, source


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve coreference with the name book (stage B) and the sport rules (stage C).")
    ap.add_argument("--match-id", type=int)
    ap.add_argument("--match-number", type=int)
    ap.add_argument("--stage", choices=["B", "C"], default="C", help="stage A is run_coref.py")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sport config (JSON), see configs/README.md")
    ap.add_argument("--rules", default=None,
                    help=f"rules used in stage C, from: {','.join(sorted(ALL_RULES))} (default: default_rules in the config)")
    ap.add_argument("--csv", type=Path, default=Path("data/23_24_commentary_clean.csv"))
    ap.add_argument("--line-types", type=Path, default=Path("outputs/name_book/line_types.csv"))
    ap.add_argument("--name-books", type=Path, default=Path("outputs/name_book/name_books.json"))
    ap.add_argument("--coref-dir", type=Path, default=Path("outputs/coref"))
    ap.add_argument("--model", default="fcoref", help="which baseline run to read")
    ap.add_argument("--max-lines", type=int, default=80, help="use the same value as run_coref.py")
    args = ap.parse_args()

    if (args.match_id is None) == (args.match_number is None):
        ap.error("give exactly one of --match-id or --match-number")
    match_id = args.match_id if args.match_id is not None else pick_match_id(args.csv, args.match_number)
    cfg = load_config(args.config)
    rules_arg = args.rules if args.rules is not None else cfg.default_rules
    rules = set(filter(None, rules_arg.split(","))) if args.stage == "C" else set()
    if not rules <= ALL_RULES:
        ap.error(f"unknown rules: {sorted(rules - ALL_RULES)}; choose from {sorted(ALL_RULES)}")

    baseline = args.coref_dir / f"baseline_{args.model}_{match_id}.jsonl"
    if not baseline.exists():
        raise SystemExit(f"{baseline} not found; run run_coref.py for this match first")
    records = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines() if line]
    lines = load_commentary(args.csv, args.line_types, match_id)
    if args.max_lines:
        lines = lines[:args.max_lines]
    line_idx = {line_no: i for i, (line_no, _) in enumerate(lines)}
    book = Book(json.loads(args.name_books.read_text(encoding="utf-8"))[str(match_id)])

    names = scan_names(lines, book, cfg)
    targets = resolve_clusters(records, book, line_idx, cfg)           # stage B
    add_missed_targets(lines, targets, cfg)
    drop_nested(targets)
    if args.stage == "C":                                              # stage C: the sport rules
        scorers = find_scorers(lines, names, book, cfg)
        for t in sorted(targets.values(), key=lambda t: (line_idx[t["line_no"]], t["start"])):
            i = line_idx[t["line_no"]]
            t["entity"], t["source"] = apply_rules(t, lines[i][1], i, names, scorers, book, rules, cfg)

    write_resolved(args.stage, rules, match_id, lines, names, targets, book, cfg, args.coref_dir)



if __name__ == "__main__":
    main()
