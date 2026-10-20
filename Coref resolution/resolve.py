"""Versions A, B and C of the coreference step.

A  the pretrained model alone (each cluster is named after its most common proper name)
B  A plus the match name book: aliases, nicknames, ambiguous names, cluster voting
C  B plus football rules: hosts/visitors, "GOAL!!!" lines, role phrases, recency

Needs the baseline clusters from run_coref.py, so run that first for the same match:
    python "Coref resolution/run_coref.py" --match-number 1
    python "Coref resolution/resolve.py" --match-number 1 --stage C
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from run_coref import load_commentary, pick_match_id

HISTORY_LINES = 6        # how far back the recency rules look
AMBIGUOUS_LINES = 10     # how far back we look to settle an ambiguous name like "Gomes"
ALL_RULES = {"side", "goal", "role", "recency"}
DEFAULT_RULES = "side,goal,role"      # recency is off by default, see the notes in the README

PLAYER_PRONOUNS = {"he", "him", "his", "himself"}
TEAM_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
ROLES = {
    "striker": {"FWD"}, "forward": {"FWD"}, "attacker": {"FWD"}, "frontman": {"FWD"},
    "winger": {"MID", "FWD"}, "midfielder": {"MID"}, "defender": {"DEF"},
    "centre-back": {"DEF"}, "full-back": {"DEF"}, "keeper": {"GK"}, "goalkeeper": {"GK"},
}
HOME_WORDS = {"hosts", "home side", "home team"}
AWAY_WORDS = {"visitors", "away side", "away team"}

POSS = r"(?:'s|’s|')?"
ROLE_WORDS = "|".join(ROLES)
SIDE_WORDS = "|".join(sorted(HOME_WORDS | AWAY_WORDS, key=len, reverse=True))
ROLE_RE = re.compile(rf"^(?:the|The)\s+(?:(\S+)\s+)?({ROLE_WORDS}){POSS}$")
SIDE_RE = re.compile(rf"^(?:the|The|their|Their)\s+({SIDE_WORDS}){POSS}$")
# same phrases, but searched inside a line (used to find mentions the model skipped)
FIND_PRONOUN_RE = re.compile(r"\b(?:he|him|his|himself|they|them|their|theirs|themselves)\b", re.I)
FIND_PHRASE_RE = re.compile(
    rf"\b(?:[Tt]he|[Tt]heir)\s+(?:{SIDE_WORDS}){POSS}(?!\w)"
    rf"|\b[Tt]he\s+(?:[A-Z][\w-]*\s+)?(?:{ROLE_WORDS}){POSS}(?!\w)"
)
GOAL_RE = re.compile(r"^GOAL\W")


def classify(text: str) -> dict | None:
    """Is this mention something we resolve (pronoun, role phrase, hosts/visitors)?"""
    low = text.lower()
    if low in PLAYER_PRONOUNS:
        return {"kind": "pronoun", "slot": "player", "poss": low == "his"}
    if low in TEAM_PRONOUNS:
        return {"kind": "pronoun", "slot": "team", "poss": low in {"their", "theirs"}}
    poss = text.endswith(("'s", "’s", "'"))
    m = SIDE_RE.match(text)
    if m:
        return {"kind": "side", "side": "home" if m.group(1) in HOME_WORDS else "away", "poss": poss}
    m = ROLE_RE.match(text)
    if m:
        return {"kind": "role", "roles": ROLES[m.group(2)], "modifier": m.group(1), "poss": poss}
    return None


def normalise(text: str) -> str:
    text = re.sub(r"^(?:the|The)\s+", "", text.strip())
    return re.sub(r"(?:'s|’s|')$", "", text).strip(" ,.")


class Book:
    """One match's name book, with the lookups the resolver needs."""

    def __init__(self, book: dict):
        self.players = {p["name"]: p for p in book["players"]}
        self.teams = {t["name"]: t for t in book["teams"]}
        self.alias = {**book["alias_index"], **book["team_alias_index"]}
        self.ambiguous = {**book["ambiguous_aliases"], **book["ambiguous_team_aliases"]}
        self.alias_fold = {k.casefold(): v for k, v in self.alias.items()}
        self.ambiguous_fold = {k.casefold(): v for k, v in self.ambiguous.items()}
        self.home = next((t["name"] for t in book["teams"] if t["side"] == "home"), None)
        self.away = next((t["name"] for t in book["teams"] if t["side"] == "away"), None)

    def kind_of(self, entity: str) -> str:
        return "team" if entity in self.teams else "player"

    def canonical_of(self, text: str) -> str | None:
        # "Al Dakhil, who has had his loan made permanent" -> "Al Dakhil"
        norm = re.split(r",\s*(?:who|which)\b", normalise(text))[0].strip()
        if norm in self.alias:
            return self.alias[norm]
        if norm.isupper():                          # headline lines: HAALAND, CITY
            return self.alias_fold.get(norm.casefold())
        return None

    def display(self, entity: str) -> str:
        if entity in self.teams or entity not in self.players:
            return entity                               # a team, or a raw name from stage A
        aliases = self.players[entity].get("aliases") or [entity]
        short = min(aliases, key=len)
        return short if short in self.alias else entity


def build_name_regex(book: Book) -> re.Pattern | None:
    keys = set(book.alias) | set(book.ambiguous)
    keys |= {k.upper() for k in keys if len(k) >= 3}
    if not keys:
        return None
    ordered = sorted(keys, key=len, reverse=True)       # longest first, so "Manchester City" beats "City"
    return re.compile(r"(?<![\w'’])(" + "|".join(re.escape(k) for k in ordered) + r")(?!\w)")


def scan_names(lines, book: Book) -> list[dict]:
    """Find every player/team name in the commentary, independent of the model."""
    regex = build_name_regex(book)
    found: list[dict] = []
    if regex is None:
        return found
    for idx, (line_no, text) in enumerate(lines):
        for m in regex.finditer(text):
            word = m.group(1)
            entity = book.canonical_of(word)
            candidates = None
            if entity is None:                          # ambiguous alias ("Gomes", "United")
                candidates = book.ambiguous.get(word) or book.ambiguous_fold.get(word.casefold())
                entity = latest_of(found, candidates, idx, AMBIGUOUS_LINES) if candidates else None
            found.append({"line_idx": idx, "line_no": line_no, "start": m.start(), "end": m.end(),
                          "text": word, "entity": entity, "ambiguous": candidates is not None})
    return found


def latest_of(found, candidates, idx, window) -> str | None:
    # the candidate named most recently without ambiguity
    for nm in reversed(found):
        if idx - nm["line_idx"] > window:
            break
        if not nm["ambiguous"] and nm["entity"] in candidates:
            return nm["entity"]
    return None


def recent_entity(names, book: Book, idx, start, kind=None, roles=None, team=None) -> str | None:
    """Most recent name before (idx, start) that fits the kind / role / team asked for."""
    for nm in reversed(names):
        if (nm["line_idx"], nm["start"]) >= (idx, start):
            continue
        if idx - nm["line_idx"] > HISTORY_LINES:
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


# --------------------------------------------------------------------------
# stage A and B: turn the model's clusters into entities
# --------------------------------------------------------------------------

def rep_name(members, book: Book) -> str | None:
    # stage A: name the cluster after its most common capitalised, non-pronoun mention
    counts: Counter = Counter()
    for m in members:
        norm = normalise(m["text"])
        if norm and norm[0].isupper() and classify(m["text"]) is None and len(norm.split()) <= 3:
            counts[norm] += 1
    if not counts:
        return None
    top = counts.most_common(1)[0][0]
    return book.canonical_of(top) or top


def resolve_clusters(records, book: Book, line_idx: dict, stage: str) -> dict:
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
        if stage != "A":
            for m in members:
                e = assigned.get(key(m)) or book.canonical_of(m["text"])
                if e:
                    anchors.append((pos(m), e))
        entities = {e for _, e in anchors}
        a_name = rep_name(members, book) if stage == "A" else None

        for m in members:
            info = classify(m["text"])
            if not m["in_block"] or info is None:
                continue
            if stage == "A":
                e = a_name
            elif len(entities) == 1:
                e = next(iter(entities))
            elif entities:
                before = [x for p, x in anchors if p < pos(m)]
                e = before[-1] if before else None          # mixed cluster: nearest earlier name
            else:
                e = None
            targets[key(m)] = {"line_no": m["line_no"], "start": m["start"], "end": m["end"],
                               "text": m["text"], "info": info, "entity": e,
                               "source": "model" if stage == "A" else "cluster"}
            if e:
                assigned[key(m)] = e
    return targets


def drop_nested(targets: dict) -> None:
    # "their" inside the model's "their hosts" mention is not a separate thing to resolve
    for k, t in list(targets.items()):
        for o in targets.values():
            if o is not t and o["line_no"] == t["line_no"] and o["start"] <= t["start"] and t["end"] <= o["end"] \
                    and (o["end"] - o["start"]) > (t["end"] - t["start"]):
                del targets[k]
                break


def add_missed_targets(lines, targets: dict) -> None:
    # pronouns and role phrases the model left out because they were in no cluster
    for line_no, text in lines:
        have = {(t["start"], t["end"]) for t in targets.values() if t["line_no"] == line_no}
        for regex in (FIND_PRONOUN_RE, FIND_PHRASE_RE):
            for m in regex.finditer(text):
                info = classify(m.group(0))
                if info is None or (m.start(), m.end()) in have:
                    continue
                targets[(line_no, m.start(), m.end())] = {
                    "line_no": line_no, "start": m.start(), "end": m.end(), "text": m.group(0),
                    "info": info, "entity": None, "source": "no cluster"}


# --------------------------------------------------------------------------
# stage C: football rules
# --------------------------------------------------------------------------

def apply_rules(t: dict, text: str, idx: int, names, scorers: dict, book: Book, rules: set) -> tuple[str | None, str]:
    info, entity, source = t["info"], t["entity"], t["source"]

    if "side" in rules and info["kind"] == "side":
        team = book.home if info["side"] == "home" else book.away
        if team:
            return team, "rule:side"

    if "goal" in rules and info["kind"] == "pronoun" and info["slot"] == "player":
        # in a "GOAL!!! HAALAND ..." line, and in the two lines after it, he/him/his is the scorer,
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
        found = recent_entity(names, book, idx, t["start"], kind="player", roles=info["roles"], team=team)
        return found, "rule:role" if found else "unresolved"

    if "recency" in rules and info["kind"] == "pronoun":
        if entity is None or book.kind_of(entity) != info["slot"]:
            found = recent_entity(names, book, idx, t["start"], kind=info["slot"])
            if found:
                return found, "rule:recency"

    return entity, source


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def rewrite(text: str, targets: list[dict], book: Book) -> str:
    # replace resolved pronouns / role phrases with the entity name, right to left
    last_start = len(text) + 1
    for t in sorted(targets, key=lambda t: (t["start"], t["end"]), reverse=True):
        if t["entity"] is None or t["end"] > last_start or t["text"].lower() in {"himself", "themselves"}:
            continue
        name = book.display(t["entity"])
        if t["info"]["poss"]:
            name += "'" if name.endswith("s") else "'s"
        text = text[:t["start"]] + name + text[t["end"]:]
        last_start = t["start"]
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve coreference with the name book (B) and football rules (C).")
    ap.add_argument("--match-id", type=int)
    ap.add_argument("--match-number", type=int)
    ap.add_argument("--stage", choices=["A", "B", "C"], default="C")
    ap.add_argument("--rules", default=DEFAULT_RULES, help=f"rules used in stage C, from: {','.join(sorted(ALL_RULES))}")
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
    rules = set(filter(None, args.rules.split(","))) if args.stage == "C" else set()
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

    names = scan_names(lines, book)
    targets = resolve_clusters(records, book, line_idx, args.stage)
    if args.stage != "A":
        add_missed_targets(lines, targets)
    drop_nested(targets)
    if args.stage == "C":
        scorers = {}
        for i, (_, text) in enumerate(lines):
            if GOAL_RE.match(text):
                first = next((nm["entity"] for nm in names if nm["line_idx"] == i and nm["entity"]
                              and book.kind_of(nm["entity"]) == "player"), None)
                if first:
                    scorers[i] = first
        for t in sorted(targets.values(), key=lambda t: (line_idx[t["line_no"]], t["start"])):
            i = line_idx[t["line_no"]]
            t["entity"], t["source"] = apply_rules(t, lines[i][1], i, names, scorers, book, rules)

    out_name = f"resolved_{args.stage}_{match_id}"
    args.coref_dir.mkdir(parents=True, exist_ok=True)
    report = [f"stage {args.stage}" + (f", rules: {','.join(sorted(rules))}" if args.stage == "C" else ""), ""]
    with (args.coref_dir / f"{out_name}.jsonl").open("w", encoding="utf-8") as f:
        for idx, (line_no, text) in enumerate(lines):
            line_targets = sorted((t for t in targets.values() if t["line_no"] == line_no), key=lambda t: t["start"])
            resolved = rewrite(text, line_targets, book)
            mentions = [{"start": nm["start"], "end": nm["end"], "text": nm["text"], "kind": "name",
                         "entity": nm["entity"], "source": "alias"} for nm in names if nm["line_idx"] == idx]
            mentions += [{"start": t["start"], "end": t["end"], "text": t["text"], "kind": t["info"]["kind"],
                          "entity": t["entity"], "source": t["source"]} for t in line_targets]
            f.write(json.dumps({"match_id": match_id, "line_no": line_no, "clean_text": text,
                                "resolved_text": resolved, "mentions": mentions}, ensure_ascii=False) + "\n")
            if line_targets:
                report.append(f"{line_no:>4}  {text}")
                report.append(f"      {resolved}")
                for t in line_targets:
                    shown = book.display(t["entity"]) if t["entity"] else "?"
                    report.append(f"        {t['text']} -> {shown}  [{t['source']}]")
                report.append("")
    (args.coref_dir / f"{out_name}.txt").write_text("\n".join(report), encoding="utf-8")

    counts = Counter(t["source"] if t["entity"] else "unresolved" for t in targets.values())
    print(f"match {match_id}, stage {args.stage}: {len(targets)} mentions to resolve")
    for source, n in counts.most_common():
        print(f"  {source:<12} {n}")
    print(f"saved {args.coref_dir / (out_name + '.txt')}")


if __name__ == "__main__":
    main()
