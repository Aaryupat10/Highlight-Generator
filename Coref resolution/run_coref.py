"""Baseline coreference, stage A: the pretrained neural model alone.

No name book voting and no sport rules. For one match this does two things:
  1. runs the model on the commentary and saves the clusters it finds
     (baseline_<model>_<match>.jsonl / .txt), so we can read through them
  2. names each cluster after its most common proper name and rewrites the text
     (resolved_A_<match>.jsonl / .txt)

Run from the repo root (after name_book.py has produced line_types.csv and name_books.json):
    python "Coref resolution/run_coref.py" --match-number 1
Add --resolve-only to skip the model and redo step 2 from the saved clusters.

Stages B and C (name book + sport rules, read from configs/football.json) are in resolve.py,
which reuses the helpers defined below.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from sport_config import DEFAULT_CONFIG, SportConfig, load_config


# --------------------------------------------------------------------------
# baseline: run the pretrained model
# --------------------------------------------------------------------------

def pick_match_id(csv_path: Path, number: int) -> int:
    # match number 1 is the lowest match id that has commentary, 2 the next, and so on
    df = pd.read_csv(csv_path, dtype={"clean_text": str}).fillna({"clean_text": ""})
    ids = sorted(df[df.clean_text.str.strip() != ""].match_id.unique())
    if not 1 <= number <= len(ids):
        raise SystemExit(f"--match-number must be between 1 and {len(ids)}")
    return int(ids[number - 1])


def load_commentary(csv_path: Path, types_path: Path, match_id: int) -> list[tuple[int, str]]:
    df = pd.read_csv(csv_path, dtype={"clean_text": str}).fillna({"clean_text": ""})
    types = pd.read_csv(types_path)
    df = df[df.match_id == match_id].merge(types, on=["match_id", "line_no"], how="left")
    df = df[(df.line_type == "commentary") & (df.clean_text.str.strip() != "")]
    df = df.sort_values("line_no")
    return list(zip(df.line_no.astype(int), df.clean_text))


def build_text(lines: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    # join the lines with a space and remember where each line starts and ends
    spans, parts, pos = [], [], 0
    for _, text in lines:
        spans.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text) + 1
    return " ".join(parts), spans


def line_of(offset: int, spans: list[tuple[int, int]]) -> int | None:
    for i, (start, end) in enumerate(spans):
        if start <= offset < end:
            return i
    return None


def make_blocks(n_lines: int, size: int, context: int):
    # each line is a target exactly once; the lines before the block are only context
    for a in range(0, n_lines, size):
        yield max(0, a - context), a, min(a + size, n_lines)


def run_match(model, lines, size: int = 4, context: int = 4) -> list[dict]:
    blocks = list(make_blocks(len(lines), size, context))
    texts, all_spans = [], []
    for c, _, b in blocks:
        text, spans = build_text(lines[c:b])
        texts.append(text)
        all_spans.append(spans)

    preds = model.predict(texts=texts)

    results, warned = [], False
    for (c, a, b), text, spans, pred in zip(blocks, texts, all_spans, preds):
        clusters = pred.get_clusters(as_strings=False)

        # sanity check: our slicing of the offsets should match the model's own strings
        if not warned and [[text[s:e] for s, e in cl] for cl in clusters] != pred.get_clusters(as_strings=True):
            print("warning: character offsets do not line up with the model's strings")
            warned = True

        for cid, cluster in enumerate(clusters):
            mentions = []
            for s, e in cluster:
                i = line_of(s, spans)
                if i is None:
                    continue
                line_idx = c + i
                mentions.append({
                    "line_no": lines[line_idx][0],
                    "start": s - spans[i][0],
                    "end": e - spans[i][0],
                    "text": text[s:e],
                    "in_block": a <= line_idx < b,
                })
            if any(m["in_block"] for m in mentions):
                results.append({"block_start_line": lines[a][0], "cluster": cid, "mentions": mentions})
    return results


def render(lines, results, size: int) -> str:
    # readable report: each line with its mentions tagged [text|cluster], then the clusters
    by_block: dict[int, list[dict]] = {}
    for r in results:
        by_block.setdefault(r["block_start_line"], []).append(r)

    out = ["Cluster ids (c0, c1...) are local to each block of lines.", ""]
    for a in range(0, len(lines), size):
        block = lines[a:a + size]
        clusters = by_block.get(block[0][0], [])
        for line_no, text in block:
            # one insert per bracket, applied right to left so offsets stay valid and
            # nested mentions come out as [[a|c1] b|c2]
            inserts = []
            for r in clusters:
                for m in r["mentions"]:
                    if m["line_no"] == line_no and m["in_block"]:
                        inserts.append((m["start"], 0, "["))
                        inserts.append((m["end"], 1, f"|c{r['cluster']}]"))
            for pos, _, mark in sorted(inserts, reverse=True):
                text = text[:pos] + mark + text[pos:]
            out.append(f"{line_no:>4}  {text}")
        for r in clusters:
            members = " | ".join(
                f"{m['text']} ({m['line_no']}{'' if m['in_block'] else ', context'})"
                for m in r["mentions"]
            )
            out.append(f"        c{r['cluster']}: {members}")
        out.append("")
    return "\n".join(out)


def load_model(name: str, device: str):
    from fastcoref import FCoref, LingMessCoref
    from fastcoref.coref_models.modeling_fcoref import FCorefModel
    from fastcoref.coref_models.modeling_lingmess import LingMessModel

    # fastcoref predates transformers 5, which expects every model to have this attribute
    for cls in (FCorefModel, LingMessModel):
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = {}

    return (FCoref if name == "fcoref" else LingMessCoref)(device=device)


# --------------------------------------------------------------------------
# building blocks shared with resolve.py (stages B and C)
# --------------------------------------------------------------------------

def classify(text: str, cfg: SportConfig) -> dict | None:
    """Is this mention something we resolve (pronoun, role phrase, hosts/visitors)?"""
    low = text.lower()
    if low in cfg.player_pronouns:
        return {"kind": "pronoun", "slot": "player", "poss": low in cfg.possessive_pronouns}
    if low in cfg.team_pronouns:
        return {"kind": "pronoun", "slot": "team", "poss": low in cfg.possessive_pronouns}
    poss = text.endswith(("'s", "’s", "'"))
    m = cfg.side_re.match(text) if cfg.side_re else None
    if m:
        return {"kind": "side", "side": "home" if m.group(1) in cfg.home_words else "away", "poss": poss}
    m = cfg.role_re.match(text) if cfg.role_re else None
    if m:
        return {"kind": "role", "roles": cfg.roles[m.group(2)], "modifier": m.group(1), "poss": poss}
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


def scan_names(lines, book: Book, cfg: SportConfig) -> list[dict]:
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
                entity = latest_of(found, candidates, idx, cfg.ambiguous_lines) if candidates else None
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


def drop_nested(targets: dict) -> None:
    # "their" inside the model's "their hosts" mention is not a separate thing to resolve
    for k, t in list(targets.items()):
        for o in targets.values():
            if o is not t and o["line_no"] == t["line_no"] and o["start"] <= t["start"] and t["end"] <= o["end"] \
                    and (o["end"] - o["start"]) > (t["end"] - t["start"]):
                del targets[k]
                break


def rewrite(text: str, targets: list[dict], book: Book, cfg: SportConfig) -> str:
    # replace resolved pronouns / role phrases with the entity name, right to left
    last_start = len(text) + 1
    for t in sorted(targets, key=lambda t: (t["start"], t["end"]), reverse=True):
        if t["entity"] is None or t["end"] > last_start or t["text"].lower() in cfg.reflexive_pronouns:
            continue
        name = book.display(t["entity"])
        if t["info"]["poss"]:
            name += "'" if name.endswith("s") else "'s"
        text = text[:t["start"]] + name + text[t["end"]:]
        last_start = t["start"]
    return text


def write_resolved(stage: str, rules: set, match_id: int, lines, names, targets: dict,
                   book: Book, cfg: SportConfig, coref_dir: Path) -> None:
    """Write resolved_<stage>_<match>.jsonl (every mention) and .txt (readable report)."""
    out_name = f"resolved_{stage}_{match_id}"
    coref_dir.mkdir(parents=True, exist_ok=True)
    report = [f"stage {stage}" + (f", rules: {','.join(sorted(rules))}" if stage == "C" else ""), ""]
    with (coref_dir / f"{out_name}.jsonl").open("w", encoding="utf-8") as f:
        for idx, (line_no, text) in enumerate(lines):
            line_targets = sorted((t for t in targets.values() if t["line_no"] == line_no), key=lambda t: t["start"])
            resolved = rewrite(text, line_targets, book, cfg)
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
    (coref_dir / f"{out_name}.txt").write_text("\n".join(report), encoding="utf-8")

    counts = Counter(t["source"] if t["entity"] else "unresolved" for t in targets.values())
    print(f"match {match_id}, stage {stage}: {len(targets)} mentions to resolve")
    for source, n in counts.most_common():
        print(f"  {source:<12} {n}")
    print(f"saved {coref_dir / (out_name + '.txt')}")


# --------------------------------------------------------------------------
# stage A: the pretrained model alone
# --------------------------------------------------------------------------

def rep_name(members, book: Book, cfg: SportConfig) -> str | None:
    # stage A: name the cluster after its most common capitalised, non-pronoun mention
    counts: Counter = Counter()
    for m in members:
        norm = normalise(m["text"])
        if norm and norm[0].isupper() and classify(m["text"], cfg) is None and len(norm.split()) <= 3:
            counts[norm] += 1
    if not counts:
        return None
    top = counts.most_common(1)[0][0]
    return book.canonical_of(top) or top


def resolve_clusters(records, book: Book, line_idx: dict, cfg: SportConfig) -> dict:
    """Return {(line_no, start, end): {"entity", "source", "info", ...}} for every target mention."""
    targets: dict = {}
    for rec in records:
        members = [m for m in rec["mentions"] if m["line_no"] in line_idx]
        members.sort(key=lambda m: (line_idx[m["line_no"]], m["start"]))
        key = lambda m: (m["line_no"], m["start"], m["end"])

        a_name = rep_name(members, book, cfg)

        for m in members:
            info = classify(m["text"], cfg)
            if not m["in_block"] or info is None:
                continue
            targets[key(m)] = {"line_no": m["line_no"], "start": m["start"], "end": m["end"],
                               "text": m["text"], "info": info, "entity": a_name,
                               "source": "model"}
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline neural coreference (stage A) on one match.")
    ap.add_argument("--match-id", type=int, help="the dataset's own match id, e.g. 93323")
    ap.add_argument("--match-number", type=int, help="1, 2, 3... counted from the lowest match id")
    ap.add_argument("--csv", type=Path, default=Path("data/23_24_commentary_clean.csv"))
    ap.add_argument("--line-types", type=Path, default=Path("outputs/name_book/line_types.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/coref"))
    ap.add_argument("--name-books", type=Path, default=Path("outputs/name_book/name_books.json"))
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sport config (JSON), see configs/README.md")
    ap.add_argument("--model", choices=["fcoref", "lingmess"], default="fcoref")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    ap.add_argument("--max-lines", type=int, default=80, help="0 means the whole match")
    ap.add_argument("--block", type=int, default=4, help="lines resolved per step")
    ap.add_argument("--context", type=int, default=4, help="earlier lines given as context")
    ap.add_argument("--resolve-only", action="store_true",
                    help="skip the model; read the saved clusters and redo the stage A naming")
    args = ap.parse_args()

    if (args.match_id is None) == (args.match_number is None):
        ap.error("give exactly one of --match-id or --match-number")
    if args.match_number is not None:
        args.match_id = pick_match_id(args.csv, args.match_number)

    lines = load_commentary(args.csv, args.line_types, args.match_id)
    if args.max_lines:
        lines = lines[:args.max_lines]
    if not lines:
        raise SystemExit(f"no commentary lines found for match {args.match_id}")

    stem = f"baseline_{args.model}_{args.match_id}"
    baseline = args.out_dir / f"{stem}.jsonl"
    if args.resolve_only:
        if not baseline.exists():
            raise SystemExit(f"{baseline} not found; run without --resolve-only first")
        results = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines() if line]
    else:
        device = args.device
        if device is None:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"match {args.match_id}: {len(lines)} lines, model={args.model}, device={device}")

        model = load_model(args.model, device)
        results = run_match(model, lines, args.block, args.context)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        with baseline.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"match_id": args.match_id, **r}, ensure_ascii=False) + "\n")
        (args.out_dir / f"{stem}.txt").write_text(render(lines, results, args.block), encoding="utf-8")
        print(f"{len(results)} clusters; saved {args.out_dir / (stem + '.txt')}")

    # stage A: name each cluster after its most common proper name
    if not args.name_books.exists():
        print(f"{args.name_books} not found; run name_book.py first to get resolved_A output")
        return
    cfg = load_config(args.config)
    book = Book(json.loads(args.name_books.read_text(encoding="utf-8"))[str(args.match_id)])
    line_idx = {line_no: i for i, (line_no, _) in enumerate(lines)}
    names = scan_names(lines, book, cfg)
    targets = resolve_clusters(results, book, line_idx, cfg)
    drop_nested(targets)
    write_resolved("A", set(), args.match_id, lines, names, targets, book, cfg, args.out_dir)


if __name__ == "__main__":
    main()
