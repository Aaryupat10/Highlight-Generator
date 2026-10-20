"""Baseline coreference run: pretrained neural model only, no football rules yet.

Runs on one match, skips lineup/subs lines, and writes the clusters it finds so
we can read through them and see what the model gets wrong.

Run from the repo root (after name_book.py has produced line_types.csv):
    python "Coref resolution/run_coref.py" --match-number 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline neural coreference on one match.")
    ap.add_argument("--match-id", type=int, help="the dataset's own match id, e.g. 93323")
    ap.add_argument("--match-number", type=int, help="1, 2, 3... counted from the lowest match id")
    ap.add_argument("--csv", type=Path, default=Path("data/23_24_commentary_clean.csv"))
    ap.add_argument("--line-types", type=Path, default=Path("outputs/name_book/line_types.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/coref"))
    ap.add_argument("--model", choices=["fcoref", "lingmess"], default="fcoref")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    ap.add_argument("--max-lines", type=int, default=80, help="0 means the whole match")
    ap.add_argument("--block", type=int, default=4, help="lines resolved per step")
    ap.add_argument("--context", type=int, default=4, help="earlier lines given as context")
    args = ap.parse_args()

    if (args.match_id is None) == (args.match_number is None):
        ap.error("give exactly one of --match-id or --match-number")
    if args.match_number is not None:
        args.match_id = pick_match_id(args.csv, args.match_number)

    device = args.device
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lines = load_commentary(args.csv, args.line_types, args.match_id)
    if args.max_lines:
        lines = lines[:args.max_lines]
    if not lines:
        raise SystemExit(f"no commentary lines found for match {args.match_id}")
    print(f"match {args.match_id}: {len(lines)} lines, model={args.model}, device={device}")

    model = load_model(args.model, device)
    results = run_match(model, lines, args.block, args.context)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"baseline_{args.model}_{args.match_id}"
    with (args.out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"match_id": args.match_id, **r}, ensure_ascii=False) + "\n")
    (args.out_dir / f"{stem}.txt").write_text(render(lines, results, args.block), encoding="utf-8")
    print(f"{len(results)} clusters; saved {args.out_dir / (stem + '.txt')}")


if __name__ == "__main__":
    main()
