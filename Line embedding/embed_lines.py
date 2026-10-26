"""
Stage 5 - Line-level embedding (NLP -> DL bridge)

Reads Stage 4 output  : outputs/events/events_<match_id>.jsonl
Writes per match      : outputs/embeddings/emb_<match_id>.npy        float32 [n_lines, dim]
                        outputs/embeddings/emb_<match_id>_meta.jsonl  one row per embedding row
Writes once           : outputs/embeddings/embedding_config.json     how the vectors were made

Each commentary line becomes ONE vector. The encoder input is a sentence pair:

    segment A = the (coref-resolved) line text
    segment B = the line's extracted events, flattened to text, e.g.
                "action: scored | agent: Salah | method: header | source: Trent's cross"

so the vector reflects both what was said and the structure Stage 4 pulled out of it.

Modes (for ablations in the report):
    --mode both    text + structure   (default, the pipeline as designed)
    --mode text    text only          (baseline: how much does Stage 4 add?)
    --mode struct  structure only

Row i of the .npy is row i of the _meta.jsonl, and rows are sorted by line_no.
Stage 6 should read the two files together and use line_kind to decide which
lines can be highlight targets (only "live" lines count).

Run (heavy model, on your own machine):
    python "Line embedding/embed_lines.py" --match-id 93323
    python "Line embedding/embed_lines.py" --all
    python "Line embedding/embed_lines.py" --all --mode text --out-dir outputs/embeddings_text

--encoder hash is a tiny dependency-free stand-in for testing the plumbing only.
Its vectors carry no real meaning, never train on them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

# Event fields that describe WHAT HAPPENED in a line. Order = order in the flattened string.
# type / class / primary are Stage 4's own labels; leaving them out keeps the Stage 7
# event-type head from just reading its answer off the input (see --struct-fields).
DEFAULT_STRUCT_FIELDS = ["action", "agent", "other", "team", "method", "source", "outcome", "modality"]

META_KEYS = [
    "match_id", "line_no", "line_kind", "primary_type", "primary_class",
    "primary_agent", "primary_team",
]


# ----------------------------------------------------------------------------- io
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}: bad JSON on line {i}: {e}")
    return rows


def find_matches(events_dir: Path) -> list[str]:
    ids = []
    for p in sorted(events_dir.glob("events_*.jsonl")):
        ids.append(p.stem[len("events_"):])
    return ids


# ------------------------------------------------------------- structure -> text
def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):
        # Stage 4 stores slots as dicts, e.g. {'entity': 'Erling Haaland', 'text': 'HAALAND', ...}
        # or {'label': 'header', 'text': 'nods', ...}. Prefer the canonical name / label.
        for key in ("entity", "label", "text", "name", "value"):
            if v.get(key):
                return _clean(v[key])
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(_clean(x) for x in v if _clean(x))
    s = str(v).strip()
    return "" if s.lower() in {"none", "null", "nan", "unknown"} else s


def flatten_event(ev: dict, fields: list[str]) -> str:
    parts = []
    for k in fields:
        v = _clean(ev.get(k))
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


def flatten_events(row: dict, fields: list[str], max_events: int) -> str:
    """Primary event first, then the rest in original order, capped at max_events."""
    events = row.get("events") or []
    if not events:
        return "no event"
    ordered = sorted(events, key=lambda e: 0 if e.get("primary") else 1)  # stable
    texts = [t for t in (flatten_event(e, fields) for e in ordered[:max_events]) if t]
    return " ; ".join(texts) if texts else "no event"


def build_inputs(rows: list[dict], mode: str, text_field: str, fields: list[str], max_events: int):
    a, b = [], []
    for r in rows:
        text = _clean(r.get(text_field))
        struct = flatten_events(r, fields, max_events)
        if mode == "text":
            a.append(text); b.append(None)
        elif mode == "struct":
            a.append(struct); b.append(None)
        else:
            a.append(text); b.append(struct)
    return a, b


# ---------------------------------------------------------------------- encoders
class HashEncoder:
    """Deterministic bag-of-words hashing. Plumbing test only - no semantics."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, a: list[str], b: list[str | None], **_) -> np.ndarray:
        out = np.zeros((len(a), self.dim), dtype=np.float32)
        for i, (x, y) in enumerate(zip(a, b)):
            for tok in re.findall(r"\w+", f"{x} {y or ''}".lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class HFEncoder:
    """Frozen transformer encoder + masked mean pooling (or CLS)."""

    def __init__(self, model_name: str, pooling: str, max_length: int, device: str | None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise SystemExit("Needs torch and transformers:  pip install torch transformers")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.pooling = pooling
        self.max_length = max_length
        self.name = model_name
        self.dim = self.model.config.hidden_size
        print(f"[encoder] {model_name} on {self.device}, dim={self.dim}, pooling={pooling}, "
              f"max_length={max_length}")

    def encode(self, a: list[str], b: list[str | None], batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        n = len(a)
        # sort by length so batches pad little, then restore order
        order = sorted(range(n), key=lambda i: len(a[i]) + len(b[i] or ""))
        out = np.zeros((n, self.dim), dtype=np.float32)
        use_amp = self.device.startswith("cuda")
        with torch.no_grad():
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                ta = [a[i] for i in idx]
                tb = [b[i] for i in idx]
                kw = dict(padding=True, truncation=True, max_length=self.max_length,
                          return_tensors="pt")
                enc = self.tok(ta, tb, **kw) if all(x is not None for x in tb) else self.tok(ta, **kw)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    h = self.model(**enc).last_hidden_state
                h = h.float()
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[idx] = v.cpu().numpy()
        return out


# -------------------------------------------------------------------------- main
def process_match(match_id, args, encoder, fields):
    src = Path(args.events_dir) / f"events_{match_id}.jsonl"
    dst = Path(args.out_dir) / f"emb_{match_id}.npy"
    meta_dst = Path(args.out_dir) / f"emb_{match_id}_meta.jsonl"
    if dst.exists() and not args.overwrite:
        print(f"[skip] {match_id}: {dst.name} exists (use --overwrite)")
        return None
    if not src.exists():
        print(f"[miss] {match_id}: {src} not found")
        return None

    rows = read_jsonl(src)
    if not rows:
        print(f"[empty] {match_id}: no lines")
        return None
    rows.sort(key=lambda r: r.get("line_no", 0))
    line_nos = [r.get("line_no") for r in rows]
    if len(set(line_nos)) != len(line_nos):
        raise SystemExit(f"{src}: duplicate line_no values, Stage 4 output looks wrong")
    empty_text = sum(1 for r in rows if not _clean(r.get(args.text_field)))
    if empty_text == len(rows):
        raise SystemExit(f"{src}: field '{args.text_field}' is empty on every line; "
                         f"check --text-field (keys seen: {sorted(rows[0].keys())})")

    a, b = build_inputs(rows, args.mode, args.text_field, fields, args.max_events)
    emb = encoder.encode(a, b, batch_size=args.batch_size) if isinstance(encoder, HFEncoder) \
        else encoder.encode(a, b)
    assert emb.shape == (len(rows), encoder.dim), emb.shape
    if not np.isfinite(emb).all():
        raise SystemExit(f"{match_id}: NaN/inf in embeddings")
    if args.normalize:
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.save(dst, emb.astype(np.float32))
    with open(meta_dst, "w", encoding="utf-8") as f:
        for r in rows:
            m = {k: r.get(k) for k in META_KEYS}
            m["match_id"] = m["match_id"] if m["match_id"] is not None else match_id
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    kinds = {}
    for r in rows:
        kinds[r.get("line_kind")] = kinds.get(r.get("line_kind"), 0) + 1
    with_ev = sum(1 for r in rows if r.get("events"))
    print(f"[ok] {match_id}: {len(rows)} lines -> {dst.name} {emb.shape}  "
          f"(with events: {with_ev}, kinds: {kinds})")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Stage 5: line-level embeddings")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--match-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--events-dir", default="outputs/events")
    ap.add_argument("--out-dir", default="outputs/embeddings")
    ap.add_argument("--mode", choices=["both", "text", "struct"], default="both")
    ap.add_argument("--encoder", choices=["hf", "hash"], default="hf")
    ap.add_argument("--model", default="bert-base-uncased",
                    help="any HF encoder, e.g. distilroberta-base, "
                         "sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--struct-fields", default=",".join(DEFAULT_STRUCT_FIELDS),
                    help="comma list of event keys to flatten. Add 'type,class,outcome' only if "
                         "you accept the label leakage into the Stage 7 heads")
    ap.add_argument("--max-events", type=int, default=3, help="events flattened per line")
    ap.add_argument("--normalize", action="store_true", help="L2-normalise vectors")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="print the encoder inputs for the first lines and exit (no model needed)")
    args = ap.parse_args()

    fields = [f.strip() for f in args.struct_fields.split(",") if f.strip()]
    ids = find_matches(Path(args.events_dir)) if args.all else [args.match_id]
    if not ids:
        raise SystemExit(f"No events_*.jsonl in {args.events_dir}")

    if args.preview:
        rows = sorted(read_jsonl(Path(args.events_dir) / f"events_{ids[0]}.jsonl"),
                      key=lambda r: r.get("line_no", 0))
        a, b = build_inputs(rows, args.mode, args.text_field, fields, args.max_events)
        shown = 0
        for r, x, y in zip(rows, a, b):
            if r.get("events") or shown < 3:
                print(f"--- line {r.get('line_no')} [{r.get('line_kind')}]\n  A: {x}\n  B: {y}")
                shown += 1
            if shown >= 8:
                break
        return

    if args.encoder == "hash":
        encoder = HashEncoder()
        print("[encoder] HASH stand-in: plumbing test only, vectors are meaningless")
    else:
        encoder = HFEncoder(args.model, args.pooling, args.max_length, args.device)

    total = 0
    for mid in ids:
        n = process_match(mid, args, encoder, fields)
        total += n or 0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.out_dir) / "embedding_config.json"
    cfg = {
        "encoder": encoder.name, "dim": encoder.dim, "mode": args.mode,
        "pooling": args.pooling if args.encoder == "hf" else None,
        "max_length": args.max_length, "text_field": args.text_field,
        "struct_fields": fields, "max_events": args.max_events,
        "normalized": args.normalize, "dtype": "float32",
    }
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg:
            print(f"[warn] {cfg_path.name} differs from this run's settings; embeddings in this "
                  f"folder may be mixed. Use a separate --out-dir per setting.")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[done] {len(ids)} match(es), {total} lines")


if __name__ == "__main__":
    sys.exit(main())
