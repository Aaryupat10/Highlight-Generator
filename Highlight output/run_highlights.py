"""
Stages 9 + 10: ranked highlights per match.

    python "Highlight output/run_highlights.py" --split test                 # trained model, held-out matches
    python "Highlight output/run_highlights.py" --match-id 93323
    python "Highlight output/run_highlights.py" --source weak --split all    # no model: weak labels (test 9/10 early,
                                                                              # or use as a baseline to compare against)
Writes outputs/highlights/highlights_<id>.json / .csv / .txt and highlights_all.csv.

--split test is the honest one. train/val matches were seen during training, so their
highlights look better than the model really is.

Test-time augmentation (default on): the Stage 6 model is run once per chunk offset 0..W-1 and the
scores are averaged, so the result does not depend on where chunk boundaries happen to fall.
--no-tta is W times faster.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Hierarchical model"))
sys.path.insert(0, str(HERE.parent / "Multitask heads"))

from highlight_output import build_highlights, load_events, render_line, write_all, write_match  # noqa: E402
from labels import DEFAULT_CONFIG, excitement_scores, load_config, read_meta  # noqa: E402


def load_model(ckpt_path: str, device: str):
    import torch
    from multitask import HierarchicalModel, MultiTaskModel
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a, cfg, vocab = ck["args"], ck["label_config"], ck["agent_vocab"]
    backbone = HierarchicalModel(in_dim=ck["in_dim"], d_model=a["d_model"], window=a["window"],
                                 lower=a["lower"], upper=a["upper"], dropout=a["dropout"],
                                 upper_causal=not a["upper_bidirectional"], use_kind=not a["no_kind"])
    model = MultiTaskModel(backbone, len(cfg["classes"]), len(vocab), a["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def predict(model, X: np.ndarray, meta: list[dict], window: int, device: str, tta: bool):
    import torch
    from hier_model import KINDS, UNKNOWN_KIND
    T = len(X)
    x = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    kind = torch.tensor([[KINDS.get(m.get("line_kind"), UNKNOWN_KIND) for m in meta]], device=device)
    exc = pe = pa = 0.0
    offsets = range(window) if tta else [0]
    with torch.no_grad():
        for o in offsets:
            out = model(x, mask, kind, offset=o)
            exc = exc + out["excitement"][0]
            pe = pe + torch.softmax(out["event_logits"][0], -1)
            pa = pa + torch.softmax(out["agent_logits"][0], -1)
    n = len(offsets)
    return (exc / n).cpu().numpy(), (pe / n).cpu().numpy(), (pa / n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Stage 9-10: peak detection and structured highlight output")
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default=None,
                    help="matches from the checkpoint's split (default test; 'all' with --source weak)")
    ap.add_argument("--source", choices=["model", "weak"], default="model")
    ap.add_argument("--ckpt", default="outputs/model/best.pt")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--events-dir", default="outputs/events")
    ap.add_argument("--out-dir", default="outputs/highlights")
    ap.add_argument("--label-config", default=None, help="only for --source weak; model runs use the checkpoint's")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-distance", type=int, default=5, help="lines between two highlights")
    ap.add_argument("--top-k", type=int, default=12, help="max highlights per match (0 = no limit)")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window over the scores (1 = off)")
    ap.add_argument("--span-floor", type=float, default=0.25, help="score a line needs to count as build-up")
    ap.add_argument("--max-before", type=int, default=10)
    ap.add_argument("--max-after", type=int, default=3)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--with-text", action="store_true", help="include the commentary of the whole moment in the JSON")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model = ck = None
    if a.source == "model":
        import torch
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(a.ckpt).exists():
            raise SystemExit(f"{a.ckpt} not found. Train first (Multitask heads/train.py) or use --source weak")
        model, ck = load_model(a.ckpt, device)
        cfg, classes, vocab = ck["label_config"], ck["label_config"]["classes"], ck["agent_vocab"]
        window = ck["args"]["window"]
        print(f"[model] {a.ckpt} on {device}, TTA {'off' if a.no_tta else f'x{window}'}")
    else:
        device, cfg = "cpu", load_config(a.label_config)
        classes, vocab, window = cfg["classes"], None, 1

    all_ids = sorted(p.stem[len("emb_"):-len("_meta")] for p in Path(a.emb_dir).glob("emb_*_meta.jsonl"))
    if a.match_id:
        ids = [a.match_id]
    else:
        split = a.split or ("test" if a.source == "model" else "all")
        ids = all_ids if split == "all" else (ck["split"][split] if ck else all_ids)
        print(f"[data] split '{split}': {len(ids)} matches")
        if split in ("train", "val"):
            print("[warn] the model saw these matches during training; judge it on --split test")
    if not ids:
        raise SystemExit(f"no matches found in {a.emb_dir}")

    params = {k: getattr(a, k) for k in ("source", "threshold", "min_distance", "top_k", "smooth",
                                         "span_floor", "max_before", "max_after")}
    params["tta"] = not a.no_tta
    allrec = []
    for mid in ids:
        meta = read_meta(a.emb_dir, mid)
        valid = np.array([m.get("line_kind") in cfg["live_kinds"] for m in meta], dtype=bool)
        rows = load_events(a.events_dir, mid)
        if model is not None:
            X = np.load(Path(a.emb_dir) / f"emb_{mid}.npy")
            scores, pe, pa = predict(model, X, meta, window, device, not a.no_tta)
        else:
            scores, _ = excitement_scores(meta, cfg)
            pe = pa = None
        scores = np.where(valid, scores, 0.0)
        recs = build_highlights(
            mid, scores, valid, meta, rows, classes, pe, pa, vocab, cfg.get("narrative"),
            a.threshold, a.min_distance, a.top_k or None, a.smooth, a.span_floor,
            a.max_before, a.max_after, a.with_text)
        write_match(a.out_dir, mid, recs, params)
        allrec += recs
        top = recs[0] if recs else None
        print(f"[ok] {mid}: {len(recs)} highlights" + (f"; best: {render_line(top)}" if top else
              f" (none above threshold {a.threshold}; max score {scores.max():.2f})"))
    write_all(a.out_dir, allrec)
    print(f"[done] {len(allrec)} highlights from {len(ids)} matches -> {a.out_dir}")


if __name__ == "__main__":
    main()
