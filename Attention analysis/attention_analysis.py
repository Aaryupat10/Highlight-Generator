"""
Stage 8 - attention interpretability.  Reads the trained model (outputs/model/best.pt) and the saved
embeddings; trains nothing and changes nothing the pipeline reads.

    python "Attention analysis/attention_analysis.py"                 # test matches (the honest ones)
    python "Attention analysis/attention_analysis.py" --split all
    python "Attention analysis/attention_analysis.py" --match-id 93323 --top-k 6

Three questions, from the two attention mechanisms in the Stage 6 model:

  A  Lower tier (inside one chunk of 8 lines): attention pooling picks which lines make up the chunk summary.
     Measured as "attention lift" = weight x chunk size, averaged over all chunk offsets, so 1.0 means
     "as much as an average line". Reported per Stage 4 class (and per gold class when --gold is found).
     If the model looks at the right lines, goal / near_miss lines should sit above 1 and routine below 1.
     Also: how often the single most-attended line of an event chunk is itself an event line.

  B  Upper tier (chunk looks at earlier chunks, causal transformer): the attention matrix averaged over
     heads and layers. For event chunks: how much attention goes to itself, to the previous chunk, to older
     chunks, and "event lift" = share of its attention on earlier event chunks / share of earlier chunks
     that are event chunks (1.0 = no preference for build-up that contains events).

  C  Per highlight peak (Stage 9 peaks of the model's excitement curve): which earlier chunks it attends to
     (readable list with the lines), plus a faithfulness check, because attention is not an explanation by
     itself: remove the most-attended earlier chunk and see how much the peak's score moves, compared with
     removing a random earlier chunk. "attended > random" means the attention points at something the
     prediction really depends on.

Writes outputs/attention/attention_<id>.json / .txt and attention_summary.json.
Honest reading: attention is evidence, not proof; numbers on a handful of test matches are indicative.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("Highlight output", "Hierarchical model", "Multitask heads"):
    sys.path.insert(0, str(ROOT / sub))
from highlight_output import load_events  # noqa: E402
from labels import load_config, read_meta  # noqa: E402
from peak_detection import find_peaks  # noqa: E402
from run_highlights import load_model  # noqa: E402

EVENT_CLASSES = ("goal", "near_miss", "card")


# --------------------------------------------------------------------------- model access
class Capture:
    """Records the upper tier's attention weights (head-averaged) on every forward pass.

    nn.TransformerEncoderLayer asks for need_weights=False, so we wrap each layer's self_attn.forward.
    Grad must be enabled while capturing, otherwise PyTorch takes a fused fast path that never calls it.
    """

    def __init__(self, upper):
        self.store: list = []
        self.patched = []
        self.ok = getattr(upper, "kind", None) == "transformer"
        if not self.ok:
            return
        for layer in upper.enc.layers:
            sa, orig = layer.self_attn, layer.self_attn.forward

            def f(*a, _o=orig, **k):
                k["need_weights"], k["average_attn_weights"] = True, True
                out = _o(*a, **k)
                self.store.append(out[1].detach())
                return out
            sa.forward = f
            self.patched.append(sa)

    def reset(self):
        self.store.clear()

    def matrix(self):
        """[N, N] attention averaged over layers (and heads)."""
        if not self.store:
            return None
        return torch_mean(self.store)

    def close(self):
        for sa in self.patched:
            if "forward" in sa.__dict__:
                del sa.__dict__["forward"]


def torch_mean(ts):
    import torch
    return torch.stack(ts).mean(0)[0].cpu().numpy()


class Runner:
    def __init__(self, model, X, meta, device, window):
        import torch
        from hier_model import KINDS, UNKNOWN_KIND
        self.t, self.model, self.device, self.W = torch, model, device, window
        self.x = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
        self.kind = torch.tensor([[KINDS.get(m.get("line_kind"), UNKNOWN_KIND) for m in meta]], device=device)
        self.T = len(X)
        self.cap = Capture(model.backbone.upper)

    def run(self, offset=0, keep: np.ndarray | None = None):
        torch = self.t
        mask = torch.ones(1, self.T, dtype=torch.bool, device=self.device)
        if keep is not None:
            mask = torch.from_numpy(keep).unsqueeze(0).to(self.device)
        self.cap.reset()
        with torch.enable_grad():                 # keeps the fused fast path off; nothing is backpropagated
            out = self.model(self.x, mask, self.kind, offset=offset)
        return {"exc": out["excitement"][0].detach().cpu().numpy(),
                "pool": out["pool_attn"][0].detach().cpu().numpy(),
                "chunk_id": out["chunk_id"].cpu().numpy(),
                "A": self.cap.matrix()}


# --------------------------------------------------------------------------- analysis pieces
def lower_tier(R: Runner, classes: list[str | None], live: np.ndarray, event_flag: np.ndarray) -> dict:
    """Attention lift per class + top-attended-line hit rate."""
    W, T = R.W, R.T
    lift = np.zeros(T)
    hit = tot = 0
    chance_sum = 0.0
    for o in range(W):
        out = R.run(o)
        cid, pa = out["chunk_id"], out["pool"]
        size = np.bincount(cid)[cid]
        lift += pa * size
        if o == 0:
            for c in np.unique(cid):
                idx = np.flatnonzero((cid == c) & live)
                if len(idx) == 0 or not event_flag[idx].any():
                    continue
                tot += 1
                hit += bool(event_flag[idx[int(np.argmax(pa[idx]))]])
                chance_sum += event_flag[idx].mean()
    lift /= W
    by = defaultdict(list)
    for i in np.flatnonzero(live):
        by[classes[i] or "routine"].append(lift[i])
    return {"lift": lift,
            "by_class": {c: {"mean_lift": round(float(np.mean(v)), 3), "n": len(v)} for c, v in sorted(by.items())},
            "top_line_is_event": {"event_chunks": tot, "hits": int(hit),
                                  "rate": round(hit / tot, 3) if tot else None,
                                  "chance": round(chance_sum / tot, 3) if tot else None}}


def upper_tier(A: np.ndarray, chunk_event: np.ndarray, chunk_live: np.ndarray) -> dict:
    """Self / previous / older shares of attention for event chunks, and event lift."""
    N = len(A)
    rows = {"event": defaultdict(list), "other": defaultdict(list)}
    lifts = []
    for i in range(N):
        if not chunk_live[i]:
            continue
        key = "event" if chunk_event[i] else "other"
        a = A[i, : i + 1].astype(float)
        rows[key]["self"].append(a[i])
        rows[key]["prev"].append(a[i - 1] if i >= 1 else 0.0)
        rows[key]["older"].append(a[: max(i - 1, 0)].sum())
        if i >= 1:
            rows[key]["dist"].append(float(((i - np.arange(i)) * a[:i]).sum() / max(a[:i].sum(), 1e-9)))
        if key == "event" and i >= 3:
            b = a[:i] / max(a[:i].sum(), 1e-9)             # attention over strictly earlier chunks
            base = chunk_event[:i].mean()
            if base > 0:
                lifts.append(float((b * chunk_event[:i]).sum() / base))
    out = {k: {m: round(float(np.mean(v)), 3) for m, v in d.items() if v} for k, d in rows.items()}
    out["event_chunks"] = int(sum(chunk_event & chunk_live))
    out["event_lift_mean"] = round(float(np.mean(lifts)), 3) if lifts else None
    out["event_lifts"] = [round(x, 3) for x in lifts]
    return out


def describe_chunk(j: int, cid: np.ndarray, meta: list[dict], rows: dict, classes: list[str | None]) -> dict:
    idx = np.flatnonzero(cid == j)
    rank = {"goal": 3, "near_miss": 2, "card": 2, "foul": 1}
    best = max(idx, key=lambda i: (rank.get(classes[i] or "", 0), -i))
    ln = lambda i: meta[i].get("line_no", i)  # noqa: E731
    return {"chunk": int(j), "lines": [ln(idx[0]), ln(idx[-1])],
            "key_class": classes[best] or "routine", "key_line": ln(best),
            "key_text": (rows.get(ln(best)) or {}).get("text", "")[:140]}


def peak_analysis(R: Runner, scores: np.ndarray, valid: np.ndarray, meta: list[dict], rows: dict,
                  classes: list[str | None], A: np.ndarray, cid: np.ndarray, args, rng) -> list[dict]:
    peaks = find_peaks(scores, valid, args.threshold, args.min_distance, args.top_k, 1)
    base = R.run(0)["exc"]
    out = []
    for rank, p in enumerate(peaks, 1):
        c = int(cid[p])
        rec = {"rank": rank, "line_no": meta[p].get("line_no", p), "score": round(float(scores[p]), 3),
               "chunk": c, "text": (rows.get(meta[p].get("line_no", p)) or {}).get("text", "")[:140]}
        if A is not None and c >= 1:
            a = A[c, :c].astype(float)
            a = a / max(a.sum(), 1e-9)
            order = np.argsort(-a)
            rec["attends_to"] = [{**describe_chunk(int(j), cid, meta, rows, classes), "weight": round(float(a[j]), 3)}
                                 for j in order[:3]]
            rec["self_share"] = round(float(A[c, c]), 3)
            # faithfulness: drop the most attended earlier chunk vs random earlier chunks
            others = [j for j in range(c) if j != int(order[0])]
            if len(others) >= 3:
                def effect(j):
                    keep = np.ones(R.T, dtype=bool)
                    keep[cid == j] = False
                    return abs(float(R.run(0, keep)["exc"][p]) - float(base[p]))
                top = effect(int(order[0]))
                rnd = float(np.mean([effect(int(j)) for j in rng.choice(others, size=min(5, len(others)),
                                                                        replace=False)]))
                rec["faithfulness"] = {"drop_attended": round(top, 4), "drop_random_mean": round(rnd, 4),
                                       "attended_matters_more": bool(top > rnd)}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- driver
def load_gold_classes(path: Path) -> dict[str, dict[int, str]]:
    g: dict[str, dict[int, str]] = defaultdict(dict)
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("explicit") == "yes":
                    g[str(r["match_id"])][int(r["line_no"])] = r["gold_class"]
    return g


def render_txt(mid: str, res: dict) -> str:
    L = [f"Match {mid}  ({'held-out test match' if res['held_out'] else 'seen in training/validation'})", ""]
    lt = res["lower_tier"]
    L.append("Lower tier: mean attention lift per Stage 4 class (1.0 = average line)")
    L += [f"    {c:10s} {v['mean_lift']:.2f}  (n={v['n']})" for c, v in lt["by_class"].items()]
    h = lt["top_line_is_event"]
    L.append(f"    most-attended line of an event chunk is an event line: {h['hits']}/{h['event_chunks']}"
             f" (chance {h['chance']})")
    if res.get("upper_tier"):
        u = res["upper_tier"]
        L += ["", "Upper tier (event chunks): " + ", ".join(f"{k} {v}" for k, v in u["event"].items()),
              f"    event lift {u['event_lift_mean']} (1.0 = no preference for earlier chunks that contain events)"]
    L += ["", "Highlight peaks and what they attend to:"]
    for r in res["peaks"]:
        L.append(f"  #{r['rank']} line {r['line_no']} score {r['score']}: \"{r['text']}\"")
        for t in r.get("attends_to", []):
            L.append(f"        <- lines {t['lines'][0]}-{t['lines'][1]} weight {t['weight']} "
                     f"[{t['key_class']}] \"{t['key_text']}\"")
        if "faithfulness" in r:
            f = r["faithfulness"]
            L.append(f"        drop attended chunk: score moves {f['drop_attended']}; random chunk: {f['drop_random_mean']}"
                     f"  -> {'attention matters' if f['attended_matters_more'] else 'not more than a random chunk'}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Stage 8: attention analysis of the trained model")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "model" / "best.pt"))
    ap.add_argument("--emb-dir", default=str(ROOT / "outputs" / "embeddings"))
    ap.add_argument("--events-dir", default=str(ROOT / "outputs" / "events"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "attention"))
    ap.add_argument("--gold", default=str(ROOT / "Gold set" / "gold_lines.csv"))
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--max-matches", type=int, default=0, help="0 = all in the split")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-distance", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(a.ckpt).exists():
        raise SystemExit(f"{a.ckpt} not found: train first (Multitask heads/train.py)")
    model, ck = load_model(a.ckpt, device)
    window, cfg = ck["args"]["window"], ck["label_config"]
    test_ids = set(ck["split"].get("test", []))
    all_ids = sorted(p.stem[len("emb_"):-len("_meta")] for p in Path(a.emb_dir).glob("emb_*_meta.jsonl"))
    if a.match_id:
        ids = [a.match_id]
    else:
        ids = all_ids if a.split == "all" else list(ck["split"].get(a.split, []))
    if a.max_matches:
        ids = ids[: a.max_matches]
    if not ids:
        raise SystemExit("no matches to analyse")
    print(f"[data] {len(ids)} matches ({a.split}); model on {device}")
    if a.split != "test" and not a.match_id:
        print("[warn] these matches were used in training/validation; use --split test for honest statements")
    gold = load_gold_classes(Path(a.gold))
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    agg = {"lift": defaultdict(list), "gold_lift": defaultdict(list), "hit": [0, 0, 0.0], "up": defaultdict(list),
           "lifts": [], "faith": []}
    done = {}
    for mid in ids:
        meta = read_meta(a.emb_dir, mid)
        X = np.load(Path(a.emb_dir) / f"emb_{mid}.npy")
        rows = load_events(a.events_dir, mid)
        live = np.array([m.get("line_kind") in cfg["live_kinds"] for m in meta], dtype=bool)
        classes = [m.get("primary_class") if live[i] else None for i, m in enumerate(meta)]
        event_flag = np.array([c in EVENT_CLASSES for c in classes])
        R = Runner(model, X, meta, device, window)
        try:
            low = lower_tier(R, classes, live, event_flag)
            base = R.run(0)
            cid, A = base["chunk_id"], base["A"]
            N = int(cid.max()) + 1
            chunk_event = np.array([bool(event_flag[cid == j].any()) for j in range(N)])
            chunk_live = np.array([bool(live[cid == j].any()) for j in range(N)])
            up = upper_tier(A, chunk_event, chunk_live) if A is not None else None
            tta = np.mean([R.run(o)["exc"] for o in range(window)], axis=0)
            scores = np.where(live, tta, 0.0)
            peaks = peak_analysis(R, scores, live, meta, rows, classes, A, cid, a, rng)
        finally:
            R.cap.close()
        res = {"match_id": mid, "held_out": mid in test_ids,
               "lower_tier": {k: v for k, v in low.items() if k != "lift"},
               "upper_tier": up, "peaks": peaks}
        # gold-class view of the lower tier
        if gold.get(mid):
            gb = defaultdict(list)
            for i, m in enumerate(meta):
                c = gold[mid].get(m.get("line_no"))
                if c:
                    gb[c].append(low["lift"][i])
            res["lower_tier"]["by_gold_class"] = {c: {"mean_lift": round(float(np.mean(v)), 3), "n": len(v)}
                                                  for c, v in sorted(gb.items())}
            for c, v in gb.items():
                agg["gold_lift"][c] += v
        (out / f"attention_{mid}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / f"attention_{mid}.txt").write_text(render_txt(mid, res), encoding="utf-8")
        for c, v in low["by_class"].items():
            agg["lift"][c] += [v["mean_lift"]] * v["n"]
        h = low["top_line_is_event"]
        agg["hit"][0] += h["event_chunks"]; agg["hit"][1] += h["hits"]
        agg["hit"][2] += (h["chance"] or 0) * h["event_chunks"]
        if up:
            agg["lifts"] += up["event_lifts"]
            for k in ("self", "prev", "older", "dist"):
                if k in up["event"]:
                    agg["up"][k].append(up["event"][k])
        agg["faith"] += [p["faithfulness"] for p in peaks if "faithfulness" in p]
        done[mid] = res
        print(f"[ok] {mid}: {len(peaks)} peaks analysed")

    n_ev, n_hit, ch = agg["hit"]
    summary = {
        "matches": list(done), "held_out_matches": [m for m in done if done[m]["held_out"]],
        "lower_tier_lift_by_stage4_class": {c: round(float(np.mean(v)), 3) for c, v in sorted(agg["lift"].items())},
        "lower_tier_lift_by_gold_class": {c: round(float(np.mean(v)), 3) for c, v in sorted(agg["gold_lift"].items())},
        "top_attended_line_is_event": {"event_chunks": n_ev, "hits": int(n_hit),
                                       "rate": round(n_hit / n_ev, 3) if n_ev else None,
                                       "chance": round(ch / n_ev, 3) if n_ev else None},
        "upper_tier_event_chunks": {k: round(float(np.mean(v)), 3) for k, v in agg["up"].items()},
        "upper_tier_event_lift": round(float(np.mean(agg["lifts"])), 3) if agg["lifts"] else None,
        "faithfulness": {
            "peaks_tested": len(agg["faith"]),
            "attended_matters_more": int(sum(f["attended_matters_more"] for f in agg["faith"])),
            "mean_drop_attended": round(float(np.mean([f["drop_attended"] for f in agg["faith"]])), 4) if agg["faith"] else None,
            "mean_drop_random": round(float(np.mean([f["drop_random_mean"] for f in agg["faith"]])), 4) if agg["faith"] else None},
        "reading": "lift > 1 on goal/near_miss and < 1 on routine = lower tier looks at the right lines; "
                   "event lift > 1 = upper tier prefers build-up that contains events; faithfulness: "
                   "'attended_matters_more' close to all peaks = attention reflects real dependence."}
    (out / "attention_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nper-match readable files: {out}/attention_<id>.txt")


if __name__ == "__main__":
    main()
