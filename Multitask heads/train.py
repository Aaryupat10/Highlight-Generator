"""
Stage 7 training: Stage 6 backbone + the three heads, trained jointly.

    python "Multitask heads/train.py" --emb-dir outputs/embeddings

Needs Stage 5 output (emb_<id>.npy + emb_<id>_meta.jsonl) for several matches, so run
embed_lines.py --all first. Matches are split into train / val / test BY MATCH (never by line:
neighbouring lines share context and would leak). Gold values you fill in outputs/labels/*.csv
are picked up automatically.

Writes to --out-dir (default outputs/model):
    best.pt          weights + everything needed to rebuild the model (config, vocabularies, split)
    metrics.json     validation history and the final test metrics
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multitask import (LabelledMatches, MultiTaskModel, collate_labelled, evaluate,  # noqa: E402
                       multitask_loss, HierarchicalModel)
from labels import build_agent_vocab, load_config, load_gold, read_meta  # noqa: E402


def split_ids(ids, val_frac, test_frac, seed):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n - n_test - n_val < 2:
        raise SystemExit(f"only {n} matches: need at least 4 to split into train/val/test "
                         f"(run embed_lines.py --all first)")
    return ids[n_test + n_val:], ids[n_test:n_test + n_val], ids[:n_test]


def main():
    ap = argparse.ArgumentParser(description="Stage 7: train the multi-task model")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--labels-dir", default="outputs/labels", help="where hand-edited gold CSVs live")
    ap.add_argument("--out-dir", default="outputs/model")
    ap.add_argument("--label-config", default=None, help="JSON overriding the label defaults (other sports)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--lower", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--upper", choices=["gru", "transformer"], default="transformer")
    ap.add_argument("--upper-bidirectional", action="store_true")
    ap.add_argument("--no-kind", action="store_true", help="do not give the model line_kind as an input")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4, help="matches per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weights", default="1,1,0.5", help="loss weights for heads A,B,C")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--no-offset-aug", action="store_true", help="always use chunk offset 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    w = tuple(float(x) for x in a.weights.split(","))
    cfg = load_config(a.label_config)

    all_ids = sorted(p.stem[len("emb_"):-len("_meta")] for p in Path(a.emb_dir).glob("emb_*_meta.jsonl"))
    tr, va, te = split_ids(all_ids, a.val_frac, a.test_frac, a.seed)
    print(f"[data] {len(all_ids)} matches -> train {len(tr)}, val {len(va)}, test {len(te)}; device {device}")

    vocab = build_agent_vocab([read_meta(a.emb_dir, m) for m in tr], cfg)   # training matches only
    gold = load_gold(a.labels_dir) if Path(a.labels_dir).exists() else {}
    n_gold = sum(len(v) for v in gold.values())
    print(f"[labels] {len(cfg['classes'])} event classes, {len(vocab)} agent classes, {n_gold} gold lines")

    mk = lambda ids, shuffle: torch.utils.data.DataLoader(
        LabelledMatches(a.emb_dir, ids, cfg, vocab, gold), batch_size=a.batch_size, shuffle=shuffle,
        collate_fn=collate_labelled)
    L_tr, L_va, L_te = mk(tr, True), mk(va, False), mk(te, False)

    # event-class weights from the training set: inverse sqrt frequency, so 'routine' does not drown the rest
    counts = np.zeros(len(cfg["classes"]))
    for b in mk(tr, False):
        t = b["t_event"][b["t_event"] >= 0].numpy()
        counts += np.bincount(t, minlength=len(counts))
    ew = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    ew = torch.tensor(ew / ew[counts > 0].mean(), dtype=torch.float32, device=device)
    print("[labels] train event counts:", dict(zip(cfg["classes"], counts.astype(int).tolist())))

    d_in = np.load(next(Path(a.emb_dir).glob("emb_*.npy")), mmap_mode="r").shape[1]
    backbone = HierarchicalModel(in_dim=d_in, d_model=a.d_model, window=a.window, lower=a.lower,
                                 upper=a.upper, dropout=a.dropout,
                                 upper_causal=not a.upper_bidirectional, use_kind=not a.no_kind)
    model = MultiTaskModel(backbone, len(cfg["classes"]), len(vocab), a.dropout).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist, best, bad = [], float("inf"), 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for b in L_tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            off = 0 if a.no_offset_aug else random.randrange(a.window)
            loss = multitask_loss(model(b["x"], b["mask"], b["kind"], offset=off), b, w, ew)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
        v = evaluate(model, L_va, device, w, ew, len(cfg["classes"]))
        hist.append({"epoch": ep, "train_loss": run / len(L_tr), **{k: x for k, x in v.items() if k != "B_per_class"}})
        print(f"ep {ep:3d}  train {run / len(L_tr):.4f}  val {v['total']:.4f} "
              f"(A {v['A']:.4f} B {v['B']:.4f} C {v['C']:.4f})  "
              f"A-rho {v.get('A_spearman', float('nan')):.3f}  B-F1 {v.get('B_macro_f1', float('nan')):.3f}  "
              f"{time.time() - t0:.1f}s")
        if v["total"] < best - 1e-4:
            best, bad = v["total"], 0
            torch.save({"model": model.state_dict(), "args": vars(a), "label_config": cfg,
                        "agent_vocab": vocab, "in_dim": d_in, "split": {"train": tr, "val": va, "test": te}},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[stop] no improvement for {a.patience} epochs")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    test = evaluate(model, L_te, device, w, ew, len(cfg["classes"]))
    test["B_per_class"] = {cfg["classes"][c]: m for c, m in test.get("B_per_class", {}).items()}
    (out_dir / "metrics.json").write_text(json.dumps({"history": hist, "test": test}, indent=2), encoding="utf-8")

    print("\n== TEST (best checkpoint, chunk offset 0) ==")
    print(f"Head A  MAE {test.get('A_mae', float('nan')):.3f} (predict-the-mean: {test.get('A_mae_mean_baseline', float('nan')):.3f})"
          f"   Pearson {test.get('A_pearson', float('nan')):.3f}   Spearman {test.get('A_spearman', float('nan')):.3f}")
    if "A_gold_n" in test:
        print(f"        on {test['A_gold_n']} hand-checked lines: MAE {test['A_gold_mae']:.3f}  Spearman {test['A_gold_spearman']:.3f}")
    print(f"Head B  acc {test.get('B_acc', float('nan')):.3f} (majority class: {test.get('B_majority_acc', float('nan')):.3f})"
          f"   macro-F1 {test.get('B_macro_f1', float('nan')):.3f}")
    for c, m in test["B_per_class"].items():
        print(f"          {c:10s} recall {m['recall']:.2f}  precision {m['precision']:.2f}  n={m['support']}")
    print(f"Head C  acc {test.get('C_acc', float('nan')):.3f} (always-most-common: {test.get('C_majority_acc', float('nan')):.3f})"
          f"   on {test.get('C_n', 0)} lines with a player")
    print(f"saved {out_dir / 'best.pt'} and metrics.json")


if __name__ == "__main__":
    main()
