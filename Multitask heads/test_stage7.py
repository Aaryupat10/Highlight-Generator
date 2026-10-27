"""Checks for Stage 7 on synthetic data (no real matches needed):  python "Multitask heads/test_stage7.py"

1. label rules: exact scores for a hand-built match
2. dataset / collate / model / loss shapes, gradients reach all three heads
3. an end-to-end train.py run on fake matches whose embeddings really encode the class,
   which must learn it (otherwise something in the pipeline is broken)
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import DEFAULT_CONFIG, make_labels, excitement_scores, build_agent_vocab, load_gold  # noqa: E402
from multitask import (LabelledMatches, MultiTaskModel, collate_labelled, multitask_loss,  # noqa: E402
                       HierarchicalModel)

fails = []


def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {extra}")
    if not ok:
        fails.append(name)


def row(n, cls, kind="live", agent=None):
    return {"match_id": "m", "line_no": n, "line_kind": kind, "primary_type": cls, "primary_class": cls,
            "primary_agent": agent, "primary_team": None}


# ------------------------------------------------------------------ 1. label rules
meta = [row(0, "routine", "prematch"), row(1, "routine"), row(2, "routine"), row(3, "routine"),
        row(4, "goal", agent={"entity": "Erling Haaland"}), row(5, "goal", agent="Erling Haaland"),
        row(6, "routine"), row(7, "routine"), row(8, "routine"), row(9, "card", agent="Rodri")]
exc, live = excitement_scores(meta, DEFAULT_CONFIG)
check("non-live line scores 0", exc[0] == 0 and not live[0])
check("goal line scores 1.0", abs(exc[4] - 1.0) < 1e-6)
check("repeated goal line damped to 0.8", abs(exc[5] - 0.8) < 1e-6, f"(got {exc[5]:.3f})")
check("1 line before goal = 0.6", abs(exc[3] - 0.6) < 1e-6)
check("2 lines before goal = 0.36", abs(exc[2] - 0.36) < 1e-6)
check("3 lines before goal = 0.216", abs(exc[1] - 0.216) < 1e-6)
check("after the repeat goal: +1 line 0.4", abs(exc[6] - 0.4) < 1e-6, f"(got {exc[6]:.3f})")
quiet, _ = excitement_scores([row(i, "routine") for i in range(6)], DEFAULT_CONFIG)
check("a match with no events stays at the routine floor 0.05", bool(np.allclose(quiet, 0.05)))
check("card 0.5, line before it 0.3", abs(exc[9] - 0.5) < 1e-6 and abs(exc[8] - 0.3) < 1e-6)
check("scores stay in [0,1]", exc.min() >= 0 and exc.max() <= 1)

vocab = build_agent_vocab([meta * 2], DEFAULT_CONFIG)
check("agent vocab: '<other>' first, Haaland kept (dict and str forms merged)",
      vocab[0] == "<other>" and "Erling Haaland" in vocab, str(vocab))
lab = make_labels(meta, DEFAULT_CONFIG, vocab, gold={3: 0.9})
check("gold overrides weak score", abs(lab["excitement"][3] - 0.9) < 1e-6 and lab["gold"][3] and lab["gold"].sum() == 1)
check("non-live lines get ignore targets", lab["event"][0] == -100 and lab["agent"][0] == -100)

# ------------------------------------------------------------ 2. shapes, loss, grads
tmp = Path(tempfile.mkdtemp())
emb = tmp / "embeddings"; emb.mkdir()
CLS = DEFAULT_CONFIG["classes"]
rng = np.random.default_rng(0)
PLAYERS = ["Haaland", "Salah", "Saka", "Rodri", "Son", "Foden"]


def fake_match(mid, n=64):
    kinds = ["prematch"] * 6 + ["live"] * (n - 12) + ["recap"] * 6
    cls = [rng.choice(CLS, p=[.55, .15, .1, .1, .1]) if k == "live" else "routine" for k in kinds]
    agent = [str(rng.choice(PLAYERS)) if (k == "live" and c != "routine") else None for k, c in zip(kinds, cls)]
    X = rng.normal(size=(n, 48)).astype(np.float32)
    for i, c in enumerate(cls):                          # plant the class in the embedding
        X[i, CLS.index(c)] += 4.0
        if agent[i]:
            X[i, 10 + PLAYERS.index(agent[i])] += 4.0
    np.save(emb / f"emb_{mid}.npy", X)
    with open(emb / f"emb_{mid}_meta.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"match_id": mid, "line_no": i, "line_kind": kinds[i], "primary_type": cls[i],
                                "primary_class": cls[i], "primary_agent": agent[i], "primary_team": None}) + "\n")


ids = [f"{i:02d}" for i in range(14)]
for m in ids:
    fake_match(m)

metas = [json.loads(l) for l in open(emb / "emb_00_meta.jsonl")]
vocab = build_agent_vocab([metas], DEFAULT_CONFIG)
ds = LabelledMatches(emb, ids[:3], DEFAULT_CONFIG, vocab)
b = collate_labelled([ds[0], ds[1], ds[2]])
check("collate shapes", b["t_exc"].shape == b["mask"].shape == b["t_event"].shape == (3, 64))
backbone = HierarchicalModel(in_dim=48, d_model=32, window=8, heads=2, dropout=0.0)
model = MultiTaskModel(backbone, len(CLS), len(vocab), 0.0)
out = model(b["x"], b["mask"], b["kind"], offset=3)
check("head output shapes", out["excitement"].shape == (3, 64) and out["event_logits"].shape == (3, 64, len(CLS))
      and out["agent_logits"].shape == (3, 64, len(vocab)))
check("excitement within (0,1)", bool(((out["excitement"] > 0) & (out["excitement"] < 1)).all()))
L = multitask_loss(out, b)
L["total"].backward()
dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
check("all three losses finite", all(bool(torch.isfinite(L[k])) for k in "ABC"), str({k: round(float(L[k]), 3) for k in "ABC"}))
check("gradients reach every parameter of backbone and heads", not dead, str(dead[:3]))
empty = {**b, "t_live": torch.zeros_like(b["t_live"]), "t_event": torch.full_like(b["t_event"], -100),
         "t_agent": torch.full_like(b["t_agent"], -100)}
check("loss with no live lines is finite (no NaN)", bool(torch.isfinite(multitask_loss(model(b["x"], b["mask"], b["kind"]), empty)["total"])))

# ------------------------------------------------------ gold CSV round trip
lab_dir = tmp / "labels"; lab_dir.mkdir()
subprocess.run([sys.executable, str(HERE / "labels.py"), "--emb-dir", str(emb), "--match-id", "00",
                "--out-dir", str(lab_dir)], check=True, capture_output=True)
rows = list(csv.DictReader(open(lab_dir / "labels_00.csv", encoding="utf-8-sig")))
rows[20]["gold_excitement"] = "0.95"
with open(lab_dir / "labels_00.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
check("gold cell is read back", load_gold(lab_dir) == {"00": {20: 0.95}})
r = subprocess.run([sys.executable, str(HERE / "labels.py"), "--emb-dir", str(emb), "--match-id", "00",
                    "--out-dir", str(lab_dir)], capture_output=True, text=True)
check("re-running labels.py does not overwrite hand edits", "[skip]" in r.stdout and load_gold(lab_dir) == {"00": {20: 0.95}})

# ------------------------------------------------------ 3. end-to-end training must learn planted signal
r = subprocess.run([sys.executable, str(HERE / "train.py"), "--emb-dir", str(emb), "--labels-dir", str(lab_dir),
                    "--out-dir", str(tmp / "model"), "--d-model", "32", "--epochs", "25", "--batch-size", "4",
                    "--lr", "2e-3", "--patience", "25", "--device", "cpu"],
                   capture_output=True, text=True)
print(r.stdout[-1800:] if r.returncode == 0 else r.stdout[-800:] + r.stderr[-1500:])
check("train.py exits cleanly", r.returncode == 0)
if r.returncode == 0:
    t = json.loads((tmp / "model" / "metrics.json").read_text())["test"]
    check("Head B beats the majority class clearly", t["B_acc"] > t["B_majority_acc"] + 0.1,
          f"(acc {t['B_acc']:.2f} vs majority {t['B_majority_acc']:.2f}, macro-F1 {t['B_macro_f1']:.2f})")
    check("Head A tracks the weak score (Spearman > 0.5)", t["A_spearman"] > 0.5, f"(rho {t['A_spearman']:.2f})")
    check("Head A beats predict-the-mean", t["A_mae"] < t["A_mae_mean_baseline"],
          f"(MAE {t['A_mae']:.3f} vs {t['A_mae_mean_baseline']:.3f})")
    check("Head C beats the most common player", t["C_acc"] > t["C_majority_acc"] + 0.1,
          f"(acc {t['C_acc']:.2f} vs {t['C_majority_acc']:.2f})")
    check("checkpoint written", (tmp / "model" / "best.pt").exists())

print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
