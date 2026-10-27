"""
Stage 7 - Multi-task output heads (deep DL #2)

Three heads share the Stage 6 backbone, so one set of per-line vectors h has to serve all of them:
    Head A  excitement   sigmoid regression 0..1          loss: MSE
    Head B  event type   classes from the label config    loss: cross-entropy (class-weighted)
    Head C  player       vocabulary of frequent players   loss: cross-entropy
Total loss = wA * A + wB * B + wC * C  (default 1, 1, 0.5).

All losses and metrics use LIVE lines only (see labels.py).

Limitation to state in the report: Head C is a classifier over a fixed player vocabulary, so it
cannot name a player it never saw in training; those fall into '<other>'. A pointer head that
scores the players named inside the match would generalise better (future work).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Hierarchical model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hier_model import HierarchicalModel, MatchEmbeddings, collate_matches  # noqa: E402
from labels import IGNORE, make_labels  # noqa: E402


# ---------------------------------------------------------------------------- data
class LabelledMatches(torch.utils.data.Dataset):
    def __init__(self, emb_dir, match_ids, cfg, agent_vocab, gold=None):
        self.base = MatchEmbeddings(emb_dir, match_ids)
        self.cfg, self.vocab, self.gold = cfg, agent_vocab, gold or {}

    @property
    def ids(self):
        return self.base.ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        lab = make_labels(item["meta"], self.cfg, self.vocab, self.gold.get(item["match_id"]))
        item["labels"] = lab
        return item


def collate_labelled(batch: list[dict]) -> dict:
    out = collate_matches(batch)
    B, T = out["mask"].shape
    exc = torch.zeros(B, T)
    live = torch.zeros(B, T, dtype=torch.bool)
    gold = torch.zeros(B, T, dtype=torch.bool)
    event = torch.full((B, T), IGNORE, dtype=torch.long)
    agent = torch.full((B, T), IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        n, L = len(b["x"]), b["labels"]
        exc[i, :n] = torch.from_numpy(L["excitement"])
        live[i, :n] = torch.from_numpy(L["live"])
        gold[i, :n] = torch.from_numpy(L["gold"])
        event[i, :n] = torch.from_numpy(L["event"])
        agent[i, :n] = torch.from_numpy(L["agent"])
    out.update({"t_exc": exc, "t_live": live, "t_gold": gold, "t_event": event, "t_agent": agent})
    return out


# ---------------------------------------------------------------------------- model
class MultiTaskModel(nn.Module):
    def __init__(self, backbone: HierarchicalModel, n_events: int, n_agents: int, dropout: float = 0.1):
        super().__init__()
        d = backbone.d_model
        self.backbone = backbone
        self.head_exc = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d // 2, 1))
        self.head_event = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_events))
        self.head_agent = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_agents))

    def forward(self, x, mask, kind=None, offset: int = 0) -> dict:
        out = self.backbone(x, mask, kind, offset=offset)
        h = out["h"]
        out["excitement"] = torch.sigmoid(self.head_exc(h)).squeeze(-1)
        out["event_logits"] = self.head_event(h)
        out["agent_logits"] = self.head_agent(h)
        return out


def _ce(logits, target, weight=None):
    if not bool((target != IGNORE).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                           weight=weight, ignore_index=IGNORE)


def multitask_loss(out: dict, batch: dict, w=(1.0, 1.0, 0.5), event_weight=None) -> dict:
    live = batch["t_live"]
    if bool(live.any()):
        la = F.mse_loss(out["excitement"][live], batch["t_exc"][live])
    else:
        la = out["excitement"].sum() * 0.0
    lb = _ce(out["event_logits"], batch["t_event"], event_weight)
    lc = _ce(out["agent_logits"], batch["t_agent"])
    return {"total": w[0] * la + w[1] * lb + w[2] * lc, "A": la, "B": lb, "C": lc}


# --------------------------------------------------------------------------- metrics
def _ranks(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a))
    for v in np.unique(a):                       # average ranks over ties
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def corr(a: np.ndarray, b: np.ndarray, spearman=False) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    if spearman:
        a, b = _ranks(a), _ranks(b)
    return float(np.corrcoef(a, b)[0, 1])


def macro_f1(pred: np.ndarray, true: np.ndarray, n_classes: int) -> tuple[float, dict]:
    f1s, per = [], {}
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        if tp + fn == 0:
            continue                              # class absent from this split
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f)
        per[c] = {"recall": r, "precision": p, "f1": f, "support": tp + fn}
    return (float(np.mean(f1s)) if f1s else float("nan")), per


@torch.no_grad()
def evaluate(model, loader, device, w, event_weight, n_events) -> dict:
    model.eval()
    tot = {"total": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
    nb = 0
    ex_p, ex_t, g_p, g_t = [], [], [], []
    ev_p, ev_t, ag_p, ag_t = [], [], [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["x"], b["mask"], b["kind"], offset=0)
        L = multitask_loss(out, b, w, event_weight)
        for k in tot:
            tot[k] += float(L[k])
        nb += 1
        live = b["t_live"]
        ex_p.append(out["excitement"][live].cpu().numpy()); ex_t.append(b["t_exc"][live].cpu().numpy())
        gm = b["t_gold"][live]
        g_p.append(out["excitement"][live][gm].cpu().numpy()); g_t.append(b["t_exc"][live][gm].cpu().numpy())
        m = b["t_event"] != IGNORE
        ev_p.append(out["event_logits"].argmax(-1)[m].cpu().numpy()); ev_t.append(b["t_event"][m].cpu().numpy())
        m = b["t_agent"] != IGNORE
        ag_p.append(out["agent_logits"].argmax(-1)[m].cpu().numpy()); ag_t.append(b["t_agent"][m].cpu().numpy())
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    ex_p, ex_t, g_p, g_t = map(cat, (ex_p, ex_t, g_p, g_t))
    ev_p, ev_t, ag_p, ag_t = map(cat, (ev_p, ev_t, ag_p, ag_t))
    res = {k: v / max(nb, 1) for k, v in tot.items()}
    if len(ex_p):
        res["A_mae"] = float(np.abs(ex_p - ex_t).mean())
        res["A_mae_mean_baseline"] = float(np.abs(ex_t - ex_t.mean()).mean())
        res["A_pearson"] = corr(ex_p, ex_t)
        res["A_spearman"] = corr(ex_p, ex_t, spearman=True)
    if len(g_p):
        res["A_gold_n"] = int(len(g_p))
        res["A_gold_mae"] = float(np.abs(g_p - g_t).mean())
        res["A_gold_spearman"] = corr(g_p, g_t, spearman=True)
    if len(ev_p):
        res["B_acc"] = float((ev_p == ev_t).mean())
        res["B_majority_acc"] = float(np.bincount(ev_t, minlength=n_events).max() / len(ev_t))
        res["B_macro_f1"], res["B_per_class"] = macro_f1(ev_p, ev_t, n_events)
    if len(ag_p):
        res["C_acc"] = float((ag_p == ag_t).mean())
        res["C_majority_acc"] = float(np.bincount(ag_t).max() / len(ag_t))
        res["C_n"] = int(len(ag_t))
    return res
