"""
Stage 6 - Hierarchical sequence model (deep DL #1)

Input  : Stage 5 line embeddings   outputs/embeddings/emb_<id>.npy  (+ emb_<id>_meta.jsonl)
Output : one contextual vector per line, h[i], for the Stage 7 heads.

Architecture (one match = one sequence of T lines)

    line embeddings [T, 768]
        |  LayerNorm -> Linear -> + line_kind embedding          (input block)
        v
    split into chunks of W consecutive lines                      (W = --window, default 8)
        |
    LOWER tier  (inside each chunk, ~ one passage of play)        gru (default) | transformer
        |  -> line states   [T, d]       short-range build-up
        |  -> chunk summary [N, d]       attention pooling over the chunk's lines
        v
    UPPER tier  (across the chunk summaries of the whole match)   transformer (default) | gru
        |  -> chunk context [N, d]       tension rising over 90 minutes
        v
    fuse(line state, its chunk's context)  ->  h  [T, d]

Design choices worth defending in the report
  * Chunk offset. Fixed chunks cut attacks in half at the boundaries. forward(offset=k) makes the
    first chunk shorter by k lines, shifting every boundary. Train with a random offset in
    [0, W) and the model cannot rely on where a boundary falls; evaluate with offset 0.
  * Upper tier is causal by default (a chunk only sees earlier chunks): "build-up" is a statement
    about the past. Pass upper_causal=False to let it see the whole match.
  * The lower tier is bidirectional inside its chunk: the lines right after a shout ("what a save")
    help decide that the line before it was a big moment.
  * line_kind (prematch/live/context/recap) is an input feature, so the model can learn that
    only live lines are highlight candidates, instead of being told by hand.

Returned dict (all padded positions are zeroed):
    h            [B, T, d]   per-line representation -> Stage 7 heads
    line_states  [B, T, d]   lower tier only (for the "does the upper tier help?" ablation)
    chunk_ctx    [B, N, d]   upper tier output
    chunk_summary[B, N, d]   what the upper tier received
    pool_attn    [B, T]      attention weight of each line inside its chunk (Stage 8)
    chunk_id     [T]         which chunk each line fell into

Check on real data (any machine with torch):
    python "Hierarchical model/hier_model.py" --emb-dir outputs/embeddings --match-id 93323
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = {"prematch": 0, "live": 1, "context": 2, "recap": 3}
UNKNOWN_KIND = 4


# --------------------------------------------------------------------------- data
class MatchEmbeddings(torch.utils.data.Dataset):
    """One item = one whole match (its Stage 5 vectors + the Stage 4 info that came with them)."""

    def __init__(self, emb_dir: str | Path, match_ids: list[str] | None = None):
        self.dir = Path(emb_dir)
        if match_ids is None:
            match_ids = sorted(p.stem[len("emb_"):] for p in self.dir.glob("emb_*.npy"))
        self.ids = [m for m in match_ids
                    if (self.dir / f"emb_{m}.npy").exists() and (self.dir / f"emb_{m}_meta.jsonl").exists()]
        if not self.ids:
            raise FileNotFoundError(f"no emb_<id>.npy + emb_<id>_meta.jsonl pairs in {self.dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mid = self.ids[i]
        x = np.load(self.dir / f"emb_{mid}.npy").astype(np.float32)
        with open(self.dir / f"emb_{mid}_meta.jsonl", encoding="utf-8") as f:
            meta = [json.loads(l) for l in f if l.strip()]
        if len(meta) != len(x):
            raise ValueError(f"match {mid}: {len(x)} vectors but {len(meta)} meta rows")
        kind = np.array([KINDS.get(m.get("line_kind"), UNKNOWN_KIND) for m in meta], dtype=np.int64)
        return {
            "match_id": mid,
            "meta": meta,
            "x": torch.from_numpy(x),
            "kind": torch.from_numpy(kind),
            "line_no": [m.get("line_no") for m in meta],
            "primary_class": [m.get("primary_class") for m in meta],
            "primary_agent": [m.get("primary_agent") for m in meta],
        }


def collate_matches(batch: list[dict]) -> dict:
    """Pad matches to the longest one. mask[b, t] is True for real lines."""
    B = len(batch)
    T = max(len(b["x"]) for b in batch)
    D = batch[0]["x"].shape[1]
    x = torch.zeros(B, T, D)
    kind = torch.full((B, T), UNKNOWN_KIND, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["x"])
        x[i, :n] = b["x"]
        kind[i, :n] = b["kind"]
        mask[i, :n] = True
    return {
        "x": x, "kind": kind, "mask": mask,
        "match_id": [b["match_id"] for b in batch],
        "line_no": [b["line_no"] for b in batch],
        "primary_class": [b["primary_class"] for b in batch],
        "primary_agent": [b["primary_agent"] for b in batch],
    }


# -------------------------------------------------------------------------- pieces
def sinusoid(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


class SeqBlock(nn.Module):
    """Sequence model over [B, L, d] with a validity mask. kind = 'gru' | 'transformer'.

    Positions with mask False never influence valid positions, and come back as zeros.
    `causal` makes position i see only positions <= i.
    """

    def __init__(self, kind: str, d: int, layers: int, heads: int, dropout: float,
                 bidirectional: bool, causal: bool):
        super().__init__()
        if kind not in ("gru", "transformer"):
            raise ValueError("kind must be 'gru' or 'transformer'")
        if causal and bidirectional and kind == "gru":
            raise ValueError("a causal GRU cannot be bidirectional")
        self.kind, self.causal, self.bidirectional = kind, causal, bidirectional
        if kind == "gru":
            h = d // 2 if bidirectional else d
            self.rnn = nn.GRU(d, h, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if layers > 1 else 0.0)
        else:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        x = x * mask.unsqueeze(-1)
        if self.kind == "gru":
            if self.bidirectional:
                # valid positions form a prefix of every row, so packing is exact
                lengths = mask.sum(1).clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                           enforce_sorted=False)
                out, _ = self.rnn(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
            else:
                out, _ = self.rnn(x)   # unidirectional: trailing padding cannot affect earlier steps
        else:
            x = x + sinusoid(L, d, x.device).unsqueeze(0)
            kpm = ~mask
            kpm = kpm.clone()
            kpm[kpm.all(dim=1), 0] = False        # a fully padded row would give NaN; keep slot 0
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            out = self.enc(x, mask=attn_mask, src_key_padding_mask=kpm)
        return out * mask.unsqueeze(-1)


class AttnPool(nn.Module):
    """Masked attention pooling: [B, N, W, d] -> summary [B, N, d] and weights [B, N, W]."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        s = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=-1) * mask          # empty chunks end up all-zero
        return (w.unsqueeze(-1) * h).sum(2), w


# ---------------------------------------------------------------------------- model
class HierarchicalModel(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 256, window: int = 8,
                 lower: str = "gru", upper: str = "transformer",
                 lower_layers: int = 1, upper_layers: int = 1, heads: int = 4,
                 dropout: float = 0.1, upper_causal: bool = True, use_kind: bool = True):
        super().__init__()
        assert d_model % 2 == 0 and d_model % heads == 0
        self.window, self.d_model, self.use_kind = window, d_model, use_kind
        self.in_norm = nn.LayerNorm(in_dim)      # removes the shared direction of mean-pooled BERT
        self.in_proj = nn.Linear(in_dim, d_model)
        self.kind_emb = nn.Embedding(UNKNOWN_KIND + 1, d_model) if use_kind else None
        self.drop = nn.Dropout(dropout)
        self.lower = SeqBlock(lower, d_model, lower_layers, heads, dropout,
                              bidirectional=True, causal=False)
        self.pool = AttnPool(d_model)
        self.upper = SeqBlock(upper, d_model, upper_layers, heads, dropout,
                              bidirectional=not upper_causal, causal=upper_causal)
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(d_model)

    # chunk layout: first chunk has W - offset lines, every later chunk has W
    def _layout(self, T: int, offset: int, device):
        W = self.window
        if not 0 <= offset < W:
            raise ValueError(f"offset must be in [0, {W})")
        first = W - offset
        pos = torch.arange(T, device=device)
        later = pos >= first
        chunk_id = torch.where(later, 1 + (pos - first) // W, torch.zeros_like(pos))
        slot = torch.where(later, (pos - first) % W, pos)
        return chunk_id, slot, int(chunk_id.max()) + 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor, kind: torch.Tensor | None = None,
                offset: int = 0) -> dict:
        B, T, _ = x.shape
        W, d = self.window, self.d_model
        z = self.in_proj(self.in_norm(x))
        if self.use_kind and kind is not None:
            z = z + self.kind_emb(kind)
        z = self.drop(z) * mask.unsqueeze(-1)

        chunk_id, slot, N = self._layout(T, offset, x.device)
        zc = z.new_zeros(B, N, W, d)
        mc = mask.new_zeros(B, N, W)
        zc[:, chunk_id, slot] = z
        mc[:, chunk_id, slot] = mask

        # lower tier: every chunk is an independent short sequence
        lo = self.lower(zc.reshape(B * N, W, d), mc.reshape(B * N, W)).reshape(B, N, W, d)
        summary, w = self.pool(lo, mc)                       # [B,N,d], [B,N,W]
        chunk_mask = mc.any(-1)                              # [B,N]

        # upper tier: sequence of chunk summaries for the whole match
        ctx = self.upper(summary, chunk_mask)                # [B,N,d]

        line_states = lo[:, chunk_id, slot] * mask.unsqueeze(-1)     # [B,T,d]
        line_ctx = ctx[:, chunk_id]                                  # [B,T,d]
        h = self.out_norm(line_states + self.fuse(torch.cat([line_states, line_ctx], -1)))
        h = h * mask.unsqueeze(-1)
        return {
            "h": h, "line_states": line_states, "chunk_ctx": ctx, "chunk_summary": summary,
            "pool_attn": w[:, chunk_id, slot] * mask, "chunk_id": chunk_id,
        }


# ------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description="Stage 6: forward pass on real Stage 5 output")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--match-id", default=None, help="default: first match found")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--lower", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--upper", choices=["gru", "transformer"], default="transformer")
    ap.add_argument("--upper-bidirectional", action="store_true")
    a = ap.parse_args()

    ds = MatchEmbeddings(a.emb_dir, [a.match_id] if a.match_id else None)
    batch = collate_matches([ds[0]])
    model = HierarchicalModel(in_dim=batch["x"].shape[-1], d_model=a.d_model, window=a.window,
                              lower=a.lower, upper=a.upper, upper_causal=not a.upper_bidirectional)
    model.eval()
    with torch.no_grad():
        out = model(batch["x"], batch["mask"], batch["kind"])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"match {batch['match_id'][0]}: {batch['x'].shape[1]} lines -> h {tuple(out['h'].shape)}, "
          f"{out['chunk_ctx'].shape[1]} chunks of up to {a.window} lines, {n_par:,} parameters")
    print(f"finite: {bool(torch.isfinite(out['h']).all())}")


if __name__ == "__main__":
    main()
