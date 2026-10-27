"""Checks for hier_model.py that need no data and no GPU:  python "Hierarchical model/smoke_test.py"

Run it after any change to the model. It tests the properties the design claims:
shapes, padding invariance, causality of the upper tier, locality of the lower tier,
gradient flow, and the chunk-offset layout.
"""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hier_model import HierarchicalModel, collate_matches, MatchEmbeddings  # noqa: E402

torch.manual_seed(0)
D_IN, W = 32, 8
fails = []


def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {extra}")
    if not ok:
        fails.append(name)


def make(lower, upper, causal=True):
    return HierarchicalModel(in_dim=D_IN, d_model=16, window=W, lower=lower, upper=upper,
                             heads=2, dropout=0.0, upper_causal=causal).eval()


def batch(lengths):
    T = max(lengths)
    x = torch.randn(len(lengths), T, D_IN)
    mask = torch.stack([torch.arange(T) < n for n in lengths])
    kind = torch.randint(0, 4, (len(lengths), T))
    return x, mask, kind


combos = list(itertools.product(["gru", "transformer"], ["gru", "transformer"]))

# 1. shapes, finite, padding zeroed, every offset
for lo, up in combos:
    m = make(lo, up)
    x, mask, kind = batch([37, 20, 5])
    ok = True
    for off in range(W):
        out = m(x, mask, kind, offset=off)
        ok &= out["h"].shape == (3, 37, 16) and bool(torch.isfinite(out["h"]).all())
        ok &= bool((out["h"][~mask] == 0).all())
        # pooling weights sum to 1 inside every non-empty chunk
        for b_i, n in enumerate((37, 20, 5)):
            cid = out["chunk_id"][:n]
            for c in cid.unique():
                ok &= abs(float(out["pool_attn"][b_i, :n][cid == c].sum()) - 1.0) < 1e-4
    check(f"shapes/finite/padding  lower={lo} upper={up}", ok)

# 2. padding invariance: a match alone == the same match inside a padded batch
for lo, up in combos:
    m = make(lo, up)
    x, mask, kind = batch([30, 61])
    with torch.no_grad():
        full = m(x, mask, kind)["h"][0, :30]
        alone = m(x[:1, :30], mask[:1, :30], kind[:1, :30])["h"][0]
    check(f"padding invariance     lower={lo} upper={up}", torch.allclose(full, alone, atol=1e-5),
          f"(max diff {(full - alone).abs().max():.2e})")

# 3. causal upper tier: changing later chunks must not change earlier lines
for lo, up in combos:
    m = make(lo, up, causal=True)
    x, mask, kind = batch([64])
    x2 = x.clone()
    x2[:, 40:] = torch.randn_like(x2[:, 40:])      # lines 40.. live in chunks 5.. (offset 0)
    with torch.no_grad():
        a, b = m(x, mask, kind)["h"], m(x2, mask, kind)["h"]
    check(f"upper tier is causal   lower={lo} upper={up}", torch.allclose(a[:, :40], b[:, :40], atol=1e-5),
          f"(max diff {(a[:, :40] - b[:, :40]).abs().max():.2e})")

# 4. non-causal upper tier really does see the future (so the flag does something)
m = make("gru", "gru", causal=False)
x, mask, kind = batch([64])
x2 = x.clone(); x2[:, 40:] = torch.randn_like(x2[:, 40:])
with torch.no_grad():
    a, b = m(x, mask, kind)["h"], m(x2, mask, kind)["h"]
check("non-causal upper sees later chunks", not torch.allclose(a[:, :40], b[:, :40], atol=1e-5))

# 5. lower tier is local: changing one chunk changes only that chunk's lines when upper is causal
#    and only lines in LATER chunks otherwise
m = make("transformer", "gru", causal=True)
x, mask, kind = batch([64])
x2 = x.clone(); x2[:, 16:24] = torch.randn_like(x2[:, 16:24])      # chunk 2 exactly
with torch.no_grad():
    a, b = m(x, mask, kind)["h"], m(x2, mask, kind)["h"]
check("earlier chunks unaffected by chunk 2", torch.allclose(a[:, :16], b[:, :16], atol=1e-5))
check("chunk 2 itself changed", not torch.allclose(a[:, 16:24], b[:, 16:24], atol=1e-5))
check("later chunks see chunk 2 via upper tier", not torch.allclose(a[:, 24:], b[:, 24:], atol=1e-6))

# 6. offset moves the boundaries
m = make("gru", "gru")
cid0, _, n0 = m._layout(30, 0, "cpu")
cid3, _, n3 = m._layout(30, 3, "cpu")
check("offset=3 -> first chunk has 5 lines", int((cid3 == 0).sum()) == 5 and int((cid0 == 0).sum()) == 8)
check("every chunk holds <= W lines", max(int((cid3 == c).sum()) for c in range(n3)) <= W)
x, mask, kind = batch([30])
with torch.no_grad():
    check("different offset -> different output",
          not torch.allclose(m(x, mask, kind, offset=0)["h"], m(x, mask, kind, offset=3)["h"]))

# 7. gradients reach every parameter
for lo, up in combos:
    m = make(lo, up).train()
    x, mask, kind = batch([40, 25])
    m(x, mask, kind, offset=2)["h"].pow(2).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    check(f"gradients everywhere   lower={lo} upper={up}", not missing, str(missing[:3]))

# 8. dataset round trip on real Stage 5 files, if a folder is given
if len(sys.argv) > 1:
    ds = MatchEmbeddings(sys.argv[1])
    b = collate_matches([ds[i] for i in range(min(3, len(ds)))])
    m = HierarchicalModel(in_dim=b["x"].shape[-1], d_model=16, window=W, heads=2).eval()
    with torch.no_grad():
        out = m(b["x"], b["mask"], b["kind"])
    check(f"real files in {sys.argv[1]}", bool(torch.isfinite(out["h"]).all()),
          f"({len(ds)} matches, batch {tuple(b['x'].shape)})")

print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
