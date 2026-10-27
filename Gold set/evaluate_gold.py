"""
Gold-set evaluation. Scores saved outputs against hand-checked lines. Trains nothing, writes nothing
that the pipeline reads, so it is safe to run while run_all.py is going.

    python "Gold set/evaluate_gold.py"                     # everything it can find
    python "Gold set/evaluate_gold.py" --no-model          # skip the trained model (no torch needed)
    python "Gold set/evaluate_gold.py" --device cuda

Gold files (this folder):
  gold_lines.csv   per live line: gold_class, gold_excitement (0..1), gold_player.  Lines with
                   explicit == "default" were NOT looked at individually: they got "routine / 0.1".
                   Only "explicit == yes" lines are scored for class and player; all gold lines are
                   used for the excitement curve (the default lines are the quiet stretches).
  gold_coref.csv   sampled pronoun / role mentions with the right entity (gold_entity).

What is measured, and what it says (read this before quoting numbers):
  1  Stage 4 vs gold          class accuracy / macro-F1, player accuracy. Stage 4 is rule based, so this
                              is the honest score of the extraction itself.
  2  weak label vs gold       how well the Stage 7 TRAINING targets (built from Stage 4) agree with the gold
                              excitement. If this is low, a model that copies the weak labels is capped.
  3  model vs gold            Head A / B / C against gold, reported per split. Only the 'test' split is
                              held out; 'train'/'val' numbers are optimistic. With few gold matches in test,
                              treat the numbers as indicative.
  4  highlight precision/recall  peaks from find_peaks() vs gold moments (gold excitement >= --hi), for the
                              weak-label curve and the model.
  5  coreference               accuracy of pronoun / role mentions (the part that is not a name lookup).

The gold set was drafted by Claude (annotator column) and is only as good as its review. Correct the CSV
where you disagree and set reviewed=yes; rerun to see the numbers move.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("Highlight output", "Hierarchical model", "Multitask heads"):
    sys.path.insert(0, str(ROOT / sub))
from labels import agent_name, excitement_scores, load_config  # noqa: E402
from peak_detection import find_peaks  # noqa: E402


# --------------------------------------------------------------------------- small helpers
def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace("'", "").split())


def same_name(a, b) -> bool:
    """'Jarrod Bowen' == 'Bowen'; 'Manchester City' == 'Man City' is not handled on purpose."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    return ta[-1] == tb[-1] and (len(ta) == 1 or len(tb) == 1) or a in b or b in a


def pearson(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a):                       # average ties
        m = a == v
        r[m] = r[m].mean()
    return r


def spearman(x, y) -> float | None:
    return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def class_report(gold: list[str], pred: list[str], classes: list[str]) -> dict:
    out, f1s = {}, []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        p, r, f = prf(tp, fp, fn)
        n = sum(g == c for g in gold)
        out[c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": n}
        if n:
            f1s.append(f)
    acc = float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else None
    return {"n": len(gold), "accuracy": None if acc is None else round(acc, 3),
            "macro_f1": round(float(np.mean(f1s)), 3) if f1s else None, "per_class": out}


def gold_moments(pos: np.ndarray, exc: np.ndarray, hi: float, gap: int = 3) -> list[tuple[int, int]]:
    """Positions with gold excitement >= hi, merged when at most `gap` positions apart."""
    idx = [int(p) for p, e in zip(pos, exc) if e >= hi]
    spans: list[list[int]] = []
    for p in idx:
        if spans and p - spans[-1][1] <= gap:
            spans[-1][1] = p
        else:
            spans.append([p, p])
    return [(a, b) for a, b in spans]


def highlight_pr(moments, peaks, tol) -> dict:
    hit_m = [any(a - tol <= p <= b + tol for p in peaks) for a, b in moments]
    hit_p = [any(a - tol <= p <= b + tol for a, b in moments) for p in peaks]
    return {"moments": len(moments), "peaks": len(peaks), "moments_found": sum(hit_m),
            "peaks_correct": sum(hit_p),
            "recall": round(sum(hit_m) / len(moments), 3) if moments else None,
            "precision": round(sum(hit_p) / len(peaks), 3) if peaks else None}


# --------------------------------------------------------------------------- loading
def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_meta(mid: str, emb_dir: Path, events_dir: Path) -> tuple[list[dict], str]:
    p = emb_dir / f"emb_{mid}_meta.jsonl"
    if p.exists():
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()], "embeddings"
    p = events_dir / f"events_{mid}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return sorted(rows, key=lambda r: r["line_no"]), "events"
    raise FileNotFoundError(f"neither {emb_dir}/emb_{mid}_meta.jsonl nor {p} exists")


def mean_or_none(v):
    return round(float(np.mean(v)), 3) if len(v) else None


# --------------------------------------------------------------------------- the report
def main():
    ap = argparse.ArgumentParser(description="Score saved pipeline outputs against the gold set")
    ap.add_argument("--gold-dir", default=str(HERE))
    ap.add_argument("--emb-dir", default=str(ROOT / "outputs" / "embeddings"))
    ap.add_argument("--events-dir", default=str(ROOT / "outputs" / "events"))
    ap.add_argument("--coref-dir", default=str(ROOT / "outputs" / "coref"))
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "model" / "best.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "gold_eval"))
    ap.add_argument("--label-config", default=None, help="JSON overriding DEFAULT_CONFIG (other sports)")
    ap.add_argument("--hi", type=float, default=0.7, help="gold excitement that counts as a highlight")
    ap.add_argument("--tol", type=int, default=2, help="a peak this many lines outside a gold moment still counts")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-distance", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()

    gold_dir, emb_dir, events_dir = Path(a.gold_dir), Path(a.emb_dir), Path(a.events_dir)
    cfg = load_config(a.label_config)
    classes = cfg["classes"]
    report: dict = {"notes": [], "gold_dir": str(gold_dir)}

    gl = read_csv(gold_dir / "gold_lines.csv")
    by_match: dict[str, list[dict]] = defaultdict(list)
    for r in gl:
        by_match[str(r["match_id"])].append(r)
    annotators = Counter(r.get("annotator", "") for r in gl)
    reviewed = sum(str(r.get("reviewed", "")).strip().lower() in ("yes", "y", "1", "true") for r in gl)
    report["gold"] = {"matches": sorted(by_match), "lines": len(gl),
                      "explicit_lines": sum(r["explicit"] == "yes" for r in gl),
                      "annotators": dict(annotators), "reviewed_lines": reviewed}
    if reviewed == 0:
        report["notes"].append("No gold line is marked reviewed=yes: this is still the unreviewed Claude draft.")

    # ------- per match: align gold to positions in meta
    per: dict[str, dict] = {}
    for mid, rows in by_match.items():
        try:
            meta, src = load_meta(mid, emb_dir, events_dir)
        except FileNotFoundError as e:
            report["notes"].append(f"match {mid} skipped: {e}")
            continue
        pos_of = {m.get("line_no"): i for i, m in enumerate(meta)}
        valid = np.array([m.get("line_kind") in cfg["live_kinds"] for m in meta], dtype=bool)
        g_exc = np.full(len(meta), np.nan)
        items = []
        for r in rows:
            i = pos_of.get(int(r["line_no"]))
            if i is None or not valid[i]:
                continue
            g_exc[i] = float(r["gold_excitement"])
            items.append((i, r))
        weak, _ = excitement_scores(meta, cfg)
        per[mid] = {"meta": meta, "valid": valid, "g_exc": g_exc, "items": items, "weak": weak, "src": src}
    if not per:
        raise SystemExit("no gold match could be aligned to saved outputs; run the pipeline first")

    # ------- 1  Stage 4 vs gold
    gold_c, s4_c, name_hits, named, gold_named = [], [], 0, 0, 0
    miss_examples = []
    for mid, d in per.items():
        for i, r in d["items"]:
            if r["explicit"] != "yes":
                continue
            m = d["meta"][i]
            gold_c.append(r["gold_class"])
            s4_c.append(m.get("primary_class") or "routine")
            if r["gold_class"] != s4_c[-1] and len(miss_examples) < 12:
                miss_examples.append({"match": mid, "line_no": int(r["line_no"]), "gold": r["gold_class"],
                                      "stage4": s4_c[-1], "text": r["text"][:110]})
            gp = r["gold_player"].strip()
            if gp:
                gold_named += 1
                sp = agent_name(m.get("primary_agent"))
                if sp:
                    named += 1
                    name_hits += same_name(sp, gp)
    report["stage4"] = {
        "class": class_report(gold_c, s4_c, classes),
        "player": {"gold_lines_with_player": gold_named, "stage4_named_someone": named,
                   "correct": name_hits,
                   "accuracy_all": round(name_hits / gold_named, 3) if gold_named else None,
                   "accuracy_when_named": round(name_hits / named, 3) if named else None},
        "class_mistakes_sample": miss_examples,
        "note": "Scored on the lines marked explicit=yes only (annotated one by one)."}

    # ------- 2  weak label vs gold excitement
    gx, wx = [], []
    for d in per.values():
        for i, _ in d["items"]:
            gx.append(d["g_exc"][i]); wx.append(d["weak"][i])
    gx, wx = np.array(gx), np.array(wx)
    report["weak_vs_gold"] = {"n_lines": len(gx), "pearson": pearson(gx, wx), "spearman": spearman(gx, wx),
                              "mae": round(float(np.abs(gx - wx).mean()), 3),
                              "note": "How far the training targets are from the gold curve: the ceiling for "
                                      "a model that learns them faithfully."}

    # ------- 4a  highlight P/R for the weak curve
    def hl_eval(curve_of) -> dict:
        res = {}
        for tag, kw in {"default": dict(threshold=a.threshold, min_distance=a.min_distance, top_k=a.top_k),
                        "top3": dict(threshold=0.0, min_distance=a.min_distance, top_k=3),
                        "top5": dict(threshold=0.0, min_distance=a.min_distance, top_k=5),
                        "top8": dict(threshold=0.0, min_distance=a.min_distance, top_k=8)}.items():
            tot = Counter()
            for mid, d in per.items():
                sc = curve_of(mid, d)
                if sc is None:
                    continue
                sc = np.where(d["valid"], sc, 0.0)
                pos = np.array([i for i, _ in d["items"]])
                moments = gold_moments(pos, d["g_exc"][pos], a.hi)
                peaks = find_peaks(sc, d["valid"], kw["threshold"], kw["min_distance"], kw["top_k"], 1)
                r = highlight_pr(moments, peaks, a.tol)
                for k in ("moments", "peaks", "moments_found", "peaks_correct"):
                    tot[k] += r[k]
            if tot["moments"]:
                res[tag] = {**dict(tot), "recall": round(tot["moments_found"] / tot["moments"], 3),
                            "precision": round(tot["peaks_correct"] / tot["peaks"], 3) if tot["peaks"] else None}
        return res

    report["highlights_weak"] = hl_eval(lambda mid, d: d["weak"])
    report["highlights_params"] = {"gold_moment_if_excitement>=": a.hi, "tolerance_lines": a.tol,
                                   "default_peak_params": {"threshold": a.threshold,
                                                           "min_distance": a.min_distance, "top_k": a.top_k}}

    # ------- 3 + 4b  trained model
    ckpt = Path(a.ckpt)
    if a.no_model:
        report["notes"].append("model skipped (--no-model)")
    elif not ckpt.exists():
        report["notes"].append(f"model skipped: {ckpt} not found (train first)")
    else:
        try:
            import torch
            from run_highlights import load_model, predict
            device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, ck = load_model(str(ckpt), device)
            window, mclasses, vocab = ck["args"]["window"], ck["label_config"]["classes"], ck["agent_vocab"]
            split_of = {m: s for s, ids in ck["split"].items() for m in ids}
            preds = {}
            for mid, d in per.items():
                f = emb_dir / f"emb_{mid}.npy"
                if not f.exists():
                    report["notes"].append(f"model: no embeddings for {mid}; skipped")
                    continue
                X = np.load(f)
                preds[mid] = predict(model, X, d["meta"], window, device, not a.no_tta)
            groups: dict[str, list[str]] = defaultdict(list)
            for mid in preds:
                groups[split_of.get(mid, "not_in_split")].append(mid)
            mod = {}
            for sp, mids in sorted(groups.items()):
                gx_, px_, gc, pc, gp_n, gp_ok, gp_named = [], [], [], [], 0, 0, 0
                for mid in mids:
                    d, (pe_x, pe_p, pa_p) = per[mid], preds[mid]
                    for i, r in d["items"]:
                        gx_.append(d["g_exc"][i]); px_.append(float(pe_x[i]))
                        if r["explicit"] == "yes":
                            gc.append(r["gold_class"]); pc.append(mclasses[int(pe_p[i].argmax())])
                            if r["gold_player"].strip():
                                gp_n += 1
                                who = vocab[int(pa_p[i].argmax())]
                                if who != "<other>":
                                    gp_named += 1
                                    gp_ok += same_name(who, r["gold_player"])
                gx_, px_ = np.array(gx_), np.array(px_)
                mod[sp] = {
                    "matches": mids,
                    "head_A_excitement": {"n_lines": len(gx_), "pearson": pearson(gx_, px_),
                                          "spearman": spearman(gx_, px_),
                                          "mae": round(float(np.abs(gx_ - px_).mean()), 3)},
                    "head_B_class": class_report(gc, pc, mclasses),
                    "head_C_player": {"gold_lines_with_player": gp_n, "model_named_someone": gp_named,
                                      "correct": gp_ok,
                                      "accuracy_all": round(gp_ok / gp_n, 3) if gp_n else None},
                    "highlights": hl_eval(lambda mid, d, _m=set(mids): preds[mid][0] if mid in _m else None),
                    "held_out": sp == "test"}
            report["model"] = mod
            if "test" not in mod:
                report["notes"].append("none of the gold matches is in the model's TEST split, so every model "
                                       "number here comes from matches it trained or validated on (optimistic). "
                                       "Add gold for test matches: see the list in outputs/model/best.pt split.")
            report["model_test_ids"] = ck["split"].get("test", [])
        except Exception as e:  # noqa: BLE001
            report["notes"].append(f"model evaluation failed: {type(e).__name__}: {e}")

    # ------- 5  coreference
    cpath = gold_dir / "gold_coref.csv"
    if cpath.exists():
        cg = read_csv(cpath)
        fresh: dict[tuple, dict] = {}
        for mid in {r["match_id"] for r in cg}:
            p = Path(a.coref_dir) / f"resolved_C_{mid}.jsonl"
            if p.exists():
                for l in open(p, encoding="utf-8"):
                    if l.strip():
                        rec = json.loads(l)
                        for m in rec["mentions"]:
                            fresh[(mid, rec["line_no"], m["start"])] = m
        tot, ok = Counter(), Counter()
        mistakes = []
        used_fresh = 0
        for r in cg:
            key = (r["match_id"], int(r["line_no"]), int(r["start"]))
            m = fresh.get(key)
            sys_ent = m["entity"] if m else (r["system_entity"] or None)
            used_fresh += m is not None
            gold = r["gold_entity"].strip()
            good = same_name(sys_ent, gold) if gold else sys_ent is None
            for tag in ("all", f"kind={r['kind']}", f"source={r['source'].split(':')[0]}"):
                tot[tag] += 1; ok[tag] += good
            if not good and len(mistakes) < 15:
                mistakes.append({"match": r["match_id"], "line_no": int(r["line_no"]), "mention": r["mention"],
                                 "system": sys_ent, "gold": gold, "context": r["context"][:100]})
        report["coref"] = {
            "n": tot["all"], "accuracy": round(ok["all"] / tot["all"], 3) if tot["all"] else None,
            "by_group": {k: {"n": tot[k], "accuracy": round(ok[k] / tot[k], 3)} for k in sorted(tot) if k != "all"},
            "unresolved_by_system": sum(1 for r in cg if not (fresh.get((r["match_id"], int(r["line_no"]), int(r["start"])), {}).get("entity")
                                                              or r["system_entity"])),
            "read_from_current_coref_files": used_fresh,
            "mistakes_sample": mistakes,
            "note": "Only pronoun / role mentions were sampled; name and alias lookups are not in this number. "
                    "Gold defaults to the system answer where it looked right, so unreviewed rows flatter the system."}
    else:
        report["notes"].append("gold_coref.csv not found; coreference skipped")

    # ------- write + print
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    show(report)
    print(f"\nfull report: {out / 'report.json'}")


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def show(r: dict):
    g = r["gold"]
    print(f"\n=== gold set: {g['lines']} live lines in {len(g['matches'])} matches "
          f"({g['explicit_lines']} annotated one by one), reviewed lines: {g['reviewed_lines']} ===")
    for n in r["notes"]:
        print("  note:", n)
    s = r["stage4"]
    print(f"\n[1] Stage 4 vs gold (n={s['class']['n']})  class accuracy {f3(s['class']['accuracy'])}  "
          f"macro-F1 {f3(s['class']['macro_f1'])}")
    for c, v in s["class"]["per_class"].items():
        if v["support"]:
            print(f"      {c:10s} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  (n={v['support']})")
    p = s["player"]
    print(f"    player: {p['correct']}/{p['gold_lines_with_player']} right "
          f"(accuracy {f3(p['accuracy_all'])}; when Stage 4 named someone {f3(p['accuracy_when_named'])})")
    w = r["weak_vs_gold"]
    print(f"\n[2] weak training label vs gold excitement (n={w['n_lines']})  Pearson {f3(w['pearson'])}  "
          f"Spearman {f3(w['spearman'])}  MAE {f3(w['mae'])}")

    def hl(d):
        for tag, v in d.items():
            print(f"      {tag:8s} recall {f3(v['recall'])} ({v['moments_found']}/{v['moments']})   "
                  f"precision {f3(v['precision'])} ({v['peaks_correct']}/{v['peaks']})")
    print("\n[4] highlights, weak-label curve")
    hl(r["highlights_weak"])
    for sp, v in (r.get("model") or {}).items():
        a_, b_, c_ = v["head_A_excitement"], v["head_B_class"], v["head_C_player"]
        tag = "HELD OUT" if v["held_out"] else "seen in training/validation, optimistic"
        print(f"\n[3] model, split '{sp}' ({tag}); matches {', '.join(v['matches'])}")
        print(f"      Head A  Pearson {f3(a_['pearson'])}  Spearman {f3(a_['spearman'])}  MAE {f3(a_['mae'])}  (n={a_['n_lines']})")
        print(f"      Head B  accuracy {f3(b_['accuracy'])}  macro-F1 {f3(b_['macro_f1'])}  (n={b_['n']})")
        print(f"      Head C  {c_['correct']}/{c_['gold_lines_with_player']} players right")
        print("      highlights:")
        hl(v["highlights"])
    c = r.get("coref")
    if c:
        print(f"\n[5] coreference, pronoun/role mentions: accuracy {f3(c['accuracy'])} (n={c['n']})")
        for k, v in c["by_group"].items():
            print(f"      {k:22s} {v['accuracy']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
