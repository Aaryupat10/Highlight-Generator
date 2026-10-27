"""
Run the whole pipeline over every match, stage by stage.  Safe to stop and start again.

    python Pipeline/run_all.py --limit 3                 # trial: first 3 matches through everything
    python Pipeline/run_all.py                           # all matches, all stages
    python Pipeline/run_all.py --stages 5,6,7,9          # only these stages (e.g. after changing Stage 5)
    python Pipeline/run_all.py --stop-after-min 120      # coref for at most 2 hours, then carry on
    python Pipeline/run_all.py --dry-run                 # print the commands, run nothing

Stages (numbers follow the project document):
    1-2  data + name books    checked only (data/23_24_commentary_clean.csv, outputs/name_book/*)
    3    coreference          run_coref.py + resolve.py --stage C, one call per match, WHOLE match
    4    event extraction     extract_events.py, per match whose Stage 3 output is newer than its events
    5    line embeddings      embed_lines.py, one call for every match that is missing or stale
    6    labels (CSV)         labels.py: writes the per-match CSVs you can hand-check; never overwrites
    7    training             Multitask heads/train.py  (this trains Stages 6+7 together: the only learned part)
    9    highlights (9+10)    run_highlights.py on the test matches, plus a weak-label baseline

Only Stage 7 is trained. The coreference model is pretrained (fastcoref), Stage 4 is rules, Stage 5 uses a
frozen BERT. Stage 8 (attention analysis) is not part of this run.

Resume logic: Stage 3 writes outputs/coref/full_C_<id>.flag when a match finished over ALL its lines. Matches
without the flag are (re)done, which is why match 93323, first run with --max-lines 80, is redone once.
Stages 4 and 5 compare file times, so only changed matches are redone. Failures never stop the run: the
match is logged in outputs/pipeline_logs/ and listed in outputs/pipeline_report.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
S = {
    "coref": ROOT / "Coref resolution" / "run_coref.py",
    "resolve": ROOT / "Coref resolution" / "resolve.py",
    "name_book": ROOT / "Coref resolution" / "name_book.py",
    "events": ROOT / "Event extraction" / "extract_events.py",
    "embed": ROOT / "Line embedding" / "embed_lines.py",
    "labels": ROOT / "Multitask heads" / "labels.py",
    "train": ROOT / "Multitask heads" / "train.py",
    "highlights": ROOT / "Highlight output" / "run_highlights.py",
}
CSV = ROOT / "data" / "23_24_commentary_clean.csv"
NAME_BOOK = ROOT / "outputs" / "name_book"
COREF = ROOT / "outputs" / "coref"
EVENTS = ROOT / "outputs" / "events"
EMB = ROOT / "outputs" / "embeddings"
LOGS = ROOT / "outputs" / "pipeline_logs"
report: dict = {"stages": {}, "failed": []}


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def run(cmd: list, name: str, a, stream: bool = False) -> bool:
    cmd = [str(c) for c in cmd]
    shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if a.dry_run:
        print("   $", shown)
        return True
    if stream:                                           # long jobs: let the user watch
        print("   $", shown)
        return subprocess.run(cmd, cwd=ROOT).returncode == 0
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + shown + "\n\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        report["failed"].append({"step": name, "log": str(log.relative_to(ROOT))})
    return rc == 0


def commentary_match_ids() -> list[str]:
    import pandas as pd
    df = pd.read_csv(CSV, dtype={"clean_text": str}).fillna({"clean_text": ""})
    ids = sorted(df[df.clean_text.str.strip() != ""].match_id.unique())
    return [str(int(i)) for i in ids]


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def done_matches() -> list[str]:
    """Matches whose Stage 3 finished over the whole match."""
    return sorted(p.stem[len("full_C_"):] for p in COREF.glob("full_C_*.flag")
                  if (COREF / f"resolved_C_{p.stem[len('full_C_'):]}.jsonl").exists())


# --------------------------------------------------------------------------- stages
def stage_data(a):
    print("\n== Stages 1-2: data and name books ==")
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found. Stage 1-2 outputs are needed first "
                         f"(Preprocessing/prepare_event_text.py).")
    if not (NAME_BOOK / "name_books.json").exists() or not (NAME_BOOK / "line_types.csv").exists():
        print("  name books missing -> building them")
        run([PY, S["name_book"]], "name_book", a)
    else:
        print("  ok:", CSV.relative_to(ROOT), "and outputs/name_book/*")


def stage_coref(a, ids):
    print(f"\n== Stage 3: coreference, {len(ids)} matches, whole match each ==")
    todo = [m for m in ids if a.redo or not ((COREF / f"full_C_{m}.flag").exists()
                                             and (COREF / f"resolved_C_{m}.jsonl").exists())]
    print(f"  {len(ids) - len(todo)} already done, {len(todo)} to do")
    if todo and importlib.util.find_spec("fastcoref") is None and not a.dry_run:
        print("  WARNING: the 'fastcoref' package is not installed in this Python; the first match will fail.")
    dev = {"cuda": "cuda:0"}.get(a.device, a.device)
    t0, ok, bad = time.time(), 0, 0
    for k, m in enumerate(todo, 1):
        if a.stop_after_min and (time.time() - t0) / 60 >= a.stop_after_min:
            print(f"  stopping after {a.stop_after_min} min as asked; {len(todo) - k + 1} matches left "
                  f"(run again to continue)")
            break
        t1 = time.time()
        cmd = [PY, S["coref"], "--match-id", m, "--max-lines", 0] + (["--device", dev] if dev else [])
        good = run(cmd, f"stage3_coref_{m}", a) and run(
            [PY, S["resolve"], "--match-id", m, "--stage", "C", "--max-lines", 0], f"stage3_resolve_{m}", a)
        if good and not a.dry_run:
            COREF.mkdir(parents=True, exist_ok=True)
            (COREF / f"full_C_{m}.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        ok, bad = ok + good, bad + (not good)
        if not a.dry_run:
            per = (time.time() - t0) / k
            print(f"  [{k}/{len(todo)}] match {m} {'ok' if good else 'FAILED'} in {fmt(time.time() - t1)}; "
                  f"about {fmt(per * (len(todo) - k))} left")
    report["stages"]["3"] = {"done_before": len(ids) - len(todo), "ok": ok, "failed": bad}


def stage_events(a):
    print("\n== Stage 4: event extraction ==")
    todo = [m for m in done_matches()
            if mtime(EVENTS / f"events_{m}.jsonl") < mtime(COREF / f"resolved_C_{m}.jsonl")]
    print(f"  {len(todo)} matches new or changed")
    ok = sum(run([PY, S["events"], "--match-id", m], f"stage4_events_{m}", a) for m in todo)
    report["stages"]["4"] = {"ok": ok, "failed": len(todo) - ok}


def stage_embed(a):
    print("\n== Stage 5: line embeddings ==")
    todo = [m for m in done_matches() if (EVENTS / f"events_{m}.jsonl").exists()
            and mtime(EMB / f"emb_{m}.npy") < mtime(EVENTS / f"events_{m}.jsonl")]
    print(f"  {len(todo)} matches missing or older than their events")
    if todo:
        good = run([PY, S["embed"], "--ids", ",".join(todo), "--overwrite"] +
                   (["--device", a.device] if a.device else []), "stage5_embed", a, stream=True)
        report["stages"]["5"] = {"matches": len(todo), "ok": bool(good)}


def stage_labels(a):
    print("\n== Stage 6 (labels): per-match CSVs for hand-checking ==")
    run([PY, S["labels"], "--emb-dir", EMB, "--all", "--events-dir", EVENTS], "stage6_labels", a, stream=True)


def stage_train(a):
    print("\n== Stage 7: training (Stage 6 backbone + heads, jointly) ==")
    n = len(list(EMB.glob("emb_*_meta.jsonl"))) if EMB.exists() else 0
    print(f"  {n} embedded matches")
    if n < 6 and not a.dry_run:
        print("  need at least 6 embedded matches to split into train/val/test sensibly; skipping. "
              "Use more matches (drop --limit).")
        return
    cmd = [PY, S["train"]] + a.train_args.split() + (["--device", a.device] if a.device else [])
    report["stages"]["7"] = {"ok": bool(run(cmd, "stage7_train", a, stream=True))}


def stage_highlights(a):
    print("\n== Stages 9-10: highlights ==")
    ckpt = ROOT / "outputs" / "model" / "best.pt"
    if not ckpt.exists() and not a.dry_run:
        print("  no outputs/model/best.pt yet; run stage 7 first. Doing only the weak-label baseline.")
    else:
        run([PY, S["highlights"], "--split", "test"] + (["--device", a.device] if a.device else []),
            "stage9_highlights_model", a, stream=True)
    run([PY, S["highlights"], "--source", "weak", "--split", "all", "--out-dir",
         ROOT / "outputs" / "highlights_weak"], "stage9_highlights_weak", a, stream=True)


def main():
    ap = argparse.ArgumentParser(description="Run the pipeline over every match (resumable)")
    ap.add_argument("--stages", default="3,4,5,6,7,9", help="comma list from 3,4,5,6,7,9")
    ap.add_argument("--limit", type=int, default=0, help="only the first N matches (trial run)")
    ap.add_argument("--ids", default=None, help="only these match ids, comma separated")
    ap.add_argument("--redo", action="store_true", help="redo Stage 3 even for finished matches")
    ap.add_argument("--stop-after-min", type=float, default=0, help="time budget for Stage 3 (minutes)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: each script picks)")
    ap.add_argument("--train-args", default="", help='extra flags for train.py, e.g. "--epochs 60 --lr 5e-4"')
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stages = {s.strip() for s in a.stages.split(",")}

    print(f"repo: {ROOT}\npython: {PY}")
    if importlib.util.find_spec("torch") is not None:
        import torch
        print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    for name, p in S.items():
        if not p.exists():
            raise SystemExit(f"missing script: {p}")

    stage_data(a)
    ids = commentary_match_ids()
    if a.ids:
        want = {s.strip() for s in a.ids.split(",")}
        ids = [m for m in ids if m in want]
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} matches with commentary in scope")

    t0 = time.time()
    if "3" in stages:
        stage_coref(a, ids)
    if "4" in stages:
        stage_events(a)
    if "5" in stages:
        stage_embed(a)
    if "6" in stages:
        stage_labels(a)
    if "7" in stages:
        stage_train(a)
    if "9" in stages:
        stage_highlights(a)

    report["seconds"] = round(time.time() - t0)
    if not a.dry_run:
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n== finished in {fmt(time.time() - t0)} ==")
    if report["failed"]:
        print(f"{len(report['failed'])} steps failed (first few):")
        for f in report["failed"][:10]:
            print("   ", f["step"], "->", f["log"])
    else:
        print("no failures")


if __name__ == "__main__":
    main()
