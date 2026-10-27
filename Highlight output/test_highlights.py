"""Checks for Stages 9-10 on synthetic data:  python "Highlight output/test_highlights.py"

Part 1 tests the peak logic on hand-made curves, part 2 the record fields and the output format,
part 3 runs run_highlights.py end to end: first with weak labels (no model), then with a model
trained for a few seconds by Multitask heads/train.py.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Multitask heads"))
from peak_detection import build_span, find_peaks, local_maxima, smooth  # noqa: E402
from highlight_output import (build_highlights, describe_method, narrative_context,  # noqa: E402
                              render_line, slot_text)
from labels import DEFAULT_CONFIG  # noqa: E402

fails = []


def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {extra}")
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------ 1. peak logic
s = np.array([0, .1, .9, .2, .1, .3, .95, .3, .1, .1, .6, .1])
v = np.ones(len(s), bool)
check("local maxima found", local_maxima(s, v) == [2, 6, 10])
check("ranked best first", find_peaks(s, v, 0.5, 2, None) == [6, 2, 10])
check("threshold removes weak peaks", find_peaks(s, v, 0.7, 2, None) == [6, 2])
check("min_distance suppresses the lower of two close peaks", find_peaks(s, v, 0.5, 5, None) == [6, 10] or
      find_peaks(s, v, 0.5, 5, None) == [6])
check("min_distance=5 keeps 6 and drops 2 and 10", find_peaks(s, v, 0.5, 5, None) == [6])
check("top_k limits", find_peaks(s, v, 0.5, 2, 2) == [6, 2])
check("plateau reported once (centre)", local_maxima(np.array([0, .8, .8, .8, 0]), np.ones(5, bool)) == [2])
check("plateau 'first'", local_maxima(np.array([0, .8, .8, .8, 0]), np.ones(5, bool), "first") == [1])
check("peak at the first and last line", local_maxima(np.array([.9, .1, .1, .9]), np.ones(4, bool)) == [0, 3])
check("non-live lines are never peaks", find_peaks(np.array([0, .9, 0, .5]), np.array([1, 0, 1, 1], bool), 0.1, 1) == [3])
check("flat curve has no peaks", find_peaks(np.full(6, .5), np.ones(6, bool), 0.1, 1) == [])
check("smoothing keeps length and flattens a spike",
      len(smooth(s, 3)) == len(s) and smooth(s, 3).max() < s.max())
cur = np.array([0.0, .3, .4, .6, 1.0, .5, .3, .1, .0])
st, en = build_span(cur, np.ones(9, bool), 4, set(), floor=0.25, max_before=10, max_after=3)
check("span covers contiguous lines above floor", (st, en) == (1, 6), str((st, en)))
st, en = build_span(cur, np.ones(9, bool), 4, {2}, floor=0.25)
check("span never crosses another peak", st == 3)
st, en = build_span(cur, np.ones(9, bool), 4, set(), floor=0.25, max_before=2)
check("max_before caps the build-up", st == 2)

# ---------------------------------------------------------- 2. records and format
check("slot_text handles dict, str, None", slot_text({"label": "header"}) == "header"
      and slot_text("cross") == "cross" and slot_text(None) is None)
check("slot_text: empty list -> None (was printed as '[]')", slot_text([]) is None)
check("slot_text: list of dicts", slot_text([{"label": "low", "text": "low cross"}], ("text", "label")) == "low cross"
      and slot_text([{"label": "header", "text": "header"}] * 2) == "header")
lst = {10: {"text": "x", "events": [{"method": [{"label": "low", "text": "low cross"}], "source": "assist"}]}}
check("method given as a list of dicts", describe_method(lst, 10, 8, 12) == "low cross from assist")
check("empty method list and no source -> None",
      describe_method({10: {"events": [{"method": [], "source": None}]}}, 10, 8, 12) is None)
rows = {10: {"text": "He heads it in", "events": [{"method": {"label": "header"}, "source": "cross", "primary": True}]}}
check("method string", describe_method(rows, 10, 7, 11) == "header from cross")
check("method from neighbouring line", describe_method(rows, 11, 7, 12) == "header from cross")
check("no method -> None", describe_method({}, 10, 7, 11) is None)

meta = [{"line_no": i, "line_kind": "live", "primary_class": "routine", "primary_agent": None} for i in range(30)]
for i in (14, 15):
    meta[i]["primary_class"] = "near_miss"
meta[20].update(primary_class="goal", primary_agent={"entity": "Erling Haaland"})
valid = np.ones(30, bool)
check("narrative: two near misses -> sustained pressure",
      narrative_context(meta, valid, 18, 13) == "sustained pressure", narrative_context(meta, valid, 18, 13))
check("narrative: late phase is named", "late" in narrative_context(meta, valid, 28, 25))
check("narrative: one-line build-up -> sudden", narrative_context(meta, valid, 20, 20).startswith("sudden"))

sc = np.zeros(30); sc[16:21] = [.3, .45, .6, .8, 1.0]; sc[21] = .5
rows = {20: {"text": "GOAL!", "events": [{"method": {"label": "header"}, "source": "Trent's cross"}], "minute": "34'"}}
recs = build_highlights("m", sc, valid, meta, rows, DEFAULT_CONFIG["classes"])
r = recs[0]
check("one record for the one peak", len(recs) == 1 and r["line_no"] == 20 and r["rank"] == 1)
check("player from Stage 4 agent dict", r["player"] == "Erling Haaland" and r["player_source"] == "stage4")
check("buildup length counted", r["buildup_length"] == 4, str(r["buildup_length"]))
check("time field used when present", r["time"] == "34'")

# line_no has gaps in real data (line_no != position): the peak must read ITS OWN row
meta_g = [{"line_no": 100 + 3 * i, "line_kind": "live", "primary_class": "routine", "primary_agent": None} for i in range(30)]
meta_g[20].update(primary_class="goal", primary_agent="Haaland")
rows_g = {100 + 3 * i: {"text": f"text of line {i}", "events": []} for i in range(30)}
rows_g[100 + 3 * 20]["events"] = [{"method": {"label": "header"}, "source": "cross"}]
rows_g[20] = {"text": "WRONG", "events": [{"method": {"label": "WRONG"}, "source": "WRONG"}]}   # decoy: line_no == position
sc_g = np.zeros(30); sc_g[19:22] = [.6, 1.0, .5]
rg = build_highlights("m", sc_g, np.ones(30, bool), meta_g, rows_g, DEFAULT_CONFIG["classes"], with_text=True)[0]
check("gapped line_no: text comes from the peak's own line", rg["text"] == "text of line 20" and rg["line_no"] == 160, rg["text"])
check("gapped line_no: method comes from the peak's own line", rg["method"] == "header from cross", str(rg["method"]))
check("gapped line_no: span text lines up", rg["span_text"][0] == "text of line 19" and len(rg["span_text"]) == 3)

# player: never borrowed from a different kind of line or from far away
meta_p = [{"line_no": i, "line_kind": "live", "primary_class": "routine", "primary_agent": None} for i in range(30)]
meta_p[20]["primary_class"] = "near_miss"                      # the peak: no agent found by Stage 4
meta_p[17].update(primary_class="near_miss", primary_agent="Solanke")      # same class but 3 lines away
meta_p[19].update(primary_class="foul", primary_agent="Rice")              # adjacent but a foul
sc_p = np.zeros(30); sc_p[17:21] = [.5, .5, .6, .9]
rp = build_highlights("m", sc_p, np.ones(30, bool), meta_p, {}, DEFAULT_CONFIG["classes"], span_floor=0.1)[0]
check("player is not borrowed from a far line or another class", rp["player"] is None, str(rp["player"]))
line = render_line(r)
check("rendered format",
      re.fullmatch(r"\[34', event: GOAL, player: Erling Haaland, method: header from Trent's cross, "
                   r"excitement: 1\.00, buildup_length: 4 lines, narrative_context: \"[^\"]+\"\]", line) is not None, line)
rows2 = {20: {"text": "GOAL!", "events": []}}
check("no time -> 'line N'", render_line(build_highlights("m", sc, valid, meta, rows2, DEFAULT_CONFIG["classes"])[0]).startswith("[line 20, "))

# -------------------------------------------------- 3. end to end on synthetic matches
tmp = Path(tempfile.mkdtemp())
emb, ev = tmp / "embeddings", tmp / "events"
emb.mkdir(); ev.mkdir()
rng = np.random.default_rng(1)
CLS = DEFAULT_CONFIG["classes"]
PLAYERS = ["Haaland", "Salah", "Saka", "Rodri", "Son", "Foden"]


def fake_match(mid, n=70):
    kinds = ["prematch"] * 6 + ["live"] * (n - 12) + ["recap"] * 6
    cls = []
    for i, k in enumerate(kinds):
        c = "routine"
        if k == "live" and rng.random() < 0.3 and (not cls or cls[-1] == "routine"):
            c = str(rng.choice(CLS[1:], p=[.35, .15, .25, .25]))
        cls.append(c)
    agent = [str(rng.choice(PLAYERS)) if c != "routine" else None for c in cls]
    X = rng.normal(size=(n, 48)).astype(np.float32)
    for i, c in enumerate(cls):
        X[i, CLS.index(c)] += 4.0
        if agent[i]:
            X[i, 10 + PLAYERS.index(agent[i])] += 4.0
    np.save(emb / f"emb_{mid}.npy", X)
    with open(emb / f"emb_{mid}_meta.jsonl", "w") as fm, open(ev / f"events_{mid}.jsonl", "w") as fe:
        for i in range(n):
            fm.write(json.dumps({"match_id": mid, "line_no": 2 * i + 5, "line_kind": kinds[i], "primary_type": cls[i],
                                 "primary_class": cls[i], "primary_agent": agent[i], "primary_team": None}) + "\n")
            evs = []
            if cls[i] != "routine":
                evs = [{"class": cls[i], "action": "x", "agent": {"entity": agent[i]}, "primary": True,
                        "method": {"label": "header"} if cls[i] == "goal" else None,
                        "source": "cross" if cls[i] == "goal" else None}]
            fe.write(json.dumps({"match_id": mid, "line_no": 2 * i + 5, "text": f"line {i} {cls[i]}", "line_kind": kinds[i],
                                 "events": evs}) + "\n")


ids = [f"{i:02d}" for i in range(14)]
for m in ids:
    fake_match(m)

out_w = tmp / "hl_weak"
r = subprocess.run([sys.executable, str(HERE / "run_highlights.py"), "--source", "weak", "--split", "all",
                    "--emb-dir", str(emb), "--events-dir", str(ev), "--out-dir", str(out_w)],
                   capture_output=True, text=True)
check("weak-label run exits cleanly", r.returncode == 0, r.stderr[-400:])
if r.returncode == 0:
    d = json.loads((out_w / "highlights_00.json").read_text())
    hs = d["highlights"]
    check("json has params and highlights", "params" in d and isinstance(hs, list))
    check("weak: highlights ranked by excitement", all(hs[i]["excitement"] >= hs[i + 1]["excitement"] for i in range(len(hs) - 1)))
    check("weak: spaced by min_distance", all(abs(a["line_no"] - b["line_no"]) >= 5 for i, a in enumerate(hs) for b in hs[i + 1:]))
    check("weak: top highlight is a goal or a chance, not routine", hs[0]["event"] in ("goal", "near_miss", "card"), hs[0]["event"])
    goals = [h for h in hs if h["event"] == "goal"]
    check("weak: goals carry method and player", all(h["method"] == "header from cross" and h["player"] for h in goals) if goals else True)
    check("weak: shown text is the peak's own line (line_no has gaps here)",
          all(h["text"].endswith(h["event_stage4"]) for h in hs), str([h["text"] for h in hs[:3]]))
    check("txt and csv written", (out_w / "highlights_00.txt").exists() and (out_w / "highlights_all.csv").exists())
    print((out_w / "highlights_00.txt").read_text().splitlines()[3])

model_dir = tmp / "model"
r = subprocess.run([sys.executable, str(ROOT / "Multitask heads" / "train.py"), "--emb-dir", str(emb),
                    "--labels-dir", str(tmp / "nolabels"), "--out-dir", str(model_dir), "--d-model", "32",
                    "--epochs", "25", "--lr", "2e-3", "--patience", "25", "--device", "cpu"],
                   capture_output=True, text=True)
check("training run for the model test", r.returncode == 0, r.stderr[-400:])
if r.returncode == 0:
    out_m = tmp / "hl_model"
    r = subprocess.run([sys.executable, str(HERE / "run_highlights.py"), "--split", "test", "--ckpt",
                        str(model_dir / "best.pt"), "--emb-dir", str(emb), "--events-dir", str(ev),
                        "--out-dir", str(out_m), "--device", "cpu", "--with-text"],
                       capture_output=True, text=True)
    check("model run exits cleanly", r.returncode == 0, r.stderr[-600:])
    if r.returncode == 0:
        test_ids = json.loads(json.dumps(__import__("torch").load(model_dir / "best.pt", weights_only=False)["split"]["test"]))
        files = sorted(p.name for p in out_m.glob("highlights_*.json"))
        check("one output per test match", files == [f"highlights_{m}.json" for m in sorted(test_ids)], str(files))
        hs = json.loads((out_m / files[0]).read_text())["highlights"]
        check("model: finds highlights above threshold", len(hs) >= 1)
        check("model: records have model fields", all("event_prob" in h and 0 <= h["excitement"] <= 1 for h in hs))
        check("model: top highlight is a real event", hs[0]["event_stage4"] != "routine", str(hs[0]["event_stage4"]))
        check("model: span text included", all(len(h["span_text"]) >= 1 for h in hs))
        agree = np.mean([h["event"] == h["event_stage4"] for h in hs])
        check("model: Head B agrees with Stage 4 on most peaks", agree >= 0.6, f"({agree:.2f})")
        print("   ", render_line(hs[0]))

print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
