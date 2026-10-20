#!/usr/bin/env python3
"""Turn a SoccerNet/MatchTime Labels-caption.json file into a ranked timeline."""

import argparse
import csv
import json
from pathlib import Path


def hhmmss(seconds: float) -> str:
    n = max(0, int(seconds))
    return f"{n // 3600:02d}:{(n % 3600) // 60:02d}:{n % 60:02d}"


def priority(row: dict) -> tuple[float, str]:
    """Transparent presentation priority; `important` remains the dataset label."""
    label = (row.get("label") or "").casefold()
    text = (row.get("description") or "").casefold()
    if label == "soccer-ball" or "goal!" in text or "score is" in text:
        return 1.0, "goal"
    if "red card" in text or "sent off" in text:
        return 0.95, "red_card"
    if any(term in text for term in ("great opportunity", "great chance", "exquisite save", "flies just a few", "narrowly wide", "blocked by")):
        return 0.82, "chance_or_save"
    if label in {"penalty", "var"}:
        return 0.78, label
    if label in {"y-card", "yellow-card"}:
        return 0.62, "yellow_card"
    if label == "substitution":
        return 0.40, "substitution"
    if label == "whistle":
        return 0.35, "whistle"
    return (0.50 if row.get("important") else 0.20), (label or "notable_event")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_json", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/caption_highlights"))
    args = parser.parse_args()
    data = json.loads(args.labels_json.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ann in data.get("annotations", []):
        if not ann.get("important"):
            continue
        game_time = ann.get("gameTime", "")
        try:
            period_text, clock = [x.strip() for x in game_time.split("-", 1)]
            period = int(period_text)
            mm, ss = clock.split(":")
            game_time_seconds = int(mm) * 60 + int(ss)
        except (ValueError, AttributeError):
            period, game_time_seconds = 0, 0
        video_position = int(ann.get("position") or 0) / 1000.0
        score, predicted_type = priority(ann)
        rows.append({
            "period": period,
            "video_file": f"{period}_720p.mkv" if period in (1, 2) else "check_period.mkv",
            "source_video_position_seconds": round(video_position, 2),
            "source_video_position": hhmmss(video_position),
            "game_clock": game_time,
            "contrastive_aligned_game_clock": ann.get("contrastive_aligned_gameTime", ""),
            "event_label": ann.get("label", ""),
            "dataset_important": bool(ann.get("important")),
            "baseline_priority": round(score, 2),
            "baseline_event_type": predicted_type,
            "commentary_caption": ann.get("description", ""),
        })

    rows.sort(key=lambda x: (-x["baseline_priority"], x["period"], x["source_video_position_seconds"]))
    fields = list(rows[0]) if rows else []
    with (args.out / "ranked_highlights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "match": " - ".join(data.get("teams", [])),
        "score": data.get("score"),
        "match_date": data.get("gameDate"),
        "source": args.labels_json.name,
        "method_note": "Uses the dataset's timestamped captions and important annotations. Priority is a transparent baseline, not an audio transcription or trained-model prediction.",
        "timestamp_note": "source_video_position is the per-half video offset from the JSON position field; verify each point in the matching half before cutting clips.",
        "ranked_highlights": rows,
    }
    (args.out / "ranked_highlights.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Ranked highlight commentary: {' - '.join(data.get('teams', []))} ({data.get('score', '')})",
        "",
        f"Match date: {data.get('gameDate', 'unknown')}",
        "",
        "Input is timestamped dataset captions with an `important` annotation. Priority is a transparent baseline, not Whisper output or a trained-model prediction.",
        "Video offsets are per half; open the matching `1_720p.mkv` or `2_720p.mkv` and verify timestamps before editing.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        lines.extend([
            f"## {i}. {row['baseline_event_type'].replace('_', ' ').title()} — {row['video_file']} at {row['source_video_position']}",
            f"Priority: {row['baseline_priority']:.2f} | Dataset label: {row['event_label'] or 'unlabelled important caption'} | Match clock: {row['game_clock']}",
            "",
            row["commentary_caption"],
            "",
        ])
    (args.out / "ranked_highlights.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Match: {' - '.join(data.get('teams', []))} {data.get('score', '')}")
    print(f"Timestamped important captions: {len(rows)}")
    print(f"Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
