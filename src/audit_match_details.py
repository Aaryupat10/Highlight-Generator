#!/usr/bin/env python3
"""Audit a match-details CSV before using its commentary for training.

This is a read-only source audit. It does not import or label the commentary.
It checks whether opening text names agree with the row's declared teams and
whether the event text appears to contain usable per-line timestamps.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ALIASES = {
    "man city": "manchester city", "manchester city": "manchester city",
    "newcastle": "newcastle united", "newcastle united": "newcastle united",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "nott'm forest": "nottingham forest", "nottingham forest": "nottingham forest",
    "sheffield utd": "sheffield united", "sheffield united": "sheffield united",
    "brighton": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "west ham": "west ham united", "west ham united": "west ham united",
    "man utd": "manchester united", "manchester united": "manchester united",
}


def canonical(value: str) -> str:
    key = " ".join(value.casefold().replace("&", "and").split())
    return ALIASES.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Source match-details CSV")
    parser.add_argument("--out", type=Path, help="Optional JSON audit output")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    required = {"id", "Home", "Away", "Date", "events", "summary"}
    missing_columns = sorted(required - set(columns))
    if missing_columns:
        parser.error(f"Missing required columns: {', '.join(missing_columns)}")

    teams = {team.strip() for row in rows for team in (row["Home"], row["Away"])
             if team and team.strip()}
    # Text can use full names where the metadata uses a short display name.
    teams.update(ALIASES)
    teams = sorted(teams, key=len, reverse=True)
    team_patterns = [(team, re.compile(r"(?i)(?<![\w])" + re.escape(team) + r"(?![\w])"))
                     for team in teams]
    declared_pairs = {frozenset((canonical(row["Home"]), canonical(row["Away"]))) for row in rows}
    declared_ordered_pairs = {(canonical(row["Home"]), canonical(row["Away"])) for row in rows}
    audit_rows = []
    for number, row in enumerate(rows, start=2):  # header is line 1
        opening = " ".join((row.get("events") or "").splitlines()[:4])[:700]
        found = []
        for team, pattern in team_patterns:
            match = pattern.search(opening)
            if match:
                found.append((match.start(), team))
        found.sort()
        opening_teams = []
        seen = set()
        for _, team in found:
            norm = canonical(team)
            if norm not in seen:
                seen.add(norm)
                opening_teams.append(team)
        opening_teams = opening_teams[:2]
        declared = {canonical(row.get("Home", "")), canonical(row.get("Away", ""))}
        pair_checked = len(opening_teams) == 2
        pair_matches = pair_checked and {canonical(t) for t in opening_teams} == declared
        text = row.get("events") or ""
        audit_rows.append({
            "csv_line": number,
            "id": row.get("id", ""),
            "date": row.get("Date", ""),
            "home": row.get("Home", ""),
            "away": row.get("Away", ""),
            "opening_teams_detected": opening_teams,
            "opening_pair_checked": pair_checked,
            "opening_pair_matches_metadata": bool(pair_matches),
            "opening_pair_exists_in_metadata_set": bool(
                pair_checked and frozenset(canonical(t) for t in opening_teams) in declared_pairs
            ),
            "opening_order_exists_in_metadata_set": bool(
                pair_checked and tuple(canonical(t) for t in opening_teams) in declared_ordered_pairs
            ),
            "event_lines": len(text.splitlines()) if text else 0,
            "event_text_chars": len(text),
            "has_mm_ss_timestamp": bool(re.search(r"\b\d{1,2}:\d{2}\b", text)),
            "has_minute_marker": bool(re.search(r"\b\d{1,3}(?:\+\d{1,2})?\s*(?:min|minute|mins)\b", text, re.I)),
        })

    opening_pair_counts = Counter(
        tuple(canonical(t) for t in row["opening_teams_detected"])
        for row in audit_rows if row["opening_pair_checked"]
    )
    summary = {
        "source": str(args.csv),
        "row_count": len(rows),
        "columns": columns,
        "unique_ids": len({r.get("id", "") for r in rows if r.get("id", "").strip()}),
        "date_range": [min((r.get("Date", "") for r in rows if r.get("Date")), default=""),
                       max((r.get("Date", "") for r in rows if r.get("Date")), default="")],
        "missing_values": {name: sum(not (r.get(name) or "").strip() for r in rows)
                           for name in columns},
        "opening_pair_checked": sum(r["opening_pair_checked"] for r in audit_rows),
        "opening_pair_matches_metadata": sum(r["opening_pair_matches_metadata"] for r in audit_rows),
        "opening_pair_exists_in_metadata_set": sum(r["opening_pair_exists_in_metadata_set"] for r in audit_rows),
        "opening_order_exists_in_metadata_set": sum(r["opening_order_exists_in_metadata_set"] for r in audit_rows),
        "repeated_opening_pair_keys": sum(count > 1 for count in opening_pair_counts.values()),
        "rows_with_mm_ss_timestamp": sum(r["has_mm_ss_timestamp"] for r in audit_rows),
        "rows_with_minute_marker": sum(r["has_minute_marker"] for r in audit_rows),
        "audit_rows": audit_rows,
    }
    print(f"Rows: {summary['row_count']}; unique IDs: {summary['unique_ids']}; date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Opening team pair checked: {summary['opening_pair_checked']}; matches declared row teams: {summary['opening_pair_matches_metadata']}")
    print(f"Opening pair exists somewhere in metadata (without date/orientation assurance): {summary['opening_pair_exists_in_metadata_set']}")
    print(f"Opening team order exists somewhere in metadata: {summary['opening_order_exists_in_metadata_set']}; repeated detected pair keys: {summary['repeated_opening_pair_keys']}")
    print(f"Rows with mm:ss-like event text: {summary['rows_with_mm_ss_timestamp']}; rows with minute markers: {summary['rows_with_minute_marker']}")
    for name, count in summary["missing_values"].items():
        if count:
            print(f"Missing {name}: {count}")
    mismatches = [r for r in audit_rows if r["opening_pair_checked"] and not r["opening_pair_matches_metadata"]]
    for item in mismatches[:5]:
        print(f"Mismatch, CSV line {item['csv_line']} id={item['id']}: metadata {item['home']} - {item['away']}; opening text names {', '.join(item['opening_teams_detected'])}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Detailed audit: {args.out}")


if __name__ == "__main__":
    main()
