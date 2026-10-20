#!/usr/bin/env python3
"""Create a conservative, traceable match-association review copy of a CSV.

The source CSV's match metadata and commentary blocks appear out of alignment.
This script associates a text block only when its opening line names a unique
pair of teams and explicitly names the home stadium of exactly one listed
fixture. It preserves source-row provenance, excludes ambiguous rows, and does
not create labels or timestamps. The original CSV is never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

TEAM_ALIASES = {
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
VENUE_ALIASES = {
    "amex stadium": "american express stadium",
    "the amex": "american express stadium",
    "brentford community stadium": "gtech community stadium",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", value.casefold())
                  .encode("ascii", "ignore").decode()).strip()


def canonical_team(value: str) -> str:
    key = " ".join(value.casefold().replace("&", "and").split())
    return TEAM_ALIASES.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Unmodified source match-details CSV")
    parser.add_argument("--out", type=Path, default=Path("data/23_24_match_details_reconciled_review.csv"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/23_24_match_details_reconciliation.json"))
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"id", "Home", "Away", "Date", "Stadium", "events"}
    if not source_rows or not required.issubset(source_rows[0]):
        parser.error("Source CSV must contain id, Home, Away, Date, Stadium, and events columns.")

    metadata_by_pair: dict[frozenset[str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        pair = frozenset((canonical_team(row["Home"]), canonical_team(row["Away"])))
        metadata_by_pair[pair].append(row)

    team_names = {team.strip() for row in source_rows for team in (row["Home"], row["Away"])
                  if team and team.strip()}
    team_names.update(TEAM_ALIASES)
    team_names = sorted(team_names, key=len, reverse=True)
    team_patterns = [(team, re.compile(r"(?i)(?<![\w])" + re.escape(team) + r"(?![\w])"))
                     for team in team_names]

    proposed = []
    skipped = defaultdict(int)
    for source_index, row in enumerate(source_rows, start=2):
        lines = (row.get("events") or "").splitlines()
        if not lines:
            skipped["empty_events"] += 1
            continue
        opener = lines[0]
        mentions = []
        for team, pattern in team_patterns:
            match = pattern.search(opener)
            if match:
                mentions.append((match.start(), team))
        detected = []
        seen = set()
        for _, team in sorted(mentions):
            key = canonical_team(team)
            if key not in seen:
                seen.add(key)
                detected.append(key)
        if len(detected) != 2:
            skipped["opening_pair_not_detected"] += 1
            continue

        pair = frozenset(detected)
        candidates = metadata_by_pair.get(pair, [])
        if not candidates:
            skipped["team_pair_not_in_metadata"] += 1
            continue
        opener_norm = normalize(opener)
        venue_matches = []
        matched_venues = []
        for candidate in candidates:
            venue = candidate["Stadium"].split(",", 1)[0].strip()
            venue_key = " ".join(venue.casefold().split())
            canonical_venue = VENUE_ALIASES.get(venue_key, venue_key)
            if normalize(canonical_venue) and normalize(canonical_venue) in opener_norm:
                venue_matches.append(candidate)
                matched_venues.append(venue)
        if len(venue_matches) != 1:
            skipped["venue_did_not_uniquely_confirm_fixture"] += 1
            continue
        target = venue_matches[0]
        proposed.append({
            "match_id": target["id"],
            "match_date": target["Date"],
            "home": target["Home"],
            "away": target["Away"],
            "stadium": target["Stadium"],
            "source_csv_line": source_index,
            "source_row_id_unverified": row["id"],
            "source_row_date_unverified": row["Date"],
            "source_row_home_unverified": row["Home"],
            "source_row_away_unverified": row["Away"],
            "detected_opening_teams": " | ".join(detected),
            "venue_evidence": matched_venues[0],
            "association_method": "inferred from opening team pair + exact home-stadium name",
            "review_status": "inferred; verify against source before training",
            "events": row["events"],
        })

    target_counts: dict[str, int] = defaultdict(int)
    for item in proposed:
        target_counts[item["match_id"]] += 1
    collisions = {key for key, count in target_counts.items() if count > 1}
    accepted = [item for item in proposed if item["match_id"] not in collisions]
    for item in proposed:
        if item["match_id"] in collisions:
            skipped["multiple_source_blocks_for_target_fixture"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(accepted[0]) if accepted else ["match_id", "events"]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    audit = {
        "source": str(args.csv),
        "source_rows": len(source_rows),
        "accepted_review_rows": len(accepted),
        "rejected_source_rows": len(source_rows) - len(accepted),
        "accepted_target_ids_unique": len({item["match_id"] for item in accepted}) == len(accepted),
        "association_method": "opening team pair plus an exact home-stadium name from the candidate fixture; unique match required",
        "warning": "This inferred reassociation is a review aid, not ground truth. It provides no event timestamps or labels and must be confirmed before model training.",
        "skip_reasons": dict(skipped),
        "output": str(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Accepted for review: {len(accepted)} unique fixture associations")
    print(f"Excluded as empty, ambiguous, conflicting, or unconfirmed: {len(source_rows) - len(accepted)}")
    print(f"Review copy: {args.out}")
    print(f"Audit summary: {args.audit}")


if __name__ == "__main__":
    main()
