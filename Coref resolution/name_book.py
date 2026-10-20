"""Building per-match name book (players, roles, team aliases) from lineup lines.

Run from the repo root:
    python "Coref resolution/name_book.py" data/23_24_commentary_clean.csv
Team nicknames are read from teams.json, which sits next to this file.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_TEAMS_FILE = Path(__file__).with_name("teams.json")

# "MANCHESTER CITY (4-2-3-1): Ederson; Kyle Walker, ..."
# the separator after the formation varies in the data, so it is kept loose
LINEUP_RE = re.compile(
    r"^(?P<team>[A-Z][A-Za-z .&'-]*?)\s*"
    r"\((?P<formation>\d(?:-\d)+)\)\s*[:;-]*\s*(?P<body>.+)$"
)
# "SUBS: ...", "TOTTENHAM SUBS: ...", "Subs: ..." (case-sensitive so normal sentences don't match)
SUBS_RE = re.compile(
    r"^(?:(?P<team>[A-Z][A-Z .&'-]*?)\s+)?(?:SUBS|Subs|SUBSTITUTES|Substitutes)\s*[:;-]\s*(?P<body>.+)$"
)
# "ARSENAL STARTING XI" -> "ARSENAL"
HEADER_SUFFIX_RE = re.compile(r"\s+(?:STARTING XI|STARTING LINEUP|LINEUP|XI)$")

# surname particles that stay attached: "De Bruyne", "van Hecke", "Al Dakhil"
PARTICLES = {"de", "da", "di", "dos", "van", "von", "der", "al", "el", "le", "la", "ten", "ter"}
# last words that work as a short team name on their own ("City", "United")
DISTINCTIVE_LAST_WORDS = {
    "City", "United", "Town", "Forest", "Palace", "Villa",
    "Albion", "Wanderers", "Hotspur", "Rovers", "Athletic",
}
# first words too generic to stand alone ("West" from "West Ham")
GENERIC_FIRST_WORDS = {"West", "North", "South", "East", "Crystal", "Aston", "Real", "Inter"}


def split_players(body: str) -> list[str]:
    body = re.sub(r"\([^)]*\)", "", body)      # drop notes like "(captain)"
    body = body.strip().rstrip(".")
    return [p.strip() for p in re.split(r"[;,:]", body) if p.strip()]


def assign_roles(formation: str, n_players: int) -> list[str | None]:
    # lineups list the keeper first, then defence, midfield, attack
    parts = [int(x) for x in formation.split("-")]
    if n_players != 1 + sum(parts):
        return [None] * n_players              # not 11 names, so don't guess
    return (
        ["GK"]
        + ["DEF"] * parts[0]
        + ["MID"] * sum(parts[1:-1])
        + ["FWD"] * parts[-1]
    )


def player_aliases(full_name: str) -> set[str]:
    aliases = {full_name}
    tokens = full_name.split()
    if len(tokens) > 1:
        i = len(tokens) - 1
        while i > 0 and tokens[i - 1].lower() in PARTICLES:
            i -= 1
        aliases.add(" ".join(tokens[i:]))      # surname, including any particle
    return aliases


def derive_short_names(name: str) -> set[str]:
    # "Manchester City" -> City, Manchester, Man City
    words = name.split()
    out = {name}
    if len(words) >= 2 and words[-1] in DISTINCTIVE_LAST_WORDS:
        out.add(words[-1])
        if words[0] not in GENERIC_FIRST_WORDS:
            out.add(words[0])
        if words[0] == "Manchester":
            out.add(f"Man {words[-1]}")
    return out


def load_teams(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["teams"]


def resolve_team(header: str, teams: dict[str, dict]) -> tuple[str, set[str], bool]:
    """Return (canonical name, aliases, whether the team is listed in teams.json)."""
    h = HEADER_SUFFIX_RE.sub("", header.strip().upper())
    for canonical, info in teams.items():
        if any(h == k or h.startswith(k + " ") for k in info["header_keys"]):
            return canonical, derive_short_names(canonical) | set(info.get("nicknames", [])), True
    # team not in the config: keep the header and derive short names, no nicknames
    unknown = h.title()
    return unknown, derive_short_names(unknown), False


def split_unique_ambiguous(alias_to_owners: dict[str, set[str]]) -> tuple[dict, dict]:
    unique = {a: sorted(v)[0] for a, v in alias_to_owners.items() if len(v) == 1}
    ambiguous = {a: sorted(v) for a, v in alias_to_owners.items() if len(v) > 1}
    return unique, ambiguous


def build_match(lines: list[tuple[int, str]], teams_cfg: dict[str, dict]) -> tuple[dict, dict[int, str]]:
    line_types: dict[int, str] = {}
    teams: list[dict] = []
    players: list[dict] = []
    current_team = None                        # plain "SUBS:" lines belong to the last lineup's team
    unknown_teams: list[str] = []

    for line_no, text in lines:
        m = LINEUP_RE.match(text)
        if m and m.group("team").isupper():
            line_types[line_no] = "lineup"
            canonical, team_aliases, known = resolve_team(m.group("team"), teams_cfg)
            if any(t["name"] == canonical for t in teams):
                continue                       # some matches repeat a lineup line
            if not known:
                unknown_teams.append(m.group("team"))
            names = split_players(m.group("body"))
            roles = assign_roles(m.group("formation"), len(names))
            teams.append({
                "name": canonical,
                "side": "home" if not teams else "away",   # first lineup is the home side
                "formation": m.group("formation"),
                "aliases": sorted(team_aliases),
            })
            current_team = canonical
            for name, role in zip(names, roles):
                players.append({"name": name, "team": canonical, "role": role, "starter": True})
            continue

        s = SUBS_RE.match(text)
        if s and s.group("body").count(",") + s.group("body").count(";") >= 2:
            line_types[line_no] = "subs"
            team = resolve_team(s.group("team"), teams_cfg)[0] if s.group("team") else current_team
            for name in split_players(s.group("body")):
                players.append({"name": name, "team": team, "role": None, "starter": False})
            continue

        line_types[line_no] = "commentary"

    # an alias shared by two players ("Gomes") is ambiguous and left for the resolver
    player_owners: dict[str, set[str]] = defaultdict(set)
    for p in players:
        p["aliases"] = sorted(player_aliases(p["name"]))
        for a in p["aliases"]:
            player_owners[a].add(p["name"])
    alias_index, ambiguous = split_unique_ambiguous(player_owners)

    # same check for teams ("United" in a match with two United sides)
    team_owners: dict[str, set[str]] = defaultdict(set)
    for t in teams:
        for a in t["aliases"]:
            team_owners[a].add(t["name"])
    team_alias_index, ambiguous_team = split_unique_ambiguous(team_owners)

    book = {
        "teams": teams,
        "players": players,
        "alias_index": alias_index,
        "ambiguous_aliases": ambiguous,
        "team_alias_index": team_alias_index,
        "ambiguous_team_aliases": ambiguous_team,
        "unknown_teams": unknown_teams,
    }
    return book, line_types


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-match name books from the cleaned commentary CSV.")
    ap.add_argument("input_csv", type=Path, nargs="?", default=Path("data/23_24_commentary_clean.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/name_book"))
    ap.add_argument("--teams", type=Path, default=DEFAULT_TEAMS_FILE, help="team nickname config (JSON)")
    args = ap.parse_args()

    teams_cfg = load_teams(args.teams)
    df = pd.read_csv(args.input_csv, dtype={"clean_text": str}).fillna({"clean_text": ""})
    df = df.sort_values(["match_id", "line_no"])

    books: dict[str, dict] = {}
    type_rows: list[dict] = []
    for match_id, grp in df.groupby("match_id"):
        lines = list(zip(grp["line_no"], grp["clean_text"]))
        book, line_types = build_match(lines, teams_cfg)
        books[str(match_id)] = book
        type_rows += [{"match_id": match_id, "line_no": n, "line_type": t} for n, t in line_types.items()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "name_books.json").write_text(
        json.dumps(books, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    types_df = pd.DataFrame(type_rows)
    types_df.to_csv(args.out_dir / "line_types.csv", index=False)

    # summary
    print(f"matches: {len(books)}")
    print(f"matches with 2 lineups: {sum(len(b['teams']) == 2 for b in books.values())}")
    print(f"matches with 22 starters: {sum(sum(p['starter'] for p in b['players']) == 22 for b in books.values())}")
    print(f"matches with team not in teams.json: {sum(bool(b['unknown_teams']) for b in books.values())}")
    print(f"matches with ambiguous player aliases: {sum(bool(b['ambiguous_aliases']) for b in books.values())}")
    print(f"matches with ambiguous team aliases: {sum(bool(b['ambiguous_team_aliases']) for b in books.values())}")
    print(types_df.line_type.value_counts().to_string())
    print(f"saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
