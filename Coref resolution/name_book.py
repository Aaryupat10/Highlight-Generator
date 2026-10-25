"""Building per-match name book (players, roles, team aliases) from lineup lines.

Run from the repo root:
    python "Coref resolution/name_book.py" data/23_24_commentary_clean.csv

Everything specific to one sport (lineup line format, surname particles, short team
names, team nicknames) is read from a config file, configs/football.json by default.
Use --config to point at another sport's file. See configs/README.md.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from sport_config import DEFAULT_CONFIG, SportConfig, load_config


def split_players(body: str) -> list[str]:
    body = re.sub(r"\([^)]*\)", "", body)      # drop notes like "(captain)"
    body = body.strip().rstrip(".")
    return [p.strip() for p in re.split(r"[;,:]", body) if p.strip()]


def assign_roles(formation: str | None, n_players: int, cfg: SportConfig) -> list[str | None]:
    # lineups list the keeper first, then defence, midfield, attack
    if not cfg.roles_from_formation or not formation:
        return [None] * n_players              # this sport has no formation roles
    parts = [int(x) for x in formation.split("-")]
    if n_players != 1 + sum(parts):
        return [None] * n_players              # not 11 names, so don't guess
    return (
        ["GK"]
        + ["DEF"] * parts[0]
        + ["MID"] * sum(parts[1:-1])
        + ["FWD"] * parts[-1]
    )


def player_aliases(full_name: str, cfg: SportConfig) -> set[str]:
    aliases = {full_name}
    tokens = full_name.split()
    if len(tokens) > 1:
        i = len(tokens) - 1
        while i > 0 and tokens[i - 1].lower() in cfg.surname_particles:
            i -= 1
        aliases.add(" ".join(tokens[i:]))      # surname, including any particle
    return aliases


def derive_short_names(name: str, cfg: SportConfig) -> set[str]:
    # "Manchester City" -> City, Manchester, Man City
    words = name.split()
    out = {name}
    if len(words) >= 2 and words[-1] in cfg.distinctive_last_words:
        out.add(words[-1])
        if words[0] not in cfg.generic_first_words:
            out.add(words[0])
        if words[0] in cfg.first_word_abbreviations:
            out.add(f"{cfg.first_word_abbreviations[words[0]]} {words[-1]}")
    return out


def load_teams(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}                              # no team list: teams still work, just no nicknames
    return json.loads(path.read_text(encoding="utf-8"))["teams"]


def resolve_team(header: str, teams: dict[str, dict], cfg: SportConfig) -> tuple[str, set[str], bool]:
    """Return (canonical name, aliases, whether the team is listed in the teams file)."""
    h = header.strip().upper()
    if cfg.header_suffix_re:
        h = cfg.header_suffix_re.sub("", h)
    for canonical, info in teams.items():
        if any(h == k or h.startswith(k + " ") for k in info["header_keys"]):
            return canonical, derive_short_names(canonical, cfg) | set(info.get("nicknames", [])), True
    # team not in the config: keep the header and derive short names, no nicknames
    unknown = h.title()
    return unknown, derive_short_names(unknown, cfg), False


def split_unique_ambiguous(alias_to_owners: dict[str, set[str]]) -> tuple[dict, dict]:
    unique = {a: sorted(v)[0] for a, v in alias_to_owners.items() if len(v) == 1}
    ambiguous = {a: sorted(v) for a, v in alias_to_owners.items() if len(v) > 1}
    return unique, ambiguous


def build_match(lines: list[tuple[int, str]], teams_cfg: dict[str, dict],
                cfg: SportConfig) -> tuple[dict, dict[int, str]]:
    line_types: dict[int, str] = {}
    teams: list[dict] = []
    players: list[dict] = []
    current_team = None                        # plain "SUBS:" lines belong to the last lineup's team
    unknown_teams: list[str] = []

    for line_no, text in lines:
        m = cfg.lineup_re.match(text) if cfg.lineup_re else None
        if m and m.group("team").isupper():
            line_types[line_no] = "lineup"
            canonical, team_aliases, known = resolve_team(m.group("team"), teams_cfg, cfg)
            if any(t["name"] == canonical for t in teams):
                continue                       # some matches repeat a lineup line
            if not known:
                unknown_teams.append(m.group("team"))
            names = split_players(m.group("body"))
            formation = m.groupdict().get("formation")
            roles = assign_roles(formation, len(names), cfg)
            if cfg.first_lineup_is_home:
                side = "home" if not teams else "away"   # first lineup is the home side
            else:
                side = None                    # sport does not say which lineup is home
            teams.append({
                "name": canonical,
                "side": side,
                "formation": formation,
                "aliases": sorted(team_aliases),
            })
            current_team = canonical
            for name, role in zip(names, roles):
                players.append({"name": name, "team": canonical, "role": role, "starter": True})
            continue

        s = cfg.subs_re.match(text) if cfg.subs_re else None
        if s and s.group("body").count(",") + s.group("body").count(";") >= 2:
            line_types[line_no] = "subs"
            team = resolve_team(s.group("team"), teams_cfg, cfg)[0] if s.group("team") else current_team
            for name in split_players(s.group("body")):
                players.append({"name": name, "team": team, "role": None, "starter": False})
            continue

        line_types[line_no] = "commentary"

    # an alias shared by two players ("Gomes") is ambiguous and left for the resolver
    player_owners: dict[str, set[str]] = defaultdict(set)
    for p in players:
        p["aliases"] = sorted(player_aliases(p["name"], cfg))
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
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sport config (JSON), see configs/README.md")
    ap.add_argument("--teams", type=Path, default=None,
                    help="team list (JSON); default is the teams_file named in the sport config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    teams_cfg = load_teams(args.teams or cfg.teams_file)
    df = pd.read_csv(args.input_csv, dtype={"clean_text": str}).fillna({"clean_text": ""})
    df = df.sort_values(["match_id", "line_no"])

    books: dict[str, dict] = {}
    type_rows: list[dict] = []
    for match_id, grp in df.groupby("match_id"):
        lines = list(zip(grp["line_no"], grp["clean_text"]))
        book, line_types = build_match(lines, teams_cfg, cfg)
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
    print(f"matches with team not in the teams file: {sum(bool(b['unknown_teams']) for b in books.values())}")
    print(f"matches with ambiguous player aliases: {sum(bool(b['ambiguous_aliases']) for b in books.values())}")
    print(f"matches with ambiguous team aliases: {sum(bool(b['ambiguous_team_aliases']) for b in books.values())}")
    print(types_df.line_type.value_counts().to_string())
    print(f"saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
