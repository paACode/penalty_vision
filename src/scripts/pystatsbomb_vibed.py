#!/usr/bin/env python3
"""Where should a goalkeeper dive on a penalty?

Downloads every penalty (regular + shoot-out) from the free StatsBomb open
data exposed by statsbombpy, projects each shot's end location onto a 3x3
grid of the goal mouth (9 zones, in percent of on-target shots) and reports
where takers actually place the ball -- from the GOALKEEPER's point of view.

Grid (goalkeeper facing the taker):
    rows   : top / middle / bottom          height bands of the 2.44 m frame
    columns: keeper-left / centre / keeper-right  width bands of the 7.32 m frame

Usage:
    python penalty_target_grid.py --list                    # show free competitions
    python penalty_target_grid.py                           # default tournaments
    python penalty_target_grid.py --competitions "FIFA World Cup" "UEFA Euro"
    python penalty_target_grid.py --all                     # every free competition (slow)
    python penalty_target_grid.py --gender male
    python penalty_target_grid.py --limit-matches 25        # quick smoke test
    python penalty_target_grid.py --refresh                 # ignore cache

Requires: pip install statsbombpy pandas
Outputs : data/penalty_shots.csv, data/penalty_zone_summary.csv (+ data/cache/)
"""

import argparse
import hashlib
import pickle
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

GOAL_Y_LOW = 36.0
GOAL_Y_HIGH = 44.0
GOAL_WIDTH = GOAL_Y_HIGH - GOAL_Y_LOW
GOAL_HEIGHT = 2.44
HALF_X = 60.0
ZONE_TOL = 0.05

ROW_NAMES = ("top", "middle", "bottom")
COL_NAMES = ("keeper-left", "centre", "keeper-right")

DEFAULT_COMPETITIONS = ("FIFA World Cup", "UEFA Euro", "Copa America")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

CSV_COLUMNS = [
    "match_id", "match_date", "competition", "season", "home_team", "away_team",
    "team", "player", "shootout", "period", "minute", "shot_body_part",
    "shot_outcome", "goal", "saved", "on_target", "status",
    "zone_row", "zone_col", "off_side", "off_over",
    "start_x", "start_y", "end_x", "end_y", "end_z",
    "keeper_side_frac", "height_frac",
]


def norm(text):
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(ch)
    )
    return stripped.casefold()


def clean(value):
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_loc(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            out = [float(value[0]), float(value[1])]
            if len(value) >= 3:
                out.append(float(value[2]))
            return out
        except (TypeError, ValueError):
            return None
    return None


def classify_end(end):
    res = {
        "on_target": False,
        "status": "no end location",
        "zone_row": None,
        "zone_col": None,
        "off_side": None,
        "off_over": False,
        "keeper_side_frac": None,
        "height_frac": None,
    }
    if not end:
        return res
    ey = end[1]
    ez = end[2] if len(end) >= 3 else None
    attacking_plus_x = end[0] >= HALF_X
    if attacking_plus_x:
        shooter_frac = (GOAL_Y_HIGH - ey) / GOAL_WIDTH
    else:
        shooter_frac = (ey - GOAL_Y_LOW) / GOAL_WIDTH
    keeper_frac = 1.0 - shooter_frac
    res["keeper_side_frac"] = round(keeper_frac, 4)
    if ez is not None:
        res["height_frac"] = round(max(ez, 0.0) / GOAL_HEIGHT, 4)
    y_in = (GOAL_Y_LOW - ZONE_TOL) <= ey <= (GOAL_Y_HIGH + ZONE_TOL)
    z_in = ez is not None and (-ZONE_TOL <= ez <= GOAL_HEIGHT + ZONE_TOL)
    col = min(max(int(keeper_frac * 3), 0), 2)
    if y_in and z_in:
        res["on_target"] = True
        res["status"] = "on target"
        res["zone_col"] = COL_NAMES[col]
        res["zone_row"] = ROW_NAMES[min(max(int((ez / GOAL_HEIGHT) * 3), 0), 2)]
    elif y_in and ez is None:
        res["on_target"] = True
        res["status"] = "on target (height unknown)"
        res["zone_col"] = COL_NAMES[col]
    elif not y_in:
        res["status"] = "off target"
        res["off_side"] = COL_NAMES[2] if keeper_frac > 0.5 else COL_NAMES[0]
        res["off_over"] = ez is not None and ez > GOAL_HEIGHT + ZONE_TOL
    else:
        res["status"] = "off target"
        res["off_over"] = True
    return res


def build_record(ev, competition, season, match):
    outcome = clean(ev.get("shot_outcome"))
    start = as_loc(ev.get("location"))
    end = as_loc(ev.get("shot_end_location"))
    rec = {
        "match_id": to_int(match.get("match_id")),
        "match_date": clean(match.get("match_date")),
        "competition": competition,
        "season": season,
        "home_team": clean(match.get("home_team")),
        "away_team": clean(match.get("away_team")),
        "team": clean(ev.get("team")),
        "player": clean(ev.get("player")),
        "shootout": to_int(ev.get("period")) == 5,
        "period": to_int(ev.get("period")),
        "minute": to_int(ev.get("minute")),
        "shot_body_part": clean(ev.get("shot_body_part")),
        "shot_outcome": outcome,
        "goal": outcome == "Goal",
        "saved": str(outcome).startswith("Saved"),
        "start_x": start[0] if start else None,
        "start_y": start[1] if start else None,
        "end_x": end[0] if end else None,
        "end_y": end[1] if end else None,
        "end_z": end[2] if (end and len(end) >= 3) else None,
    }
    rec.update(classify_end(end))
    return rec


def list_competitions(comps):
    view = comps.groupby(["competition_name", "competition_gender"])["season_name"].apply(list)
    print("Free competitions available through statsbombpy:")
    for (name, gender), seasons in sorted(view.items()):
        print(f"  {name}  [{gender}]  {len(seasons)} season(s): {', '.join(map(str, seasons))}")
    print("\nPick one with --competitions NAME (substring, case/accent-insensitive) or use --all.")


def choose_competitions(comps, args):
    df = comps.drop_duplicates(subset=["competition_id", "season_id"]).copy()
    if args.gender in ("male", "female"):
        df = df[df["competition_gender"] == args.gender]
    if args.all:
        return df
    wanted = args.competitions if args.competitions else list(DEFAULT_COMPETITIONS)
    patterns = [norm(w) for w in wanted]
    names = df["competition_name"].map(norm)
    keep = df[names.map(lambda n: any(p in n for p in patterns))]
    for label, pattern in zip(wanted, patterns):
        if not names.map(lambda n: pattern in n).any():
            print(f"WARNING: no free competition matches '{label}' (try --list)")
    return keep


def format_scope(comp_df):
    lines = []
    for name, grp in comp_df.groupby("competition_name"):
        seasons = [str(s) for s in grp["season_name"]]
        if len(seasons) <= 8:
            detail = ", ".join(seasons)
        else:
            detail = f"{len(seasons)} seasons"
        lines.append(f"- {name} ({detail})")
    return lines


def fetch_penalty_records(comp_df, limit_matches=None):
    records = []
    matches_seen = 0
    failed = 0
    seasons_failed = []
    for comp in comp_df.itertuples():
        cname = comp.competition_name
        sname = comp.season_name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdf = sb.matches(
                    competition_id=int(comp.competition_id),
                    season_id=int(comp.season_id),
                )
        except Exception:
            seasons_failed.append(f"{cname} {sname}")
            continue
        if mdf is None or len(mdf) == 0:
            continue
        if "match_date" in mdf.columns:
            mdf = mdf.sort_values("match_date", na_position="last")
        for _, m in mdf.iterrows():
            if limit_matches is not None and matches_seen >= limit_matches:
                return records, matches_seen, failed, seasons_failed
            matches_seen += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ev = sb.events(match_id=int(m["match_id"]))
                if ev is None or "shot_type" not in ev.columns or "type" not in ev.columns:
                    pens = pd.DataFrame()
                else:
                    pens = ev[(ev["type"] == "Shot") & (ev["shot_type"] == "Penalty")]
            except Exception:
                failed += 1
                continue
            if len(pens) == 0:
                if matches_seen % 50 == 0:
                    print(f"  ... {matches_seen} matches scanned ({len(records)} penalties so far)")
                continue
            for _, p in pens.iterrows():
                records.append(build_record(p, cname, sname, m))
            date = clean(m.get("match_date")) or ""
            home = clean(m.get("home_team")) or "?"
            away = clean(m.get("away_team")) or "?"
            print(
                f"  [{matches_seen:>5}] {date} {home} vs {away} "
                f"({cname} {sname}): {len(pens)} penalties"
            )
    return records, matches_seen, failed, seasons_failed


def cache_path_for(comp_df):
    tag = "|".join(
        sorted(f"{int(r.competition_id)}:{int(r.season_id)}" for r in comp_df.itertuples())
    )
    digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"penalties_{digest}.pkl"


def load_records(comp_df, args):
    use_cache = args.limit_matches is None
    cp = cache_path_for(comp_df)
    if use_cache and cp.exists() and not args.refresh:
        print(f"Using cached penalty data: {cp}")
        with open(cp, "rb") as fh:
            blob = pickle.load(fh)
        return pd.DataFrame(blob["records"], columns=CSV_COLUMNS), blob
    print("Downloading free StatsBomb open data (first run can take a while)...")
    records, matches_seen, failed, seasons_failed = fetch_penalty_records(
        comp_df, args.limit_matches
    )
    meta = {
        "records": records,
        "matches_seen": matches_seen,
        "failed": failed,
        "seasons_failed": seasons_failed,
    }
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as fh:
            pickle.dump(meta, fh)
        print(f"Cached to {cp}")
    return pd.DataFrame(records, columns=CSV_COLUMNS), meta


def zone_counts(sub):
    if len(sub) == 0:
        return pd.DataFrame(0, index=list(ROW_NAMES), columns=list(COL_NAMES))
    ct = pd.crosstab(sub["zone_row"], sub["zone_col"])
    return ct.reindex(index=list(ROW_NAMES), columns=list(COL_NAMES)).fillna(0).astype(int)


def print_grid(counts):
    n = int(counts.values.sum())
    best = counts.stack().idxmax() if n else None
    cell_w = 17
    print()
    print(" " * 13 + "".join(f"{c:>{cell_w}}" for c in COL_NAMES))
    for r in ROW_NAMES:
        cells = []
        for c in COL_NAMES:
            share = 100.0 * counts.at[r, c] / n if n else 0.0
            cells.append(f"{share:4.1f}% ({counts.at[r, c]:>3d})".rjust(cell_w))
        line = f"{r.upper():<12s}" + "".join(cells)
        if best and r == best[0]:
            line += "   <-- most targeted band"
        print(line)


def print_zone_table(counts, goals, saves, n_grid, total):
    print("\nZONE DETAILS (sorted by frequency)")
    print(
        f"  {'zone':<24s}{'n':>5}{'% on-tgt':>10}{'% all':>8}"
        f"{'goals':>7}{'scored%':>9}{'saved%':>8}"
    )
    stacked = counts.stack().sort_values(ascending=False)
    for (r, c), n in stacked.items():
        g = int(goals.at[r, c])
        s = int(saves.at[r, c])
        print(
            f"  {r + ' / ' + c:<24s}{n:>5}{100.0 * n / n_grid:>9.1f}%"
            f"{100.0 * n / total:>7.1f}%{g:>7}{100.0 * g / n:>8.1f}%"
            f"{100.0 * s / n:>7.1f}%"
        )


def print_shootout_split(reg, so):
    print("\nREGULAR vs SHOOT-OUT penalties (counts per zone)")
    print(f"  {'zone':<24s}{'regular':>9}{'shoot-out':>11}")
    for r in ROW_NAMES:
        for c in COL_NAMES:
            print(f"  {r + ' / ' + c:<24s}{reg.at[r, c]:>9}{so.at[r, c]:>11}")


def print_recommendation(counts, goals, n_grid, total, n_off):
    stacked = counts.stack().sort_values(ascending=False)
    (best_r, best_c), best_n = next(iter(stacked.items()))
    best_share = 100.0 * best_n / n_grid
    top3_share = 100.0 * stacked.head(3).sum() / n_grid
    bottom_share = 100.0 * counts.loc["bottom"].sum() / n_grid
    side_tot = counts.sum(axis=0)
    best_side = side_tot.idxmax()
    best_side_share = 100.0 * side_tot.max() / n_grid
    best_scored = 100.0 * goals.at[best_r, best_c] / best_n
    min_n = max(5, int(0.05 * n_grid))
    scored_by_zone = {
        z: 100.0 * goals.at[z[0], z[1]] / n for z, n in stacked.items() if n >= min_n
    }
    worst_zone = max(scored_by_zone, key=scored_by_zone.get) if scored_by_zone else None
    print("\nRECOMMENDATION FOR THE GOALKEEPER")
    print(
        f"  * Most frequent zone: {best_r.upper()}-{best_c.upper()} "
        f"({best_share:.1f}% of on-target shots, n={int(best_n)})"
    )
    print(
        f"  * The top 3 zones cover {top3_share:.0f}% of all on-target shots: "
        f"commit early to one of them"
    )
    print(
        f"  * {bottom_share:.0f}% of on-target penalties finish in the BOTTOM band: "
        f"dive low, arms down"
    )
    print(
        f"  * Side split (your view): left {100.0 * side_tot['keeper-left'] / n_grid:.0f}% / "
        f"centre {100.0 * side_tot['centre'] / n_grid:.0f}% / "
        f"right {100.0 * side_tot['keeper-right'] / n_grid:.0f}% -- "
        f"busiest side is {best_side} ({best_side_share:.1f}%)"
    )
    print(
        f"  * Shots placed {best_r}-{best_c} were scored {best_scored:.0f}% of the time: "
        f"keepers are rarely there, so that dive pays"
    )
    if worst_zone:
        print(
            f"  * Highest-conversion zone: {worst_zone[0]}-{worst_zone[1]} "
            f"({scored_by_zone[worst_zone]:.0f}% scored) -- the biggest blind spot"
        )
    print(
        f"  * {100.0 * n_off / total:.0f}% of all penalties missed the frame entirely: "
        f"staying on your feet already covers those"
    )


def print_report(df, meta):
    total = len(df)
    grid = df[df["zone_row"].notna() & df["zone_col"].notna()]
    n_grid = len(grid)
    n_on = int(df["on_target"].astype(bool).sum())
    off = df[df["status"] == "off target"]
    unk_h = df[df["status"] == "on target (height unknown)"]
    nodata = df[df["status"] == "no end location"]
    n_reg = int((~df["shootout"].astype(bool)).sum())
    n_so = int(df["shootout"].astype(bool).sum())
    bar = "=" * 78
    print("\n" + bar)
    print("PENALTY TARGET REPORT  (free StatsBomb open data, via statsbombpy)")
    print(bar)
    print(
        f"Matches scanned : {meta.get('matches_seen', '?')}   "
        f"(failed event downloads: {meta.get('failed', 0)}, "
        f"failed seasons: {len(meta.get('seasons_failed', []))})"
    )
    print(f"Penalties       : {total}   (regular: {n_reg}, shoot-out: {n_so})")
    if nodata:
        print(f"  - {len(nodata)} without usable end location (excluded)")
    if unk_h:
        print(f"  - {len(unk_h)} on target but without height (excluded from grid)")
    print(f"On target       : {n_on} ({100.0 * n_on / total:.1f}% of all pens)")
    print(f"Off target      : {len(off)} ({100.0 * len(off) / total:.1f}% of all pens)")
    if len(off):
        wl = int((off["off_side"] == "keeper-left").sum())
        wr = int((off["off_side"] == "keeper-right").sum())
        ov = int(off["off_over"].astype(bool).sum())
        print(
            f"  - wide of your left post: {wl}   wide of your right post: {wr}   "
            f"over the bar: {ov}"
        )
    if n_grid == 0:
        print("\nNo on-target penalties with 3D end locations found -- nothing to grid.")
        return
    counts = zone_counts(grid)
    goals = zone_counts(grid[grid["goal"].astype(bool)])
    saves = zone_counts(grid[grid["saved"].astype(bool)])
    print("\n3x3 TARGET GRID -- GOALKEEPER'S VIEW")
    print("(keeper-left/right = YOUR left/right as you face the taker)")
    print(f"(shares are % of the {n_grid} on-target penalties that fit the frame)")
    print_grid(counts)
    side_tot = counts.sum(axis=0)
    band_tot = counts.sum(axis=1)
    print(
        "  side split  : "
        + "  ".join(f"{c} {100.0 * side_tot[c] / n_grid:.1f}%" for c in COL_NAMES)
    )
    print(
        "  height split: "
        + "  ".join(f"{r} {100.0 * band_tot[r] / n_grid:.1f}%" for r in ROW_NAMES)
    )
    print_zone_table(counts, goals, saves, n_grid, total)
    reg = zone_counts(grid[~grid["shootout"].astype(bool)])
    so = zone_counts(grid[grid["shootout"].astype(bool)])
    print_shootout_split(reg, so)
    print_recommendation(counts, goals, n_grid, total, len(off))


def save_outputs(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shots_path = DATA_DIR / "penalty_shots.csv"
    df.to_csv(shots_path, index=False)
    grid = df[df["zone_row"].notna() & df["zone_col"].notna()]
    counts = zone_counts(grid)
    goals = zone_counts(grid[grid["goal"].astype(bool)])
    saves = zone_counts(grid[grid["saved"].astype(bool)])
    reg = zone_counts(grid[~grid["shootout"].astype(bool)])
    so = zone_counts(grid[grid["shootout"].astype(bool)])
    n_grid = len(grid)
    total = len(df)
    rows = []
    for r in ROW_NAMES:
        for c in COL_NAMES:
            n = int(counts.at[r, c])
            g = int(goals.at[r, c])
            s = int(saves.at[r, c])
            rows.append(
                {
                    "zone": f"{r}-{c}",
                    "n": n,
                    "share_on_target_%": round(100.0 * n / n_grid, 2) if n_grid else 0.0,
                    "share_all_penalties_%": round(100.0 * n / total, 2),
                    "goals": g,
                    "goal_rate_%": round(100.0 * g / n, 2) if n else None,
                    "saves": s,
                    "save_rate_%": round(100.0 * s / n, 2) if n else None,
                    "regular_n": int(reg.at[r, c]),
                    "shootout_n": int(so.at[r, c]),
                }
            )
    zone_path = DATA_DIR / "penalty_zone_summary.csv"
    pd.DataFrame(rows).to_csv(zone_path, index=False)
    print(f"\nSaved: {shots_path}")
    print(f"Saved: {zone_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Penalty target analysis (3x3 goal grid) on the free StatsBomb "
        "open data available via statsbombpy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --list\n"
            "  %(prog)s\n"
            '  %(prog)s --competitions "FIFA World Cup" "UEFA Euro"\n'
            "  %(prog)s --all --gender male\n"
            "  %(prog)s --limit-matches 25\n"
        ),
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        metavar="NAME",
        default=None,
        help="competition names to include (substring, case/accent-insensitive); "
        f"default: {' / '.join(DEFAULT_COMPETITIONS)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan ALL free competitions (overrides --competitions; can take hours)",
    )
    parser.add_argument(
        "--gender",
        choices=("male", "female", "both"),
        default="both",
        help="filter competitions by gender",
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="stop after N matches (smoke test; disables the cache)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even if cached data exists",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write the CSV outputs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the free competitions available via statsbombpy and exit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comps = sb.competitions()
    if args.list:
        list_competitions(comps)
        return 0
    comp_df = choose_competitions(comps, args)
    if comp_df.empty:
        print("No competition matches your selection. Run with --list to see options.")
        return 1
    print("Analysing penalties from:")
    for line in format_scope(comp_df):
        print("  " + line)
    if args.limit_matches is not None:
        print(f"(smoke test: only the first {args.limit_matches} matches will be scanned)")
    df, meta = load_records(comp_df, args)
    if df.empty:
        print("No penalties found in this scope.")
        return 0
    print_report(df, meta)
    if not args.no_save:
        save_outputs(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
