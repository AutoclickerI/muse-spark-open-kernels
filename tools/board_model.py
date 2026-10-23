#!/usr/bin/env python3
"""Fit-free overall-board model for a saved OKC leaderboard snapshot.

The service exposes only overall ranks.  This module therefore keeps the
snapshot-fitted portfolio score in one place and makes the assumptions visible:
the four decode-step boards are used, prefill is ignored, and a missing board is
the SGLang baseline (1.0x).  The score is the better of a whole-layer decode
submission and the fitted component portfolio.

Examples:
    python tools/board_model.py runs/boards/20261023T182817Z
    python tools/board_model.py runs/boards/20261023T182817Z \
        --speedup muse_decode_layer=1.52 --speedup muse_moe_w4a16=2.35
    python tools/board_model.py runs/boards/20261023T182817Z --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


DECODE = "muse_decode_layer"
COMPONENTS = ("muse_moe_w4a16", "muse_bf16_proj", "muse_attention_decode")
PROBLEMS = (DECODE, *COMPONENTS)
# Calibrated against the saved public snapshot.  The max() captures the board's
# observed behavior: a whole-layer run sets the score floor, while component
# runs can improve a weaker whole-layer run.
COMPOSE_OFFSET = 0.351
COMPOSE_WEIGHTS = {
    "muse_moe_w4a16": 0.312,
    "muse_bf16_proj": 0.268,
    "muse_attention_decode": 0.105,
}
MISSING_SPEEDUP = 1.0


def _read_json(path: Path) -> dict:
    with path.open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_snapshot(snapshot: str | Path) -> dict:
    """Load the overall and four decode-step boards from a snapshot directory."""
    root = Path(snapshot)
    if root.is_file():
        root = root.parent
    boards = {problem: _read_json(root / f"{problem}.json") for problem in ("overall", *PROBLEMS)}
    return {"root": root, "boards": boards}


def board_speedups(snapshot: Mapping, problem: str) -> dict[str, float]:
    """Return each visible team's best speedup, excluding the SGLang reference."""
    rows = snapshot["boards"][problem].get("rows", [])
    result: dict[str, float] = {}
    for row in rows:
        team = row.get("team")
        speedup = row.get("speedup")
        if team and team != "sglang" and row.get("rank") is not None and isinstance(speedup, (int, float)):
            result[str(team)] = float(speedup)
    return result


def observed_speedups(snapshot: Mapping, team: str = "AutoclickerI") -> dict[str, float]:
    """Read one team's current speedups; absent boards are 1.0x."""
    return {
        problem: board_speedups(snapshot, problem).get(team, MISSING_SPEEDUP)
        for problem in PROBLEMS
    }


def score(speedups: Mapping[str, float]) -> float:
    """Compute the fitted portfolio score (higher is better)."""
    direct = float(speedups.get(DECODE, MISSING_SPEEDUP))
    component = COMPOSE_OFFSET + sum(
        COMPOSE_WEIGHTS[problem] * float(speedups.get(problem, MISSING_SPEEDUP))
        for problem in COMPONENTS
    )
    return max(direct, component)


def _all_teams(snapshot: Mapping) -> list[str]:
    return [str(row["team"]) for row in snapshot["boards"]["overall"].get("rows", []) if row.get("team")]


def _team_speedups(snapshot: Mapping, team: str) -> dict[str, float]:
    return {
        problem: board_speedups(snapshot, problem).get(team, MISSING_SPEEDUP)
        for problem in PROBLEMS
    }


def ranked_scores(snapshot: Mapping, hypothetical: Mapping[str, float] | None = None,
                  team: str = "AutoclickerI") -> list[dict]:
    """Return all teams sorted by model score, replacing ``team`` if requested."""
    rows = []
    for name in _all_teams(snapshot):
        speeds = _team_speedups(snapshot, name)
        if name == team and hypothetical is not None:
            speeds.update({k: float(v) for k, v in hypothetical.items()})
        rows.append({"team": name, "speedups": speeds, "score": score(speeds)})
    # Team is a deterministic tie-break.  There are no score ties among the
    # current overall rows, but retaining a tie-break makes hypothetical output
    # stable instead of depending on JSON order.
    rows.sort(key=lambda row: (-row["score"], row["team"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def predict_rank(snapshot: Mapping, hypothetical: Mapping[str, float] | None = None,
                 team: str = "AutoclickerI") -> int:
    """Predict ``team``'s rank after applying hypothetical speedups."""
    rows = ranked_scores(snapshot, hypothetical, team)
    for row in rows:
        if row["team"] == team:
            return int(row["rank"])
    # A hypothetical team need not be in the saved board.  Insert it against
    # the fixed competitors in that case.
    speeds = dict(hypothetical or observed_speedups(snapshot, team))
    candidate = score(speeds)
    return 1 + sum(score(_team_speedups(snapshot, name)) > candidate for name in _all_teams(snapshot))


def _observed_rank(snapshot: Mapping, team: str) -> int | None:
    for row in snapshot["boards"]["overall"].get("rows", []):
        if row.get("team") == team and row.get("rank") is not None:
            return int(row["rank"])
    return None


def marginal_table(snapshot: Mapping, team: str = "AutoclickerI", step: float = 0.01) -> list[dict]:
    base = observed_speedups(snapshot, team)
    base_rank = predict_rank(snapshot, base, team)
    table = []
    for problem in PROBLEMS:
        trial = dict(base)
        trial[problem] += step
        trial_rank = predict_rank(snapshot, trial, team)
        table.append({
            "problem": problem,
            "speedup": base[problem],
            "plus": step,
            "predicted_rank": trial_rank,
            "rank_change": base_rank - trial_rank,
        })
    return table


def _competitor_threshold(snapshot: Mapping, team: str, target: int) -> float:
    competitors = [
        score(_team_speedups(snapshot, name))
        for name in _all_teams(snapshot)
        if name != team
    ]
    competitors.sort(reverse=True)
    if target < 1:
        raise ValueError("target rank must be positive")
    if target > len(competitors):
        return competitors[-1] if competitors else 0.0
    return competitors[target - 1]


def cheapest_path(snapshot: Mapping, team: str = "AutoclickerI", target: int = 50) -> dict:
    """Find the least total x-gain under the fitted max-of-two score model.

    Since the two branches are a max, an L1-minimal path uses either decode
    alone or the component with the largest composition coefficient alone.
    """
    current = observed_speedups(snapshot, team)
    threshold = _competitor_threshold(snapshot, team, target)
    eps = 1e-9
    direct_gain = max(0.0, threshold + eps - current[DECODE])
    best_component = max(COMPOSE_WEIGHTS, key=COMPOSE_WEIGHTS.get)
    component_score = score({**current, DECODE: MISSING_SPEEDUP})
    component_gain = max(0.0, (threshold + eps - component_score) / COMPOSE_WEIGHTS[best_component])
    if direct_gain <= component_gain:
        gains = {problem: 0.0 for problem in PROBLEMS}
        gains[DECODE] = direct_gain
        branch = "decode"
    else:
        gains = {problem: 0.0 for problem in PROBLEMS}
        gains[best_component] = component_gain
        branch = best_component
    speeds = {problem: current[problem] + gains[problem] for problem in PROBLEMS}
    return {
        "target_rank": target,
        "threshold_score": threshold,
        "branch": branch,
        "speedups": speeds,
        "gains": gains,
        "total_gain": sum(gains.values()),
        "predicted_rank": predict_rank(snapshot, speeds, team),
    }


def fit_report(snapshot: Mapping) -> dict:
    """Compare model ranks with the saved overall ranks."""
    actual = {
        str(row["team"]): int(row["rank"])
        for row in snapshot["boards"]["overall"].get("rows", [])
        if row.get("team") and row.get("rank") is not None
    }
    predicted_rows = ranked_scores(snapshot)
    predicted = {row["team"]: row["rank"] for row in predicted_rows}
    teams = [team for team in actual if team in predicted]
    exact = sum(predicted[t] == actual[t] for t in teams)
    mean_abs = sum(abs(predicted[t] - actual[t]) for t in teams) / len(teams)
    # Kendall tau-b without a dependency: ranks are unique in the API snapshot
    # and our tie-break makes predicted ranks unique too.
    ordered = sorted(teams, key=actual.__getitem__)
    inversions = sum(predicted[ordered[i]] > predicted[ordered[j]]
                     for i in range(len(ordered)) for j in range(i + 1, len(ordered)))
    n = len(ordered)
    tau = 1.0 - 4.0 * inversions / (n * (n - 1)) if n > 1 else 1.0
    top = [t for t in teams if actual[t] <= 300]
    top_ordered = sorted(top, key=actual.__getitem__)
    top_inv = sum(predicted[top_ordered[i]] > predicted[top_ordered[j]]
                  for i in range(len(top_ordered)) for j in range(i + 1, len(top_ordered)))
    nt = len(top_ordered)
    top_tau = 1.0 - 4.0 * top_inv / (nt * (nt - 1)) if nt > 1 else 1.0
    top50_overlap = sum(actual[t] <= 50 and predicted[t] <= 50 for t in teams)
    top50_exact = sum(actual[t] <= 50 and actual[t] == predicted[t] for t in teams)
    return {
        "teams": len(teams),
        "exact_matches": exact,
        "mean_abs_rank_error": mean_abs,
        "kendall_tau": tau,
        "top300_kendall_tau": top_tau,
        "top50_overlap": top50_overlap,
        "top50_exact_ranks": top50_exact,
    }


def _parse_speedups(values: Sequence[str]) -> dict[str, float]:
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"speedup must be problem=value, got {item!r}")
        problem, value = item.split("=", 1)
        if problem not in PROBLEMS:
            raise ValueError(f"unknown decode-step problem {problem!r}")
        result[problem] = float(value)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--team", default="AutoclickerI")
    parser.add_argument("--speedup", action="append", default=[], metavar="PROBLEM=VALUE",
                        help="override one speedup; repeat for multiple problems")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    snapshot = load_snapshot(args.snapshot)
    overrides = _parse_speedups(args.speedup)
    base = observed_speedups(snapshot, args.team)
    base.update(overrides)
    result = {
        "snapshot": str(snapshot["root"]),
        "formula": {
            "score": "max(decode, offset + w_moe*moe + w_proj*proj + w_attention*attention)",
            "offset": COMPOSE_OFFSET,
            "weights": COMPOSE_WEIGHTS,
            "missing_speedup": MISSING_SPEEDUP,
            "prefill": "ignored",
        },
        "team": args.team,
        "speedups": base,
        "score": score(base),
        "predicted_rank": predict_rank(snapshot, base, args.team),
        "observed_rank": _observed_rank(snapshot, args.team),
        "fit": fit_report(snapshot),
        "marginal_plus_0.01": marginal_table(snapshot, args.team),
        "cheapest_path": cheapest_path(snapshot, args.team, args.target),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{args.team}: model score {result['score']:.6f}, predicted rank {result['predicted_rank']}"
              f" (API rank {result['observed_rank']})")
        fit = result["fit"]
        print(f"fit: tau={fit['kendall_tau']:.4f}, top300 tau={fit['top300_kendall_tau']:.4f}, "
              f"top50={fit['top50_exact_ranks']}/{50} exact")
        print("marginal +0.01x:")
        for row in result["marginal_plus_0.01"]:
            print(f"  {row['problem']}: rank {row['predicted_rank']} ({row['rank_change']:+d})")
        path = result["cheapest_path"]
        print(f"cheapest rank {args.target}: {path['branch']} total gain {path['total_gain']:.4f}x, "
              f"predicted rank {path['predicted_rank']}")
        print("  " + ", ".join(f"{p}={v:.4f}" for p, v in path["speedups"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
