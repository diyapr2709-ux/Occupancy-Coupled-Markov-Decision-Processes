"""The descriptive quantities Claim A's preregistration reports beside its verdict, from the scored files alone.

    python experiments/claim_a_descriptives.py runs/claim_a/<slug>/ runs/phase/<kappa report>.json runs/phase/<delta report>.json \\
        --dev-map runs/phase/phase_20261126_184038_27744.json

Reads the filtered predictions, the null and primary.json that `claim_a_gate.py score` wrote,
and the two held-out reports, and writes descriptives.json next to them. Nothing here carries
a verdict: every quantity is labelled descriptive, unadjusted, for these 40 lines and 160 cells.
  censoring: for the measured, theory and null crossings of each line, whether the scorer
    placed it below the grid (at the smallest positive sigma), inside, or above (at the
    largest sigma), with counts, and the number of lines whose error is 0 only because
    both crossings are censored on the same side, for the theory and for the null;
  cells: each cell's mean contrast with its exact Student-t 95% half-width and SD, labelled
    unadjusted (160 marginal intervals);
  delta: per line with a measured crossing inside the grid, the change of the mean contrast
    across the bracket that holds it, and Delta_min over those lines, the quantity the
    preregistration's power statement is written in;
  spread_transfer: the share of held-out cells whose paired SD exceeds the dev map's largest
    cell SD (assumption A1 of the preregistration), with that SD* beside it.
Run only after the verdict has been printed; it reads the held-out reports in full.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from score_phase_predictions import crossing, load_predictions, measured_lines
from stats import t_quantile

CONTRAST_SIDES = ("below", "inside", "above")


def _place(sigmas: np.ndarray, at: float | None, sign: int, beyond: int) -> tuple[float, str]:
    low, high = float(sigmas[sigmas > 0].min()), float(sigmas.max())
    if at is None:
        value = low if sign == beyond else high
    else:
        value = min(max(at, low), high)
    side = "below" if value <= low else "above" if value >= high else "inside"
    return value, side


def describe(reports: list[dict], theory: dict, null: dict, beyond: int, dev_map: dict | None) -> dict:
    contrast = theory["contrast"]
    lines, episodes = measured_lines(reports, contrast)
    rows_t = {(r["kappa"], r["delta"], r["horizon"], r["delay"], r["lam"]): (r["sigma_star"], r["sign_at_max_sigma"]) for r in theory["lines"]}
    rows_n = {(r["kappa"], r["delta"], r["horizon"], r["delay"], r["lam"]): (r["sigma_star"], r["sign_at_max_sigma"]) for r in null["lines"]}
    per_line, counts = [], {who: dict.fromkeys(CONTRAST_SIDES, 0) for who in ("measured", "theory", "null")}
    zero_by_censoring = {"theory": 0, "null": 0}
    deltas = []
    for key, (sigmas, values) in sorted(lines.items()):
        means = values.mean(axis=1)
        measured = crossing(sigmas, means, beyond)
        placed = {"measured": _place(sigmas, *measured, beyond)}
        for who, rows in (("theory", rows_t), ("null", rows_n)):
            placed[who] = _place(sigmas, *rows[key], beyond) if key in rows else (None, "not_predicted")
        for who, (_, side) in placed.items():
            if side in counts[who]:
                counts[who][side] += 1
        for who in ("theory", "null"):
            if placed[who][1] != "inside" and placed[who][1] == placed["measured"][1]:
                zero_by_censoring[who] += 1
        delta = None
        if placed["measured"][1] == "inside":
            nonzero = [(s, m) for s, m in zip(sigmas, means) if m != 0]
            before = [i for i, (_, m) in enumerate(nonzero) if np.sign(m) != beyond]
            m0, m1 = nonzero[before[-1]][1], nonzero[before[-1] + 1][1]
            delta = float(abs(m1 - m0))
            deltas.append(delta)
        per_line.append({"line": dict(zip(("kappa", "delta", "horizon", "delay", "lam"), key)),
                         **{f"{who}_placed": placed[who][0] for who in placed}, **{f"{who}_side": placed[who][1] for who in placed},
                         "bracket_delta": delta})  # fmt: skip
    cells = []
    for key, (sigmas, values) in sorted(lines.items()):
        for sigma, row in zip(sigmas, values):
            sd = float(np.std(row, ddof=1))
            cells.append({"line": dict(zip(("kappa", "delta", "horizon", "delay", "lam"), key)), "sigma": float(sigma),
                          "mean": float(row.mean()), "half_width_95": float(t_quantile(0.975, len(row) - 1) * sd / np.sqrt(len(row))), "sd": sd})  # fmt: skip
    out = {
        "label": "descriptive, unadjusted, for these 40 lines and 160 cells; no verdict",
        "episodes": episodes,
        "censoring": {"counts": counts, "lines_with_zero_error_by_same_side_censoring": zero_by_censoring,
                      "median_is_zero_by_construction": {who: n >= 21 for who, n in zero_by_censoring.items()}, "per_line": per_line},  # fmt: skip
        "delta": {"per_line": [row["bracket_delta"] for row in per_line], "lines_with_interior_crossing": len(deltas),
                  "delta_min": min(deltas) if deltas else None},  # fmt: skip
        "cells": cells,
    }
    if dev_map is not None:
        dev_lines, _ = measured_lines([dev_map], contrast)
        dev_sd = max(float(np.std(row, ddof=1)) for _, v in dev_lines.values() for row in v if np.any(row != row[0]))
        above = [c for c in cells if c["sd"] > dev_sd]
        out["spread_transfer"] = {"dev_map_largest_cell_sd": dev_sd, "held_out_cells_above_it": len(above), "share": len(above) / len(cells)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scored", type=Path, help="the runs/claim_a/<slug>/ folder claim_a_gate.py score wrote")
    parser.add_argument("reports", type=Path, nargs=2)
    parser.add_argument("--dev-map", type=Path, help="the pinned dev map, for the spread-transfer share")
    args = parser.parse_args()
    out = args.scored / "descriptives.json"
    if out.exists():
        parser.error(f"{out} exists; nothing under runs/ is rewritten")
    try:
        theory = json.loads((args.scored / "predictions_heldout.json").read_text(encoding="utf-8"))
        null = json.loads((args.scored / "null.json").read_text(encoding="utf-8"))
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
        lines, _ = measured_lines(reports, theory["contrast"])
        load_predictions(args.scored / "predictions_heldout.json", lines)  # the same shape checks the scorer made
        dev_map = json.loads(args.dev_map.read_text(encoding="utf-8")) if args.dev_map else None
        result = describe(reports, theory, null, theory["sign_beyond_crossing"], dev_map)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    c = result["censoring"]
    print(f"censoring counts {c['counts']}; zero-by-censoring {c['lines_with_zero_error_by_same_side_censoring']}; "
          f"lines with interior crossing {result['delta']['lines_with_interior_crossing']}; written {out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
