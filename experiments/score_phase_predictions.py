"""Score predicted sign changes of a phase-diagram contrast against measured ones.

    python experiments/score_phase_predictions.py predictions.json runs/phase/phase_<a>.json runs/phase/phase_<b>.json ...
    python experiments/score_phase_predictions.py predictions.json <reports> --null null_predictions.json

Each report fixes (kappa, delta, horizon) and holds lines: one per (delay, lam), with a
sigma grid. Along a line the contrast (a policy pair from sweep_phase_diagram.CONTRASTS,
named in the predictions file) is averaged over episodes per sigma, and its first sign
change, linearly interpolated, is the measured crossing sigma_hat. The error of a line
is |log(sigma_hat / sigma_pred)|, with crossings outside the grid censored at its ends.
Which end is set by the contrast's sign beyond its crossing, stated once for the whole
contrast: a line that shows that sign at every sigma crossed below the grid (placed at
the smallest positive sigma), one that never shows it crosses above it (placed at the
largest). Two predictions or measurements outside the grid on the same side agree. The
statistic is E, the median error over lines.

Uncertainty: episodes are resampled jointly across every cell of every line, which keeps
the pairing of policies within a cell and of cells sharing an evaluation seed. E's 95%
percentile interval against the margin (default log sqrt 2) gives the verdict: pass when
its upper bound is below the margin, fail when its lower bound is above, otherwise
inconclusive. With --null, the same is computed for theory-free predictions on the same
resamples and the interval of E - E_null is reported. Direction checks compare log
sigma_hat between two settings of one (delay, lam) line.

Predictions file (JSON):
  {"contrast": "<name>", "sign_beyond_crossing": 1 | -1,
   "lines": [{"kappa", "delta", "horizon", "delay", "lam", "sigma_star": float | null, "sign_at_max_sigma": 1 | -1}, ...],
   "directions": [{"from": {"kappa", "delta", "horizon"}, "to": {...}, "sign": 1 | -1}, ...]}   (directions optional)
sigma_star null means no crossing inside the grid; the line keeps sign_at_max_sigma throughout.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from sweep_phase_diagram import CONTRASTS

Line = tuple[float, float, int, int, float]  # kappa, delta, horizon, delay, lam


def measured_lines(reports: list[dict], contrast: str) -> tuple[dict[Line, tuple[np.ndarray, np.ndarray]], int]:
    """Line -> (sigma grid, per-episode contrast of shape (sigmas, episodes)), and the common episode count."""
    minuend, subtrahend = CONTRASTS[contrast]
    lines, episodes = {}, None
    for report in reports:
        inst = report["instance"]
        raw = {(r["delay"], r["lam"], r["sigma"], r["policy"]): np.asarray(r["expected_failure_rate"]) for r in report["results"]}
        for delay, lam in sorted({(r["delay"], r["lam"]) for r in report["results"]}):
            sigmas = np.array(sorted({r["sigma"] for r in report["results"] if (r["delay"], r["lam"]) == (delay, lam)}))
            values = np.stack([raw[(delay, lam, s, minuend)] - raw[(delay, lam, s, subtrahend)] for s in sigmas])
            if episodes is None:
                episodes = values.shape[1]
            if values.shape[1] != episodes:
                raise ValueError("every report must use the same evaluation episodes")
            lines[(inst["kappa"], inst["delta"], inst["horizon_quarters"], delay, lam)] = (sigmas, values)
    return lines, episodes


def crossing(sigmas: np.ndarray, means: np.ndarray) -> tuple[float | None, int]:
    """First sign change of means along sigmas, and the last nonzero sign.

    Between consecutive nonzero means of opposite sign the crossing is an exact zero lying
    between them if there is one, else the linear interpolation. A crossing at or below
    zero is reported as None, i.e. below the grid.
    """
    nonzero = [(s, m) for s, m in zip(sigmas, means) if m != 0]
    last_sign = int(np.sign(nonzero[-1][1])) if nonzero else 1
    for (s0, m0), (s1, m1) in zip(nonzero, nonzero[1:]):
        if np.sign(m0) != np.sign(m1):
            zeros = [s for s, m in zip(sigmas, means) if m == 0 and s0 < s < s1]
            at = zeros[0] if zeros else s0 + (s1 - s0) * m0 / (m0 - m1)
            if at > 0:
                return float(at), last_sign
    return None, last_sign


def line_error(sigmas: np.ndarray, measured: tuple[float | None, int], predicted: tuple[float | None, int], beyond: int) -> float:
    """|log ratio| of measured to predicted crossing, censored at the grid ends; `beyond` is the sign past the crossing."""
    low, high = float(sigmas[sigmas > 0].min()), float(sigmas.max())

    def place(crossing_at: float | None, sign_at_max: int) -> float:
        if crossing_at is not None:
            return crossing_at
        return low if sign_at_max == beyond else high  # already past the crossing everywhere, or not yet anywhere

    return abs(math.log(place(*measured) / place(*predicted)))


def median_error(lines: dict, means: dict[Line, np.ndarray], predictions: dict[Line, tuple[float | None, int]], beyond: int) -> float:
    return float(np.median([line_error(lines[k][0], crossing(lines[k][0], means[k]), predictions[k], beyond) for k in predictions]))


def load_predictions(path: Path, lines: dict) -> tuple[str, int, dict[Line, tuple[float | None, int]], list[dict]]:
    spec = json.loads(path.read_text())
    contrast, beyond = spec["contrast"], spec["sign_beyond_crossing"]
    if contrast not in CONTRASTS:
        raise ValueError(f"unknown contrast {contrast!r}; known: {sorted(CONTRASTS)}")
    if beyond not in (1, -1):
        raise ValueError("sign_beyond_crossing must be 1 or -1")
    predictions = {}
    for row in spec["lines"]:
        key = (row["kappa"], row["delta"], row["horizon"], row["delay"], row["lam"])
        if row["sign_at_max_sigma"] not in (1, -1):
            raise ValueError(f"sign_at_max_sigma must be 1 or -1 for line {key}")
        predictions[key] = (row["sigma_star"], row["sign_at_max_sigma"])
    missing = sorted(set(predictions) - set(lines))
    if missing:
        raise ValueError(f"predicted lines with no measurement: {missing}")
    return contrast, beyond, predictions, spec.get("directions", [])


def score(lines: dict, episodes: int, predictions: dict, null: dict | None, directions: list[dict], beyond: int,
          margin: float, resamples: int, seed: int) -> dict:  # fmt: skip
    point = {k: lines[k][1].mean(axis=1) for k in lines}
    rng = np.random.default_rng(seed)
    draws = rng.integers(episodes, size=(resamples, episodes))
    boot_e, boot_null, boot_dir = [], [], {}
    for idx in draws:
        means = {k: lines[k][1][:, idx].mean(axis=1) for k in lines}
        boot_e.append(median_error(lines, means, predictions, beyond))
        if null:
            boot_null.append(median_error(lines, means, null, beyond))
        for d_index, direction in enumerate(directions):
            for key, value in _direction_shifts(lines, means, direction).items():
                boot_dir.setdefault((d_index, key), []).append(value)
    e = median_error(lines, point, predictions, beyond)
    lo, hi = np.percentile(boot_e, [2.5, 97.5])
    out = {
        "lines": [
            {"line": dict(zip(("kappa", "delta", "horizon", "delay", "lam"), k)), "measured_crossing": crossing(lines[k][0], point[k])[0],
             "predicted_crossing": predictions[k][0], "error": line_error(lines[k][0], crossing(lines[k][0], point[k]), predictions[k], beyond)}
            for k in predictions
        ],  # fmt: skip
        "E": e,
        "E_ci95": [float(lo), float(hi)],
        "margin": margin,
        "verdict": "pass" if hi < margin else "fail" if lo > margin else "inconclusive",
    }
    if null:
        diff = np.asarray(boot_e) - np.asarray(boot_null)
        out["E_null"] = median_error(lines, point, null, beyond)
        out["E_minus_E_null_ci95"] = [float(v) for v in np.percentile(diff, [2.5, 97.5])]
    out["directions"] = []
    for d_index, direction in enumerate(directions):
        for key, value in _direction_shifts(lines, point, direction).items():
            samples = boot_dir.get((d_index, key), [])
            interval = [float(v) for v in np.percentile(samples, [2.5, 97.5])] if len(samples) == resamples else None
            wrong = interval is not None and ((direction["sign"] > 0 and interval[1] < 0) or (direction["sign"] < 0 and interval[0] > 0))
            out["directions"].append({"from": direction["from"], "to": direction["to"], "delay": key[0], "lam": key[1],
                                      "log_shift": value, "ci95": interval, "predicted_sign": direction["sign"],
                                      "wrong_sign": wrong, "censored": interval is None})  # fmt: skip
    return out


def _direction_shifts(lines: dict, means: dict, direction: dict) -> dict[tuple[int, float], float]:
    """log sigma_hat(to) - log sigma_hat(from) for every (delay, lam) where both settings cross inside the grid."""
    a = (direction["from"]["kappa"], direction["from"]["delta"], direction["from"]["horizon"])
    b = (direction["to"]["kappa"], direction["to"]["delta"], direction["to"]["horizon"])
    shifts = {}
    for key in lines:
        if key[:3] != a:
            continue
        other = (*b, *key[3:])
        if other not in lines:
            continue
        x, y = crossing(lines[key][0], means[key])[0], crossing(lines[other][0], means[other])[0]
        if x is not None and y is not None:
            shifts[key[3:]] = math.log(y / x)
    return shifts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("reports", type=Path, nargs="+", help="runs/phase reports, one per (kappa, delta, horizon) setting")
    parser.add_argument("--null", type=Path, help="theory-free predictions in the same format")
    parser.add_argument("--margin", type=float, default=math.log(math.sqrt(2)))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    try:
        contrast = json.loads(args.predictions.read_text())["contrast"]
        lines, episodes = measured_lines(reports, contrast)
        _, beyond, predictions, directions = load_predictions(args.predictions, lines)
        null = load_predictions(args.null, lines)[2] if args.null else None
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    result = score(lines, episodes, predictions, null, directions, beyond, args.margin, args.resamples, args.seed)
    result.update(contrast=contrast, reports=[str(p) for p in args.reports], predictions=str(args.predictions))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
