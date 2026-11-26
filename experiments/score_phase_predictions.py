"""Score predicted sign changes of a phase-diagram contrast against measured ones.

    python experiments/score_phase_predictions.py predictions.json runs/phase/phase_<a>.json runs/phase/phase_<b>.json ...
    python experiments/score_phase_predictions.py predictions.json <reports> --null null_predictions.json

Each report fixes (kappa, delta, horizon) and holds lines: one per (delay, lam), with a
sigma grid. Along a line the contrast (a policy pair from sweep_phase_diagram.CONTRASTS,
named in the predictions file) is averaged over episodes per sigma. The measured
crossing sigma_hat is where it takes, for good, the sign stated for the whole contrast
as its sign beyond the crossing, linearly interpolated. The error of a line is
|log(sigma_hat / sigma_pred)|, with measured and predicted crossings alike censored to
the part of the grid that can resolve them: a line that shows that sign at every sigma
crossed below the grid (placed at the smallest positive sigma), one that does not end in
it crosses above it (placed at the largest), and numeric crossings outside that range
are clamped to it. Two crossings censored on the same side agree. The statistic is E,
the median error over every measured line: a line the predictions leave out has infinite
error and is listed under not_predicted, so leaving lines out cannot improve E. Every
report must share its evaluation episodes; a line's sigma points may be split across
reports, but no sigma may be measured twice for one line.

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
    """Line -> (sigma grid, per-episode contrast of shape (sigmas, episodes)), and the common episode count.

    A line's sigma points may come from several reports (a sweep split by sigma); they are
    merged, and a sigma measured twice for the same line is refused rather than overwritten.
    """
    minuend, subtrahend = CONTRASTS[contrast]
    points: dict[Line, dict[float, np.ndarray]] = {}
    episodes = None
    seeds = {json.dumps(report.get("evaluation_episode_seeds")) for report in reports}
    if len(seeds) > 1:
        raise ValueError("every report must use the same evaluation episodes (evaluation_episode_seeds differ)")
    for report in reports:
        inst = report["instance"]
        raw = {(r["delay"], r["lam"], r["sigma"], r["policy"]): np.asarray(r["expected_failure_rate"]) for r in report["results"]}
        for delay, lam, sigma in sorted({(r["delay"], r["lam"], r["sigma"]) for r in report["results"]}):
            value = raw[(delay, lam, sigma, minuend)] - raw[(delay, lam, sigma, subtrahend)]
            if episodes is None:
                episodes = value.shape[0]
            if value.shape[0] != episodes:
                raise ValueError("every report must use the same evaluation episodes")
            line = points.setdefault((inst["kappa"], inst["delta"], inst["horizon_quarters"], delay, lam), {})
            if sigma in line:
                raise ValueError(f"sigma {sigma} is measured twice for line (kappa, delta, horizon, delay, lam) = "
                                 f"{(inst['kappa'], inst['delta'], inst['horizon_quarters'], delay, lam)}")  # fmt: skip
            line[sigma] = value
    lines = {key: (np.array(sorted(by_sigma)), np.stack([by_sigma[s] for s in sorted(by_sigma)])) for key, by_sigma in points.items()}
    return lines, episodes


def crossing(sigmas: np.ndarray, means: np.ndarray, beyond: int) -> tuple[float | None, int]:
    """Where the contrast takes the sign `beyond` for good, and the sign of its last nonzero mean.

    The crossing is the last change into `beyond` along sigmas, after which the contrast
    keeps that sign, so a line that dips and recovers is placed where it settles. Between
    the two nonzero means around it the crossing is an exact zero lying between them if
    there is one, else the linear interpolation. None when every nonzero mean already has
    the sign `beyond` (crossed below the grid) or the last one does not (not yet crossed);
    a line with no nonzero mean has sign 0 and so never shows `beyond`.
    """
    nonzero = [(s, m) for s, m in zip(sigmas, means) if m != 0]
    if not nonzero:
        return None, 0
    last_sign = int(np.sign(nonzero[-1][1]))
    before = [i for i, (_, m) in enumerate(nonzero) if np.sign(m) != beyond]
    if last_sign != beyond or not before:
        return None, last_sign
    (s0, m0), (s1, m1) = nonzero[before[-1]], nonzero[before[-1] + 1]
    zeros = [s for s, m in zip(sigmas, means) if m == 0 and s0 < s < s1]
    return float(zeros[0] if zeros else s0 + (s1 - s0) * m0 / (m0 - m1)), last_sign


def line_error(sigmas: np.ndarray, measured: tuple[float | None, int], predicted: tuple[float | None, int], beyond: int) -> float:
    """|log ratio| of measured to predicted crossing, both censored to what the grid can resolve.

    A crossing at or below the smallest positive sigma cannot be told from one below the
    grid, nor one at or above the largest sigma from one beyond it, so measured and
    predicted crossings alike are clamped to [smallest positive sigma, largest sigma]. A
    line with no crossing sits at the low end if it already shows the sign `beyond` past
    the crossing, and at the high end if it does not yet.
    """
    low, high = float(sigmas[sigmas > 0].min()), float(sigmas.max())

    def place(crossing_at: float | None, sign_at_max: int) -> float:
        if crossing_at is None:
            return low if sign_at_max == beyond else high
        return min(max(crossing_at, low), high)

    return abs(math.log(place(*measured) / place(*predicted)))


def _error(lines: dict, means: dict[Line, np.ndarray], predictions: dict, key: Line, beyond: int) -> float:
    """A line's error; a measured line with no prediction has infinite error, so leaving a line out never helps."""
    if key not in predictions:
        return math.inf
    return line_error(lines[key][0], crossing(lines[key][0], means[key], beyond), predictions[key], beyond)


def median_error(lines: dict, means: dict[Line, np.ndarray], predictions: dict[Line, tuple[float | None, int]], beyond: int) -> float:
    """Median error over every measured line, predicted or not."""
    return float(np.median([_error(lines, means, predictions, k, beyond) for k in sorted(lines)]))


def load_predictions(path: Path, lines: dict) -> tuple[str, int, dict[Line, tuple[float | None, int]], list[dict]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
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


def _percentiles(values: list[float], q: list[float]) -> list[float]:
    """Linear percentiles that stay infinite where any weight falls on an infinite value (numpy returns nan there)."""
    huge = 1e300
    finite = np.clip(np.asarray(values, dtype=float), -huge, huge)
    return [math.copysign(math.inf, v) if abs(v) >= huge / 10 else float(v) for v in np.percentile(finite, q)]


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
            for key, value in _direction_shifts(lines, means, direction, beyond).items():
                boot_dir.setdefault((d_index, key), []).append(value)
    e = median_error(lines, point, predictions, beyond)
    lo, hi = _percentiles(boot_e, [2.5, 97.5])
    out = {
        "lines": [
            {"line": dict(zip(("kappa", "delta", "horizon", "delay", "lam"), k)), "measured_crossing": crossing(lines[k][0], point[k], beyond)[0],
             "predicted_crossing": predictions[k][0], "error": _error(lines, point, predictions, k, beyond)}
            for k in sorted(lines) if k in predictions
        ],  # fmt: skip
        "not_predicted": [dict(zip(("kappa", "delta", "horizon", "delay", "lam"), k)) for k in sorted(lines) if k not in predictions],
        "E": e,
        "E_ci95": [float(lo), float(hi)],
        "margin": margin,
        "verdict": "pass" if hi < margin else "fail" if lo > margin else "inconclusive",
    }
    if null:
        diff = np.asarray(boot_e) - np.asarray(boot_null)
        out["E_null"] = median_error(lines, point, null, beyond)
        out["E_minus_E_null_ci95"] = _percentiles(diff, [2.5, 97.5])
    out["directions"] = []
    for d_index, direction in enumerate(directions):
        for key, value in _direction_shifts(lines, point, direction, beyond).items():
            samples = boot_dir.get((d_index, key), [])
            interval = [float(v) for v in np.percentile(samples, [2.5, 97.5])] if len(samples) == resamples else None
            wrong = interval is not None and ((direction["sign"] > 0 and interval[1] < 0) or (direction["sign"] < 0 and interval[0] > 0))
            out["directions"].append({"from": direction["from"], "to": direction["to"], "delay": key[0], "lam": key[1],
                                      "log_shift": value, "ci95": interval, "predicted_sign": direction["sign"],
                                      "wrong_sign": wrong, "censored": interval is None})  # fmt: skip
    return out


def _direction_shifts(lines: dict, means: dict, direction: dict, beyond: int) -> dict[tuple[int, float], float]:
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
        x, y = crossing(lines[key][0], means[key], beyond)[0], crossing(lines[other][0], means[other], beyond)[0]
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
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    try:
        contrast = json.loads(args.predictions.read_text(encoding="utf-8"))["contrast"]
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
