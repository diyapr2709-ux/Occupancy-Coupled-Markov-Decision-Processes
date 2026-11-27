"""The episode count of Claim A's held-out test, from the dev map's paired spreads alone. Prints n and nothing else.

    python experiments/claim_a_episode_count.py runs/phase/phase_<dev map>.json [--expect-sha256 <hash of that file>]

The rule, fixed in arena/2026-11-27-claim-a-rule-outcome.md before any held-out data existed:
  SD* is the largest sample standard deviation (ddof 1), over the dev map's cells, of the
  per-episode contrast planned_once_minus_greedy; cells whose contrast is identically zero
  carry no spread and cannot raise it.
  tau = 0.005 in expected failure rate per decision: half the margin m = 0.01 that
  arena/2026-11-24-design-outcome.md committed, so a cell contrast of size m has its sign
  resolved with probability about 0.975 under a normal approximation.
  n is the smallest count in [FLOOR, CAP] with t_{0.975, n-1} SD* / sqrt(n) <= tau, the floor
  keeping the test at least as precise as the map the null is built from, and the cap set by
  the ten-hour budget on the claim platform at twenty workers. If no count up to the cap
  meets tau, n is the cap.
The script refuses a map in which a (delay, lam) line has a contrast mean of exactly zero at
every sigma: the theory-free null would then carry a line with no sign, which the scorer
refuses, and the preregistration must go back before any held-out data exist. It prints no
spread, no mean and no crossing, so running it does not expose a dev-map value.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from score_phase_predictions import measured_lines
from stats import t_quantile

CONTRAST = "planned_once_minus_greedy"
TAU = 0.005
FLOOR, CAP = 40, 100


def episode_count(report: dict) -> int:
    lines, _ = measured_lines([report], CONTRAST)
    spreads = []
    for key, (_, values) in sorted(lines.items()):
        means = values.mean(axis=1)
        if np.all(means == 0.0):
            raise ValueError(f"line (kappa, delta, horizon, delay, lam) = {key} has a zero contrast at every sigma; the null cannot carry it")
        spreads.extend(float(np.std(row, ddof=1)) for row in values if np.any(row != row[0]))
    if not spreads:
        raise ValueError("no cell of the map has any spread in the contrast")
    worst = max(spreads)
    for n in range(FLOOR, CAP + 1):
        if t_quantile(0.975, n - 1) * worst / math.sqrt(n) <= TAU:
            return n
    return CAP


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_report", type=Path)
    parser.add_argument("--expect-sha256", help="refuse a map whose bytes do not hash to this")
    args = parser.parse_args()
    data = args.dev_report.read_bytes()
    if args.expect_sha256 and hashlib.sha256(data).hexdigest() != args.expect_sha256.lower():
        parser.error(f"{args.dev_report.as_posix()} does not have the expected SHA-256")
    try:
        print(episode_count(json.loads(data.decode("utf-8"))))
    except (KeyError, ValueError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
