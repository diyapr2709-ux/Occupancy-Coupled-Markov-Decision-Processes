"""Heuristic comparators on the toy environments, averaged over the fixed seed protocol.

    python experiments/run_heuristics.py

On PairedDrugs, within-class mixing has the same action entropy as cross-class mixing
but zero pushforward entropy, and fails like the fixed guideline. That is the second
claim of the proposal, visible before any learning.
"""

import argparse

import gymnasium as gym
import numpy as np

import ocmdp.envs  # noqa: F401  registers the Gymnasium ids
from ocmdp.baselines.heuristics import Cycling, FixedGuideline, StationaryMixing, uniform_mixing
from ocmdp.eval.metrics import summarize
from ocmdp.eval.rollout import rollout
from ocmdp.eval.seeds import SEEDS
from ocmdp.theory.two_drug import solve_two_drug

POLICIES = {
    "ocmdp/TwoDrug-v0": [FixedGuideline(2, 0), Cycling(2, period=4), uniform_mixing(2)],
    "ocmdp/PairedDrugs-v0": [
        FixedGuideline(4, 0),
        StationaryMixing([0.5, 0.5, 0.0, 0.0], name="within-class"),
        StationaryMixing([0.5, 0.0, 0.5, 0.0], name="cross-class"),
        Cycling(4, period=4),
        uniform_mixing(4),
    ],
}
COLUMNS = ["failure_rate", "first_fixation_q", "efficacy_reserve", "H_action", "H_pushforward", "violation_rate"]
# Between the default toy's equilibria under uniform mixing (0.5) and a pure policy (0.67),
# so the thresholds separate the two regimes instead of sitting on one of them.
FIXATION_THRESHOLD = 0.6
EFFICACY_THRESHOLD = 0.6


def _cell(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "never"
    cell = f"{finite.mean():.3f} ± {finite.std():.3f}"
    return cell if finite.size == values.size else f"{cell} ({values.size - finite.size} never)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()

    for env_id, policies in POLICIES.items():
        env = gym.make(env_id)
        assignment = env.unwrapped.coupling.mechanism_assignment()
        print(f"\n{env_id}  ({len(args.seeds)} seeds, {env.unwrapped.config.horizon_quarters} quarters)")
        print(f"{'policy':<16}" + "".join(f"{c:>18}" for c in COLUMNS))
        for policy in policies:
            rows = [
                summarize(rollout(env, policy, seed), assignment, FIXATION_THRESHOLD, EFFICACY_THRESHOLD)
                for seed in args.seeds
            ]
            cells = [_cell(np.array([row[c] for row in rows], dtype=float)) for c in COLUMNS]
            print(f"{policy.name:<16}" + "".join(f"{cell:>18}" for cell in cells))

    sol = solve_two_drug()
    print(
        f"\nMean-field two-drug result: best stationary p = {sol.p_opt:.2f}, F = {sol.f_opt:.3f}; "
        f"best deterministic F = {sol.f_deterministic:.3f} (gap {sol.gap:.3f})"
    )


if __name__ == "__main__":
    main()
