"""Restart agreement of the mean-field planner: how far its planned failure rate moves with the planner seed.

    python experiments/planner_restarts.py                                    # 5 mechanisms x 4 drugs
    python experiments/planner_restarts.py --mechanisms 2 --drugs-per-mechanism 2
    python experiments/planner_restarts.py --steps 800 --random-starts 16

The oracle's plan is the best iterate of a non-convex optimisation from a handful of
starts, so its value depends on the random ones. The spread of the planned failure rate
over planner seeds, as a share of the plan's gain over uniform mixing, is the error
carried by anything normalised by that gain. Planning only; nothing is trained. Writes
a JSON report to runs/planner/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_agent import git_state

from ocmdp.baselines.oracle import FullInformationOracle, best_rotation
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.theory.mean_field import MeanFieldModel


def planned_rates(env: ToyOCMDPEnv, steps: int, n_random: int, planner_seeds: int) -> list[float]:
    """Planned whole-episode failure rate from the nominal start, one per planner seed."""
    rates = []
    for seed in range(planner_seeds):
        oracle = FullInformationOracle(env, steps=steps, n_random=n_random, seed=seed)
        obs, info = env.reset(seed=0)
        oracle.probs(obs, info)
        rates.append(oracle.predicted_failure_rate)
    return rates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanisms", type=int, default=5)
    parser.add_argument("--drugs-per-mechanism", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200, help="planner iterations (the oracle's default is 200)")
    parser.add_argument("--random-starts", type=int, default=4, help="random planner starts (the oracle's default is 4)")
    parser.add_argument("--planner-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/planner"))
    args = parser.parse_args()
    if min(args.mechanisms, args.drugs_per_mechanism, args.steps, args.planner_seeds) < 1 or args.random_starts < 0:
        parser.error("counts must be positive (--random-starts may be zero)")
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)

    config = ToyConfig.paired(n_mechanisms=args.mechanisms, drugs_per_mechanism=args.drugs_per_mechanism)
    env = ToyOCMDPEnv(config)
    model = MeanFieldModel(config)
    n_drugs = int(env.action_space.n)
    uniform = model.simulate(torch.full((config.horizon_quarters, n_drugs), 1.0 / n_drugs))[0].mean().item()
    _, rotation, _ = best_rotation(model, model.initial_state(), env.coupling.identical_row_classes(), config.horizon_quarters)
    rates = np.array(planned_rates(env, args.steps, args.random_starts, args.planner_seeds))
    spread, gain = float(rates.max() - rates.min()), uniform - float(rates.min())

    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git": git_state(),
        "n_drugs": n_drugs,
        "uniform": uniform,
        "best_rotation": rotation,
        "planned_failure_rate": rates.tolist(),
        "spread": spread,
        "spread_share_of_gain_over_uniform": spread / gain,
    }
    path = args.out / f"planner_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"{n_drugs} drugs, {args.mechanisms} mechanisms; {args.steps} steps, {args.random_starts} random starts")
    print(f"  uniform mixing            {uniform:.5f}")
    print(f"  best rotation             {rotation:.5f}")
    print(f"  plan, by planner seed     {' '.join(f'{r:.5f}' for r in rates)}")
    print(f"  spread over planner seeds {spread:.2e}  ({spread / gain:.1%} of the gain over uniform)")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
