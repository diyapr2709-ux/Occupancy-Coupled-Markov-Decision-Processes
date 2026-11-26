"""Phase diagram of the coupling: where acting on the observed state beats committing to a schedule.

    python experiments/sweep_phase_diagram.py --tier dev
    python experiments/sweep_phase_diagram.py --tier dev --delays 0 1 4 --lams 0 0.8 --sigmas 0 0.2 --episodes 10
    python experiments/sweep_phase_diagram.py --tier dev --backend modal

Each cell of the grid is one environment: delay d (quarters before prescribing becomes
selection pressure), occupancy memory lam (beta = 1 - lam is the weight of the newest
quarter) and resistance noise sigma, on the instance given by --class-sizes. Every
policy in a cell is rolled out on the same evaluation episodes of the tier, with the
starting state jittered as in training, so differences within a cell are paired.

Policies, none of them learned:
  uniform            uniform mixing over drugs
  greedy             the drug with the lowest current failure probability (closed loop, myopic)
  rotation           the best class rotation, chosen on the mean-field model from the nominal start
                     and never revised (a blind fixed schedule)
  planned-once       the noise-free planner: plans the whole episode once, from the episode's true
                     starting state, on the noise-free model, and commits to it
  replanned-every-4  the same planner, replanning from the state reached every 4 quarters
  replanned-every-1  the same planner, replanning every quarter

Reported per cell, paired over episodes with 95% t-intervals:
  value_of_feedback      planned-once minus replanned-every-1: what revising the plan from the
                         observed state is worth over committing at the start, with the same planner.
                         Without noise the model is exact, and the only departures from the plan are
                         patients sampled from a mixed quarter; the gain there is mostly extra
                         optimisation by the replanner, and that column is the floor the noisy
                         columns are read against.
  feedback_over_rotation rotation minus replanned-every-1
  greedy_over_rotation   rotation minus greedy: a closed loop with no model against a blind schedule
  greedy_regret          greedy minus replanned-every-1: what a model of the dynamics adds to acting on the state
  planned_once_minus_greedy  a model without feedback against feedback without a model; unlike the
                         contrasts above its sign can go either way, so a predicted sign change can fail

Planning only; nothing is trained. Writes a JSON report to runs/phase/ with every
per-episode value, so every summary number can be recomputed.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from train_agent import (ENV_KWARGS, PAIRED_DRUGS, ROOT, TIERS, check_seed_file, check_test_launch, cross_resistance,
                         git_state, launch_arguments, load_tiers, prereg_digest, record_launch)  # fmt: skip

REPORT_SCHEMA = 2  # this script's own report layout, unchanged by the training reports' schema 3

POLICIES = ("uniform", "greedy", "rotation", "planned-once", "replanned-every-4", "replanned-every-1")
CONTRASTS = {  # name: (minuend, subtrahend); positive means the subtrahend fails less
    "value_of_feedback": ("planned-once", "replanned-every-1"),
    "feedback_over_rotation": ("rotation", "replanned-every-1"),
    "greedy_over_rotation": ("rotation", "greedy"),
    "greedy_regret": ("greedy", "replanned-every-1"),
    "planned_once_minus_greedy": ("planned-once", "greedy"),
}


def cell_config(delay: int, lam: float, sigma: float, class_sizes: tuple[int, ...], horizon: int,
                kappa: float | None = None, delta: float | None = None):  # fmt: skip
    """One cell's environment; kappa and delta fall back to ToyConfig's defaults when None."""
    from ocmdp.envs.toy import ToyConfig

    rates = {name: value for name, value in (("kappa", kappa), ("delta", delta)) if value is not None}
    return ToyConfig(
        cross_resistance=cross_resistance(class_sizes), delay=delay, lam=lam, resistance_noise=sigma, horizon_quarters=horizon,
        **rates, **ENV_KWARGS,
    )  # fmt: skip


def make_policy(name: str, env):
    from ocmdp.baselines.heuristics import MyopicGreedy, uniform_mixing
    from ocmdp.baselines.oracle import FullInformationOracle, tuned_cycling

    horizon = env.unwrapped.config.horizon_quarters
    if name == "uniform":
        return uniform_mixing(env.action_space.n)
    if name == "greedy":
        return MyopicGreedy()
    if name == "rotation":
        return tuned_cycling(env.unwrapped.config)
    if name == "planned-once":
        return FullInformationOracle(env, replan_every=horizon, name=name)
    every = int(name.rsplit("-", 1)[1])
    return FullInformationOracle(env, replan_every=every, name=name)


def evaluate_cell_policy(job: tuple[int, float, float, tuple, str, int, int]) -> dict:
    """Roll one policy out on one cell's evaluation episodes."""
    delay, lam, sigma, instance, policy_name, episodes, first_episode = job
    import torch

    from ocmdp.envs.toy import ToyOCMDPEnv
    from ocmdp.eval.metrics import expected_failure_rate
    from ocmdp.eval.rollout import rollout
    from ocmdp.eval.seeds import eval_seed

    torch.set_num_threads(1)
    env = ToyOCMDPEnv(cell_config(delay, lam, sigma, *instance))
    policy = make_policy(policy_name, env)
    start = time.time()
    logs = [rollout(env, policy, eval_seed(first_episode + e)) for e in range(episodes)]
    return {
        "delay": delay,
        "lam": lam,
        "sigma": sigma,
        "policy": policy_name,
        "expected_failure_rate": [expected_failure_rate(log) for log in logs],
        "final_fail_prob": [log.fail_prob[-1].tolist() for log in logs],  # per drug at the horizon
        "seconds": time.time() - start,
    }


def summarise_cell(rates: dict[str, list[float]]) -> dict:
    from ocmdp.eval.vv import aggregate

    out = {"expected_failure_rate": {name: aggregate(values) for name, values in rates.items()}}
    for contrast, (a, b) in CONTRASTS.items():
        out[contrast] = aggregate(np.asarray(rates[a]) - np.asarray(rates[b]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=TIERS, required=True, help="seed tier: fixes the evaluation episodes")
    parser.add_argument("--prereg", type=Path, help="preregistration this sweep belongs to; required for --tier test")
    parser.add_argument("--platform", help="label for the account or machine the jobs run on, not a personal name; required for val and test")
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    parser.add_argument("--class-sizes", type=int, nargs="+", default=list(PAIRED_DRUGS), metavar="N", help="drugs per mechanism class")
    parser.add_argument("--horizon", type=int, default=40, help="quarters per episode")
    parser.add_argument("--kappa", type=float, help="acquisition rate under full pressure (ToyConfig's default when omitted)")
    parser.add_argument("--delta", type=float, help="reversion rate (ToyConfig's default when omitted)")
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes per cell, shared by every policy")
    parser.add_argument("--seeds-file", type=Path, default=ROOT / "seeds.yaml", help="tier definitions")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="local processes (--backend local)")
    parser.add_argument("--backend", choices=("local", "modal"), default="local", help="where the jobs run; see experiments/modal_backend.py")
    parser.add_argument("--out", type=Path, default=Path("runs/phase"))
    args = parser.parse_args()
    try:
        tier = load_tiers(args.seeds_file)[args.tier]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    check_seed_file(parser, args.tier, tier, args.seeds_file)
    if min(args.delays) < 0 or len(set(args.delays)) != len(args.delays):
        parser.error("--delays must be distinct non-negative integers")
    if any(not 0.0 <= lam < 1.0 for lam in args.lams) or len(set(args.lams)) != len(args.lams):
        parser.error("--lams must be distinct values in [0, 1)")
    if any(not 0.0 <= s < np.inf for s in args.sigmas) or len(set(args.sigmas)) != len(args.sigmas):
        parser.error("--sigmas must be distinct, finite and non-negative")
    if len(args.class_sizes) < 2 or min(args.class_sizes) < 1:
        parser.error("--class-sizes needs at least two classes of at least one drug each")
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    try:
        from ocmdp.envs.toy import ToyOCMDPEnv

        ToyOCMDPEnv(cell_config(0, 0.0, 0.0, tuple(args.class_sizes), args.horizon, args.kappa, args.delta))
    except ValueError as exc:
        parser.error(f"--kappa/--delta: {exc}")
    if not 2 <= args.episodes <= len(tier):
        parser.error(f"--episodes must lie in [2, {len(tier)}]: an interval needs two, and episodes are drawn from the {args.tier} tier")
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    git = git_state()  # read at launch: the tree may change while the sweep is in progress
    prereg_slug = check_test_launch(parser, args.prereg, git, args.out, args.platform) if args.tier == "test" else None
    if args.tier != "dev" and not args.platform:
        parser.error(f"--tier {args.tier} needs --platform naming the account or machine the jobs run on")
    launch_id = f"phase_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    launch_args = launch_arguments(args)
    prereg_sha256 = prereg_digest(args.prereg)
    record_launch(args.out, launch_id, launch_args, git, args.backend, prereg_slug, prereg_sha256)

    sizes = tuple(args.class_sizes)
    cells = list(itertools.product(args.delays, args.lams, args.sigmas))
    instance = (sizes, args.horizon, args.kappa, args.delta)
    jobs = [(d, lam, s, instance, p, args.episodes, tier.start) for d, lam, s in cells for p in POLICIES]
    if args.backend == "modal":
        import modal_backend

        backend = modal_backend.description()
        results = modal_backend.run_jobs([("sweep_phase_diagram", "evaluate_cell_policy", job) for job in jobs])
    else:
        backend = {"name": "local", "platform": platform.platform(), "workers": args.workers}
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(evaluate_cell_policy, jobs))

    from ocmdp.eval.seeds import eval_seed

    by_cell = {cell: {} for cell in cells}
    for r in results:
        by_cell[(r["delay"], r["lam"], r["sigma"])][r["policy"]] = r["expected_failure_rate"]
    base = cell_config(0, 0.0, 0.0, *instance)
    report = {
        "report_schema": REPORT_SCHEMA,
        "launch_id": launch_id,
        "platform": args.platform,
        "prereg_sha256": prereg_sha256,
        "args": launch_args,
        "git": git,
        "backend": backend,
        "evaluation_episode_seeds": [eval_seed(tier.start + e) for e in range(args.episodes)],
        "instance": {k: v for k, v in asdict(base).items() if k not in ("delay", "lam", "resistance_noise")},
        "policies": list(POLICIES),
        "contrasts": {name: list(pair) for name, pair in CONTRASTS.items()},
        "results": results,
        "cells": [{"delay": d, "lam": lam, "sigma": s, **summarise_cell(by_cell[(d, lam, s)])} for d, lam, s in cells],
    }
    path = args.out / f"{launch_id}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    print(f"Class sizes {list(sizes)}, horizon {args.horizon}: {len(cells)} cells, {args.episodes} {args.tier}-tier episodes each")
    print(f"  {'delay':>5} {'lam':>5} {'sigma':>5}   " + "   ".join(f"{name} (95% interval)" for name in CONTRASTS))
    for cell in report["cells"]:
        row = "   ".join(f"{cell[name]['mean']:+.4f} ± {cell[name]['ci95']:.4f}" for name in CONTRASTS)
        print(f"  {cell['delay']:>5} {cell['lam']:>5} {cell['sigma']:>5}   {row}")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
