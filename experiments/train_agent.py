"""Train occupancy-coupled SAC and standard SAC across seeds and compare them with the baselines.

    python experiments/train_agent.py                 # 5 seeds x 2 variants x 2 noise levels, in parallel
    python experiments/train_agent.py --seeds 0 1 --workers 4
    python experiments/train_agent.py --validation --set entropy_high=0.05   # hyperparameter selection

Every policy is evaluated on the same held-out episodes (jittered starting states), so
differences between policies are paired. Hyperparameters are chosen with --validation,
which evaluates on a separate set of episodes, so the evaluation episodes are never
used for tuning. Agents are evaluated both sampling from their policy and taking its
most probable drug. Writes a JSON report to runs/train/.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ENV_KWARGS = dict(expected_reward=True, markov_observation=True, r0_spread=0.04, mu0_spread=0.5)
VALIDATION_OFFSET = 10_000  # validation episodes run on evaluation seed + this, disjoint from evaluation and training
VARIANTS = {"oc-sac": {"entropy": "pushforward"}, "sac": {"entropy": "action"}}


def make_env(sigma: float):
    from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
    from ocmdp.envs.wrappers import MechanismObservation

    return MechanismObservation(ToyOCMDPEnv(ToyConfig.paired(resistance_noise=sigma, **ENV_KWARGS)))


def _episode_stats(env, policy, episodes: int, validation: bool) -> dict:
    from ocmdp.eval.metrics import expected_failure_rate, occupancy_entropy, pushforward_occupancy_entropy
    from ocmdp.eval.rollout import rollout
    from ocmdp.eval.seeds import eval_seed

    offset = VALIDATION_OFFSET if validation else 0
    logs = [rollout(env, policy, eval_seed(s) + offset) for s in range(episodes)]
    assignment = env.unwrapped.coupling.mechanism_assignment()
    return {
        "expected_failure_rate": [expected_failure_rate(log) for log in logs],
        "H_action": float(np.mean([occupancy_entropy(log.occupancy) for log in logs])),
        "H_pushforward": float(np.mean([pushforward_occupancy_entropy(log.occupancy, assignment) for log in logs])),
    }


def train_and_evaluate(job: tuple[str, float, int, int, int, dict, bool]) -> dict:
    variant, sigma, seed, total_quarters, episodes, overrides, validation = job
    import torch

    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, train

    torch.set_num_threads(1)
    env = make_env(sigma)
    cfg = replace(SACConfig(total_quarters=total_quarters), **overrides, **VARIANTS[variant])
    start = time.time()
    try:
        result = train(env, cfg, seed)
    except FloatingPointError as exc:
        return {"variant": variant, "sigma": sigma, "seed": seed, "diverged": str(exc)}
    policy = QuarterlyChoicePolicy(result.agent.choose, env.action_space.n, variant)
    argmax = QuarterlyChoicePolicy(lambda obs, rng: result.agent.choose(obs, rng, greedy=True), env.action_space.n, variant)
    return {
        "variant": variant,
        "sigma": sigma,
        "seed": seed,
        "seconds": time.time() - start,
        "learning_curve": [float(np.mean(result.episode_failure[i : i + 25])) for i in range(0, len(result.episode_failure), 25)],
        **_episode_stats(env, policy, episodes, validation),
        "argmax": _episode_stats(env, argmax, episodes, validation),
    }


def evaluate_baselines(job: tuple[float, int, bool]) -> dict:
    sigma, episodes, validation = job
    from ocmdp.baselines.heuristics import MyopicGreedy, uniform_mixing
    from ocmdp.baselines.oracle import FullInformationOracle, tuned_cycling

    env = make_env(sigma)
    policies = [uniform_mixing(4), MyopicGreedy(), tuned_cycling(env.unwrapped.config), FullInformationOracle(env)]
    return {"sigma": sigma, "baselines": {p.name: _episode_stats(env, p, episodes, validation) for p in policies}}


def summarise(runs: list[dict], baselines: dict, sigma: float) -> dict:
    from ocmdp.eval.vv import aggregate

    base = {name: np.array(stats["expected_failure_rate"]) for name, stats in baselines.items()}
    uniform, oracle = base["uniform"].mean(), base["oracle"].mean()
    out = {"baselines": {name: aggregate(rates) for name, rates in base.items()}}
    per_variant = {}
    for variant in VARIANTS:
        done = [r for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" not in r]
        per_seed = {r["seed"]: np.mean(r["expected_failure_rate"]) for r in done}
        per_variant[variant] = per_seed
        rates = np.array(list(per_seed.values()))
        out[variant] = {
            "expected_failure_rate": aggregate(rates),
            "argmax_expected_failure_rate": aggregate([np.mean(r["argmax"]["expected_failure_rate"]) for r in done]),
            "share_of_oracle_gain": aggregate((uniform - rates) / (uniform - oracle)),
            "runs_beating_uniform": int(np.sum(rates < uniform - 0.5 * (uniform - oracle))),
            "runs": len(rates),
            "diverged": sum(1 for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" in r),
            "H_action": float(np.mean([r["H_action"] for r in done])),
            "H_pushforward": float(np.mean([r["H_pushforward"] for r in done])),
        }
    shared = sorted(set(per_variant["oc-sac"]) & set(per_variant["sac"]))
    out["oc-sac minus sac (paired by seed)"] = aggregate([per_variant["oc-sac"][s] - per_variant["sac"][s] for s in shared])
    return out


def _parse_overrides(parser: argparse.ArgumentParser, pairs: list[str]) -> dict:
    from ocmdp.agents.oc_sac import SACConfig

    defaults, overrides = asdict(SACConfig()), {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in defaults or not isinstance(defaults[key], (int, float)) or not value:
            parser.error(f"--set takes KEY=VALUE for a numeric SACConfig field other than entropy, got {pair!r}")
        overrides[key] = type(defaults[key])(value)
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--quarters", type=int, default=12_000, help="agent decisions per training run")
    parser.add_argument("--episodes", type=int, default=10, help="held-out evaluation episodes per policy")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="SACConfig overrides, applied to both variants")
    parser.add_argument("--validation", action="store_true", help="evaluate on validation episodes, for hyperparameter selection")
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    args = parser.parse_args()
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    overrides = _parse_overrides(parser, args.set)

    jobs = [(v, s, seed, args.quarters, args.episodes, overrides, args.validation) for s in args.sigmas for v in VARIANTS for seed in args.seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        baseline_futures = [pool.submit(evaluate_baselines, (s, args.episodes, args.validation)) for s in args.sigmas]
        runs = list(pool.map(train_and_evaluate, jobs))
        baselines = {f.result()["sigma"]: f.result()["baselines"] for f in baseline_futures}

    from ocmdp.agents.oc_sac import SACConfig

    config = replace(SACConfig(total_quarters=args.quarters), **overrides)
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "sac_config": asdict(config), "runs": runs}
    report["summary"] = {str(s): summarise(runs, baselines[s], s) for s in args.sigmas}
    path = args.out / f"train_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(report, indent=2))

    for sigma in args.sigmas:
        summary = report["summary"][str(sigma)]
        episodes = "validation" if args.validation else "held-out evaluation"
        print(f"\nPairedDrugs, resistance noise {sigma}: expected failure rate on {args.episodes} {episodes} episodes")
        for name, agg in summary["baselines"].items():
            print(f"  {name:30} {agg['mean']:.4f} ± {agg['ci95']:.4f}")
        for variant in VARIANTS:
            row = summary[variant]
            print(
                f"  {variant:30} {row['expected_failure_rate']['mean']:.4f} ± {row['expected_failure_rate']['ci95'] or 0:.4f}"
                f"   argmax {row['argmax_expected_failure_rate']['mean']:.4f}"
                f"   oracle gain captured {row['share_of_oracle_gain']['mean']:.0%}"
                f"   runs learning {row['runs_beating_uniform']}/{row['runs']}   diverged {row['diverged']}"
                f"   H_action {row['H_action']:.3f}  H_pushforward {row['H_pushforward']:.3f}"
            )
        diff = summary["oc-sac minus sac (paired by seed)"]
        print(f"  oc-sac - sac (paired by seed): {diff['mean']:+.4f} ± {diff['ci95'] or 0:.4f}")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
