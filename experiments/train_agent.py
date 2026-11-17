"""Train occupancy-coupled SAC and standard SAC across seeds and compare them with the baselines.

    python experiments/train_agent.py --tier dev                 # 5 seeds x 2 variants x 2 noise levels, in parallel
    python experiments/train_agent.py --tier dev --seeds 0 1 --workers 4
    python experiments/train_agent.py --tier val --set entropy_low=0.05 entropy_high=0.05   # hyperparameter selection
    python experiments/train_agent.py --tier dev --class-sizes 8 5 4 2 1   # 20 drugs in 5 unequal mechanism classes
    python experiments/train_agent.py --tier test --seeds 1000 1001 ... --prereg preregistrations/<file>.yaml

Seeds come in three disjoint tiers, read from seeds.yaml: dev for exploration, val for
choosing hyperparameters, test for the one confirmatory run of a preregistration. A
tier fixes both the training seeds a run may use and the episodes it is evaluated on,
so nothing tuned on dev or val has seen a test episode. Every policy in a run is
evaluated on the same episodes (jittered starting states), so differences between
policies are paired. Agents are evaluated both sampling from their policy and taking
its most probable drug. Writes a JSON report to runs/train/, which records the tier,
the preregistration, the git commit, the instance and the entropy targets.

The instance is one mechanism class per entry of --class-sizes, each holding that many
interchangeable drugs; the default, two classes of two, is PairedDrugs. The report
carries the class sizes, their imbalance KL (log M - H(n / K), zero for equal classes)
and, per trained agent, the entropy its policy spends inside the classes.

Two diagnostics separate reasons an agent can trail the planner. Every policy's failure
rate is also reported discounted at the agent's gamma, the objective the agent is
trained on, which differs from the undiscounted rate the policies are compared on. Each
agent's reward critics are compared with the soft returns realised after the drugs it
chose in evaluation.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ENV_KWARGS = dict(expected_reward=True, markov_observation=True, r0_spread=0.04, mu0_spread=0.5)
ROOT = Path(__file__).resolve().parents[1]
TIERS = ("dev", "val", "test")
DEFAULT_SEEDS = 5  # per variant and noise level when --seeds is not given: the first seeds of the tier
VARIANTS = {"oc-sac": {"entropy": "pushforward"}, "sac": {"entropy": "action"}}
PAIRED_DRUGS = (2, 2)  # class sizes of the default instance


def make_env(sigma: float, class_sizes: tuple[int, ...] = PAIRED_DRUGS):
    """One mechanism per class, selected for only by the drugs of that class."""
    from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
    from ocmdp.envs.wrappers import MechanismObservation

    rows = [tuple(float(j == m) for j in range(len(class_sizes))) for m, size in enumerate(class_sizes) for _ in range(size)]
    return MechanismObservation(ToyOCMDPEnv(ToyConfig(cross_resistance=tuple(rows), resistance_noise=sigma, **ENV_KWARGS)))


def imbalance_kl(class_sizes: tuple[int, ...]) -> float:
    """KL from the mechanism distribution of a uniform policy over drugs to the uniform one: log M - H(n / K)."""
    share = np.asarray(class_sizes, dtype=float) / sum(class_sizes)
    return float(np.log(len(share)) + np.sum(share * np.log(share)))


def load_tiers(path: Path) -> dict[str, range]:
    """Read `tier: [first, last]` lines (inclusive) into seed ranges; the three tiers must be present and disjoint."""
    found = re.findall(r"^(\w+):\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", path.read_text(), re.M)
    tiers = {name: range(int(first), int(last) + 1) for name, first, last in found}
    if set(tiers) != set(TIERS) or len(found) != len(TIERS) or any(len(r) == 0 for r in tiers.values()):
        raise ValueError(f"{path} must define a non-empty [first, last] range for each of {', '.join(TIERS)}, once")
    ordered = sorted(tiers.values(), key=lambda r: r.start)
    if any(a.stop > b.start for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"the seed tiers in {path} overlap")
    return tiers


def git_state() -> dict:
    """Commit of the working tree the run was launched from, and whether anything outside runs/ differs from it.

    Every launch appends to the ledger under runs/ and leaves a report there, so those
    files say nothing about whether the code is the committed code.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)runs"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _rollouts(env, policy, episodes: int, first_episode: int) -> list:
    from ocmdp.eval.rollout import rollout
    from ocmdp.eval.seeds import eval_seed

    return [rollout(env, policy, eval_seed(first_episode + e)) for e in range(episodes)]


def quarterly_failure(log) -> np.ndarray:
    """(Q,) mean failure probability of the drugs given in each quarter of an episode."""
    given = log.fail_prob[log.decision_quarter, log.actions]
    return np.bincount(log.decision_quarter, weights=given) / np.bincount(log.decision_quarter)


def discounted_failure_rate(failure: np.ndarray, gamma: float) -> float:
    """Failure rate with quarter q weighted by gamma**q: what an agent discounting at gamma is trained to lower."""
    weights = gamma ** np.arange(len(failure))
    return float(weights @ failure / weights.sum())


def _episode_stats(env, logs: list, gamma: float) -> dict:
    from ocmdp.eval.metrics import expected_failure_rate, occupancy_entropy, pushforward_occupancy_entropy

    assignment = env.unwrapped.coupling.mechanism_assignment()
    return {
        "expected_failure_rate": [expected_failure_rate(log) for log in logs],
        "discounted_failure_rate": [discounted_failure_rate(quarterly_failure(log), gamma) for log in logs],
        "H_action": float(np.mean([occupancy_entropy(log.occupancy) for log in logs])),
        "H_pushforward": float(np.mean([pushforward_occupancy_entropy(log.occupancy, assignment) for log in logs])),
    }


def _policy_entropy(probs: np.ndarray, classes: list[tuple[int, ...]]) -> dict:
    """Mean over the visited decisions of the policy's entropy over drugs and of the part spent inside classes."""
    import torch

    from ocmdp.objectives.pushforward_entropy import categorical_entropy, within_class_entropy

    probs = torch.as_tensor(probs)
    return {"action": categorical_entropy(probs).mean().item(), "within_class": within_class_entropy(probs, classes).mean().item()}


def soft_returns(reward: np.ndarray, bonus: np.ndarray, gamma: float) -> np.ndarray:
    """(E, Q) per-quarter rewards and entropy bonuses alpha(s) H -> (E, Q) realised soft return from each decision.

    The return the reward critics are trained to predict for the drug chosen at quarter
    q: that quarter's reward, then every later quarter's reward plus its entropy bonus,
    discounted, with nothing beyond the horizon.
    """
    out = np.zeros_like(reward, dtype=float)
    out[:, -1] = reward[:, -1]
    for q in range(reward.shape[1] - 2, -1, -1):
        out[:, q] = reward[:, q] + gamma * (bonus[:, q + 1] + out[:, q + 1])
    return out


def _critic_check(agent, observations: np.ndarray, drugs: np.ndarray, logs: list) -> dict:
    """The reward critics' value of each drug chosen in the sampled evaluation against the soft return that followed it.

    Errors are in units of the scaled reward. `return_sd` is the spread of the returns
    themselves, the scale an error has to be read against.
    """
    import torch

    cfg = agent.cfg
    obs = torch.as_tensor(observations)
    with torch.no_grad():
        probs, log_alpha = agent.policy(obs)
        bonus = log_alpha.clamp(*cfg.log_alpha_range).exp() * agent.entropy(probs)
        predicted = agent.q(obs).gather(1, torch.as_tensor(drugs)[:, None]).squeeze(1).double().numpy()
    reward = -cfg.reward_scale * np.stack([quarterly_failure(log) for log in logs])
    realised = soft_returns(reward, bonus.double().numpy().reshape(reward.shape), cfg.gamma).ravel()
    error = predicted - realised
    return {
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "return_sd": float(realised.std()),
        "correlation": float(np.corrcoef(predicted, realised)[0, 1]),
    }


def train_and_evaluate(job: tuple[str, float, int, int, int, dict, int, tuple[int, ...]]) -> dict:
    variant, sigma, seed, total_quarters, episodes, overrides, first_episode, class_sizes = job
    import torch

    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, train

    torch.set_num_threads(1)
    env = make_env(sigma, class_sizes)
    cfg = SACConfig(**{"total_quarters": total_quarters, **overrides, **VARIANTS[variant]})
    start = time.time()
    try:
        result = train(env, cfg, seed)
    except FloatingPointError as exc:
        return {"variant": variant, "sigma": sigma, "seed": seed, "diverged": str(exc)}
    agent = result.agent
    observations, drugs = [], []  # every quarterly decision of the sampled evaluation, in order

    def choose(obs, rng):
        observations.append(np.array(obs, dtype=np.float32))
        drugs.append(agent.choose(obs, rng))
        return drugs[-1]

    policy = QuarterlyChoicePolicy(choose, env.action_space.n, variant)
    argmax = QuarterlyChoicePolicy(lambda obs, rng: agent.choose(obs, rng, greedy=True), env.action_space.n, variant)
    logs = _rollouts(env, policy, episodes, first_episode)
    observations, drugs = np.stack(observations), np.array(drugs)
    return {
        "variant": variant,
        "sigma": sigma,
        "seed": seed,
        "seconds": time.time() - start,
        "learning_curve": [float(np.mean(result.episode_failure[i : i + 25])) for i in range(0, len(result.episode_failure), 25)],
        **_episode_stats(env, logs, cfg.gamma),
        "policy_entropy": _policy_entropy(np.stack([agent.probs(obs) for obs in observations]), env.unwrapped.coupling.identical_row_classes()),
        "critic": _critic_check(agent, observations, drugs, logs),
        "argmax": _episode_stats(env, _rollouts(env, argmax, episodes, first_episode), cfg.gamma),
    }


def evaluate_baselines(job: tuple[float, int, int, tuple[int, ...], float]) -> dict:
    sigma, episodes, first_episode, class_sizes, gamma = job
    from ocmdp.baselines.heuristics import MyopicGreedy, uniform_mixing
    from ocmdp.baselines.oracle import FullInformationOracle, tuned_cycling

    env = make_env(sigma, class_sizes)
    policies = [uniform_mixing(env.action_space.n), MyopicGreedy(), tuned_cycling(env.unwrapped.config), FullInformationOracle(env)]
    return {"sigma": sigma, "baselines": {p.name: _episode_stats(env, _rollouts(env, p, episodes, first_episode), gamma) for p in policies}}


def summarise(runs: list[dict], baselines: dict, sigma: float) -> dict:
    from ocmdp.eval.vv import aggregate

    base = {name: np.array(stats["expected_failure_rate"]) for name, stats in baselines.items()}
    uniform, oracle = base["uniform"].mean(), base["oracle"].mean()
    out = {"baselines": {name: aggregate(rates) for name, rates in base.items()}}
    out["discounted_failure_rate"] = {name: aggregate(stats["discounted_failure_rate"]) for name, stats in baselines.items()}
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
            "policy_entropy": {k: float(np.mean([r["policy_entropy"][k] for r in done])) for k in ("action", "within_class")},
            "critic": {k: float(np.mean([r["critic"][k] for r in done])) for k in ("bias", "rmse", "return_sd", "correlation")},
        }
        out["discounted_failure_rate"][variant] = aggregate([np.mean(r["discounted_failure_rate"]) for r in done])
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
    parser.add_argument("--tier", choices=TIERS, required=True, help="seed tier: fixes the allowed training seeds and the evaluation episodes")
    parser.add_argument("--prereg", type=Path, help="preregistration this run belongs to; required for --tier test")
    parser.add_argument("--seeds", type=int, nargs="+", help=f"training seeds inside the tier (default: its first {DEFAULT_SEEDS})")
    parser.add_argument("--seeds-file", type=Path, default=ROOT / "seeds.yaml", help="tier definitions")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--class-sizes", type=int, nargs="+", default=list(PAIRED_DRUGS), metavar="N", help="drugs per mechanism class")
    parser.add_argument("--quarters", type=int, default=12_000, help="agent decisions per training run")
    parser.add_argument("--episodes", type=int, default=10, help="held-out evaluation episodes per policy")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="SACConfig overrides, applied to both variants")
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    args = parser.parse_args()
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        tier = load_tiers(args.seeds_file)[args.tier]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.seeds is None:
        args.seeds = list(tier[:DEFAULT_SEEDS])
    outside = sorted(set(args.seeds) - set(tier))
    if outside or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"--seeds must be distinct and inside the {args.tier} tier [{tier.start}, {tier.stop - 1}]; rejected: {outside or 'duplicates'}")
    if not 1 <= args.episodes <= len(tier):
        parser.error(f"--episodes must lie in [1, {len(tier)}]: evaluation episodes are drawn from the {args.tier} tier")
    if args.tier == "test" and (args.prereg is None or not args.prereg.is_file()):
        parser.error("--tier test needs --prereg pointing at an existing preregistration file")
    if len(args.class_sizes) < 2 or min(args.class_sizes) < 1:
        parser.error("--class-sizes needs at least two classes of at least one drug each")
    overrides = _parse_overrides(parser, args.set)
    git = git_state()  # read at launch: the tree may change while the run is in progress

    from ocmdp.agents.oc_sac import SACConfig
    from ocmdp.eval.seeds import eval_seed

    config = SACConfig(**{"total_quarters": args.quarters, **overrides})
    sizes = tuple(args.class_sizes)
    jobs = [(v, s, seed, args.quarters, args.episodes, overrides, tier.start, sizes) for s in args.sigmas for v in VARIANTS for seed in args.seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        baseline_futures = [pool.submit(evaluate_baselines, (s, args.episodes, tier.start, sizes, config.gamma)) for s in args.sigmas]
        runs = list(pool.map(train_and_evaluate, jobs))
        baselines = {f.result()["sigma"]: f.result()["baselines"] for f in baseline_futures}

    toy = make_env(0.0, sizes).unwrapped
    max_entropy = {"oc-sac": math.log(toy.coupling.n_mechanisms), "sac": math.log(toy.coupling.n_drugs)}
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git": git,
        "evaluation_episode_seeds": [eval_seed(tier.start + e) for e in range(args.episodes)],
        "instance": {k: v for k, v in asdict(toy.config).items() if k != "resistance_noise"},
        "class_structure": {"class_sizes": list(sizes), "imbalance_kl_nats": imbalance_kl(sizes)},
        "entropy_target_nats": {v: [config.entropy_low * h, config.entropy_high * h] for v, h in max_entropy.items()},
        "sac_config": asdict(config),
        "runs": runs,
    }
    report["summary"] = {str(s): summarise(runs, baselines[s], s) for s in args.sigmas}
    path = args.out / f"train_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(report, indent=2))

    for sigma in args.sigmas:
        summary = report["summary"][str(sigma)]
        print(f"\nClass sizes {list(sizes)}, resistance noise {sigma}: expected failure rate on {args.episodes} {args.tier}-tier episodes")
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
                f"   policy entropy inside classes {row['policy_entropy']['within_class']:.3f}"
            )
        diff = summary["oc-sac minus sac (paired by seed)"]
        print(f"  oc-sac - sac (paired by seed): {diff['mean']:+.4f} ± {diff['ci95'] or 0:.4f}")
        discounted = summary["discounted_failure_rate"]
        print(f"  failure rate discounted at {config.gamma} per quarter: " + "   ".join(f"{name} {agg['mean']:.4f}" for name, agg in discounted.items()))
        for variant in VARIANTS:
            critic = summary[variant]["critic"]
            print(
                f"  {variant} critic against realised soft returns: bias {critic['bias']:+.3f}  rmse {critic['rmse']:.3f}"
                f"  (returns sd {critic['return_sd']:.3f})  correlation {critic['correlation']:.3f}"
            )
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
