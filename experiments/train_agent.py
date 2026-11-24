"""Train occupancy-coupled SAC and standard SAC across seeds and compare them with the baselines.

    python experiments/train_agent.py --tier dev                 # 5 seeds x 2 variants x 2 noise levels, in parallel
    python experiments/train_agent.py --tier dev --seeds 0 1 --workers 4
    python experiments/train_agent.py --tier dev --backend modal   # one Modal container per job
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

Diagnostics of why an agent trails the planner. Every policy's failure rate is also
reported discounted at the agent's gamma, the objective the agent is trained on. Without
noise, each agent's critics and actor are checked against counterfactual values of every
class under the agent's own continuation (`critic`), and with --diagnose-gap its gap to
the planner is split by cause along its own trajectory (`gap_attribution`). Each agent's
temperature path, dual-variable path and temperatures at evaluation states are logged.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
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
REPLANNER = "oracle[replans every quarter]"  # the planner to compare against under noise, where none is a bound


def cross_resistance(class_sizes: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """One mechanism per class, selected for only by the drugs of that class."""
    return tuple(tuple(float(j == m) for j in range(len(class_sizes))) for m, size in enumerate(class_sizes) for _ in range(size))


def make_env(sigma: float, class_sizes: tuple[int, ...] = PAIRED_DRUGS, delay: int | None = None, lam: float | None = None):
    """The training environment; delay and lam fall back to ToyConfig's defaults when None."""
    from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
    from ocmdp.envs.wrappers import MechanismObservation

    coupling = {name: value for name, value in (("delay", delay), ("lam", lam)) if value is not None}
    config = ToyConfig(cross_resistance=cross_resistance(class_sizes), resistance_noise=sigma, **coupling, **ENV_KWARGS)
    return MechanismObservation(ToyOCMDPEnv(config))


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


CODE_PATHS = ("ocmdp", "experiments", "pyproject.toml", "seeds.yaml")  # what a run executes or reads


def git_state() -> dict:
    """Commit the run was launched from, whether the code differs from it, and a hash of the difference.

    Only CODE_PATHS count: notes, gathered material and earlier reports change nothing a
    run computes. Two runs with the same commit and the same `code_diff_sha256` ran the
    same code, tracked edits and untracked files included.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    try:
        diff = git("diff", "HEAD", "--", *CODE_PATHS)
        untracked = sorted(name for name in git("ls-files", "-z", "--others", "--exclude-standard", "--", *CODE_PATHS).split("\0") if name)
        digest = hashlib.sha256(diff.encode())
        for name in untracked:
            digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
        dirty = bool(diff or untracked)
        return {
            "commit": git("rev-parse", "HEAD").strip(),
            "dirty": dirty,
            "code_diff_sha256": digest.hexdigest() if dirty else None,
            "untracked_code": untracked,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "code_diff_sha256": None, "untracked_code": None}


def host() -> dict:
    """The machine a job ran on. Training is chaotic, so results reproduce only where the arithmetic is identical.

    `simd` is the level torch dispatches to, which a pinned environment forces; `cpu` is
    the model the operating system reports, which a sandboxed container may hide.
    """
    import torch

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
    except OSError:
        pass
    return {"machine": platform.machine(), "cpu": cpu, "simd": torch.backends.cpu.get_cpu_capability()}


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
        "final_fail_prob": [log.fail_prob[-1].tolist() for log in logs],  # per drug at the horizon: any efficacy threshold applies later
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


def _continuation_return(snapshot, drug: int, agent, mask_rng: np.random.Generator) -> float:
    """Soft return of giving `drug` for one quarter from `snapshot`, then following the agent's most probable drug.

    Runs on a copy of the environment, so with noise every counterfactual meets the same
    shocks (common random numbers) and differences between drugs are paired.
    """
    import torch

    from ocmdp.components.constraint_filter import sample_feasible

    cfg = agent.cfg
    env = copy.deepcopy(snapshot)
    state = env.unwrapped.state()
    quarter, mask, n_drugs = state["quarter"], state["action_mask"], env.action_space.n
    rewards, bonuses, done = [], [0.0], False
    while not done:
        total, n = 0.0, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(sample_feasible(np.eye(n_drugs)[drug], mask, mask_rng))
            total, n, done, mask = total + reward, n + 1, terminated or truncated, info["action_mask"]
            if info["quarter"] != quarter:
                break
        rewards.append(cfg.reward_scale * total / n)
        quarter = info["quarter"]
        if not done:
            with torch.no_grad():
                probs, log_alpha = agent.policy(torch.as_tensor(obs, dtype=torch.float32)[None])
                bonuses.append(float(log_alpha.clamp(*cfg.log_alpha_range).exp()[0] * agent.entropy(probs)[0]))
            drug = int(probs[0].argmax())
    return float(soft_returns(np.array([rewards]), np.array([bonuses]), cfg.gamma)[0, 0])


def _critic_check(agent, snapshots: list, observations: np.ndarray, classes: list[tuple[int, ...]]) -> dict:
    """Do the reward critics, and the actor, rank the classes the way the environment does?

    At every quarterly decision of the sampled evaluation, each class (one drug stands for
    its class: drugs in a class are interchangeable) is tried for one quarter on a copy of
    the environment, then the agent's most probable drug is followed to the horizon. The
    soft return that follows is the counterfactual value of that class under the agent's
    own continuation; the critics estimate it for the sampling policy, which differs only
    as far as the policy is not deterministic.

    Values are in units of the scaled reward. Regret is the counterfactual value lost by
    the class the critics (or the actor) rank first, against the best class; the mean gap
    between the best and second-best class is the scale it is read against.
    """
    import torch

    rng = np.random.default_rng(0)  # only used to resolve contraindications, of which the training environments have none
    representatives = [c[0] for c in classes]
    counterfactual = np.array([[_continuation_return(s, d, agent, rng) for d in representatives] for s in snapshots])
    obs = torch.as_tensor(observations)
    with torch.no_grad():
        critic = agent.q(obs)[:, representatives].double().numpy()
        probs, _ = agent.policy(obs)
    actor = np.stack([probs[:, list(c)].sum(-1).double().numpy() for c in classes], axis=-1)
    best = counterfactual.max(axis=1)
    ordered = np.sort(counterfactual, axis=1)

    def regret(scores: np.ndarray) -> np.ndarray:
        return best - counterfactual[np.arange(len(scores)), scores.argmax(axis=1)]

    def centred(values: np.ndarray) -> np.ndarray:
        return values - values.mean(axis=1, keepdims=True)

    return {
        "states": len(snapshots),
        "critic_ranks_best_class_first": float(np.mean(regret(critic) <= 1e-9)),
        "actor_ranks_best_class_first": float(np.mean(regret(actor) <= 1e-9)),
        "critic_choice_regret": float(regret(critic).mean()),
        "actor_choice_regret": float(regret(actor).mean()),
        "mean_gap_best_to_second": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "centred_value_rmse": float(np.sqrt(np.mean((centred(critic) - centred(counterfactual)) ** 2))),
        "value_bias": float((critic - counterfactual).mean()),
    }


def _planned_failure(snapshot, first_drug: int | None = None, planner_steps: int = 200) -> float:
    """Failure from `snapshot` to the horizon, in expected-failure-rate units, under the noise-free planner.

    Optionally `first_drug` is given for the rest of the current quarter first. The
    planner plans once from the state it starts in, on a copy of the environment, so the
    snapshot is left untouched. Exact without noise, where the model replays the environment.
    """
    from ocmdp.baselines.oracle import FullInformationOracle
    from ocmdp.components.constraint_filter import sample_feasible

    env = copy.deepcopy(snapshot)
    toy = env.unwrapped
    rng = np.random.default_rng(0)  # only resolves contraindications, of which the training environments have none
    state = toy.state()
    info = {"quarter": state["quarter"], "resistance": state["resistance"], "action_mask": state["action_mask"]}
    obs, total, done = None, 0.0, False
    if first_drug is not None:
        quarter = info["quarter"]
        while not done and info["quarter"] == quarter:
            obs, reward, terminated, truncated, info = env.step(sample_feasible(np.eye(env.action_space.n)[first_drug], info["action_mask"], rng))
            total, done = total - reward, terminated or truncated
    planner = FullInformationOracle(env, replan_every=toy.config.horizon_quarters, steps=planner_steps, name="planned-once")
    while not done:
        obs, reward, terminated, truncated, info = env.step(planner.act(obs, info, rng))
        total, done = total - reward, terminated or truncated
    return total / (toy.config.horizon_quarters * toy.config.decisions_per_quarter)


def _gap_attribution(agent, snapshots: list, observations: np.ndarray, chosen: np.ndarray, logs: list,
                     classes: list[tuple[int, ...]], planner_steps: int = 200) -> dict:  # fmt: skip
    """Where along its own trajectory the agent loses against the noise-free planner, and why. Without noise only.

    At every quarterly decision s_t of the sampled evaluation: V*(s_t) is the planner's
    failure from s_t to the horizon, and Q*(s_t, c) the failure of class c for this quarter
    followed by the planner. The losses V*(s_t) - Q*(s_t, a_t) (sign flipped: lower failure
    is better) telescope along each episode to the agent's gap to the planner on it, which
    `closure_error` checks. Each loss splits into what one drug per quarter costs against a
    plan that may mix (V* against the best Q*) and the choice of a worse class, and the
    choice loss is attributed, in this order, to the agent's own continuation ranking
    classes differently from the planner (a local optimum of its own policy), the critics
    misranking classes against that continuation, or the actor departing from its critics.
    All values are in expected-failure-rate units and are relative to the planner, which is
    the best schedule it finds, not a proven optimum.
    """
    import torch

    rng = np.random.default_rng(0)
    owner = {d: c for c, members in enumerate(classes) for d in members}
    representatives = [c[0] for c in classes]
    per_episode = len(snapshots) // len(logs)
    obs = torch.as_tensor(observations)
    with torch.no_grad():
        critic_rank = agent.q(obs)[:, representatives].double().numpy().argmax(axis=1)
    totals = {"policy_class": 0.0, "continuation_ranks_differently": 0.0, "critic_misranks": 0.0, "actor_departs_from_critic": 0.0}
    closure, gaps = [], []
    for e, log in enumerate(logs):
        index = range(e * per_episode, (e + 1) * per_episode)
        planned = [_planned_failure(snapshots[i], planner_steps=planner_steps) for i in index]
        loss_sum = 0.0
        for k, i in enumerate(index):
            q_star = np.array([_planned_failure(snapshots[i], d, planner_steps) for d in representatives])
            q_own = np.array([_continuation_return(snapshots[i], d, agent, rng) for d in representatives])  # higher is better
            taken = owner[int(chosen[i])]
            totals["policy_class"] += q_star.min() - planned[k]
            choice_loss = q_star[taken] - q_star.min()
            if choice_loss > 0:
                if q_own.argmax() != q_star.argmin():
                    totals["continuation_ranks_differently"] += choice_loss
                elif critic_rank[i] != q_own.argmax():
                    totals["critic_misranks"] += choice_loss
                else:
                    totals["actor_departs_from_critic"] += choice_loss
            loss_sum += q_star[taken] - planned[k]  # Q*(s_t, a_t) - V*(s_t), which telescopes over the episode
        gap = float(np.mean(log.fail_prob[log.decision_quarter, log.actions])) - planned[0]
        gaps.append(gap)
        closure.append(abs(loss_sum - gap))
    n = len(logs)
    return {
        "gap_to_planner": float(np.mean(gaps)),
        "closure_error": float(max(closure)),
        **{f"lost_to_{name}": value / n for name, value in totals.items()},
    }


def train_and_evaluate(job: tuple[str, float, int, int, int, dict, int, tuple, bool]) -> dict:
    variant, sigma, seed, total_quarters, episodes, overrides, first_episode, instance, diagnose = job
    import torch

    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, train

    torch.set_num_threads(1)
    env = make_env(sigma, *instance)
    cfg = SACConfig(**{"total_quarters": total_quarters, **overrides, **VARIANTS[variant]})
    start = time.time()
    try:
        result = train(env, cfg, seed)
    except FloatingPointError as exc:
        return {"variant": variant, "sigma": sigma, "seed": seed, "host": host(), "diverged": str(exc)}
    agent = result.agent
    observations, snapshots, chosen = [], [], []  # every quarterly decision of the sampled evaluation, in order

    def choose(obs, rng):
        observations.append(np.array(obs, dtype=np.float32))
        snapshots.append(copy.deepcopy(env))  # before the quarter's first patient
        chosen.append(agent.choose(obs, rng))
        return chosen[-1]

    policy = QuarterlyChoicePolicy(choose, env.action_space.n, variant)
    argmax = QuarterlyChoicePolicy(lambda obs, rng: agent.choose(obs, rng, greedy=True), env.action_space.n, variant)
    logs = _rollouts(env, policy, episodes, first_episode)
    observations, chosen = np.stack(observations), np.array(chosen)
    classes = env.unwrapped.coupling.identical_row_classes()
    # with noise each counterfactual follows one noise path, which a single draw cannot separate from the critic's error
    critic = _critic_check(agent, snapshots, observations, classes) if sigma == 0 else {}
    gap = _gap_attribution(agent, snapshots, observations, chosen, logs, classes) if diagnose and sigma == 0 else {}
    with torch.no_grad():
        alpha_at_evaluation = agent.policy(torch.as_tensor(observations))[1].clamp(*cfg.log_alpha_range).exp().double().numpy()
    return {
        "variant": variant,
        "sigma": sigma,
        "seed": seed,
        "seconds": time.time() - start,
        "host": host(),
        "learning_curve": [float(np.mean(result.episode_failure[i : i + 25])) for i in range(0, len(result.episode_failure), 25)],
        **_episode_stats(env, logs, cfg.gamma),
        "policy_entropy": _policy_entropy(np.stack([agent.probs(obs) for obs in observations]), env.unwrapped.coupling.identical_row_classes()),
        "critic": critic,
        "gap_attribution": gap,
        "alpha_path": result.alpha,  # mean alpha(s) over each window of dual_update_every updates
        "xi_path": result.xi,
        "alpha_at_evaluation": {f"q{q}": float(np.percentile(alpha_at_evaluation, q)) for q in (10, 50, 90)},
        "argmax": _episode_stats(env, _rollouts(env, argmax, episodes, first_episode), cfg.gamma),
    }


def committed_per_quarter(base):
    """`base`'s most probable drug at the first decision of each quarter, given to every patient in that quarter.

    The agents' policy class: one drug per quarter. The oracle may split a quarter
    between drugs, which no agent can do, so this is the oracle's behaviour as far as
    that class can copy it: a reference for the class, not an upper bound on it.
    """
    from ocmdp.baselines.heuristics import HeuristicPolicy

    class CommittedPerQuarter(HeuristicPolicy):
        name = f"{base.name}[one drug per quarter]"

        def reset(self) -> None:
            base.reset()
            self._quarter, self._drug = None, None

        def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
            if info["quarter"] != self._quarter:
                self._quarter, self._drug = info["quarter"], int(np.argmax(base.probs(obs, info)))
            return np.eye(len(info["fail_prob"]))[self._drug]

    policy = CommittedPerQuarter()
    policy.reset()
    return policy


def evaluate_baselines(job: tuple[float, int, int, tuple, float]) -> dict:
    sigma, episodes, first_episode, instance, gamma = job
    from ocmdp.baselines.heuristics import MyopicGreedy, uniform_mixing
    from ocmdp.baselines.oracle import FullInformationOracle, tuned_cycling

    env = make_env(sigma, *instance)
    policies = [
        uniform_mixing(env.action_space.n),
        MyopicGreedy(),
        tuned_cycling(env.unwrapped.config),
        committed_per_quarter(FullInformationOracle(env)),
        FullInformationOracle(env),
        FullInformationOracle(env, replan_every=1, name=REPLANNER),
    ]
    return {"sigma": sigma, "baselines": {p.name: _episode_stats(env, _rollouts(env, p, episodes, first_episode), gamma) for p in policies}}


def iqm(values) -> float:
    """Interquartile mean: the mean after dropping int(n / 4) values from each end (scipy's trim_mean at 0.25)."""
    v = np.sort(np.asarray(values, dtype=float))
    cut = int(0.25 * v.size)
    return float(v[cut : v.size - cut].mean())


def probability_of_improvement(x, y) -> float:
    """P(X < Y) over all pairs of runs, ties counting one half: how often a run of x fails less than a run of y."""
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(np.mean((x < y) + 0.5 * (x == y)))


def bootstrap_ci95(values, resamples: int = 10_000, seed: int = 0) -> list[float]:
    """Percentile 95% interval of the mean, resampling the given values (seeds) with replacement."""
    v = np.asarray(values, dtype=float)
    means = v[np.random.default_rng(seed).integers(v.size, size=(resamples, v.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def two_level_bootstrap_ci95(differences: np.ndarray, resamples: int = 10_000, seed: int = 0) -> list[float]:
    """Percentile 95% interval of the mean of a (seeds, episodes) array of paired differences.

    Every seed is evaluated on the same episodes, so seeds and episodes are crossed, not
    nested: each resample draws seeds and episodes independently and keeps the same
    episodes for every drawn seed, so an effect shared by an episode across seeds keeps
    its full weight. Resampling seeds alone treats the evaluation episodes as fixed.
    """
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(d.shape[0], size=(resamples, d.shape[0]))
    episodes = rng.integers(d.shape[1], size=(resamples, d.shape[1]))
    means = d[seeds[:, :, None], episodes[:, None, :]].mean(axis=(1, 2))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(runs: list[dict], baselines: dict, sigma: float, equivalence_margin: float | None = None) -> dict:
    from ocmdp.eval.vv import aggregate

    base = {name: np.array(stats["expected_failure_rate"]) for name, stats in baselines.items()}
    uniform, oracle = base["uniform"].mean(), base["oracle"].mean()
    out = {"baselines": {name: aggregate(rates) for name, rates in base.items()}}
    out["discounted_failure_rate"] = {name: aggregate(stats["discounted_failure_rate"]) for name, stats in baselines.items()}
    per_variant, by_seed = {}, {}
    for variant in VARIANTS:
        done = [r for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" not in r]
        per_seed = {r["seed"]: np.mean(r["expected_failure_rate"]) for r in done}
        per_variant[variant] = per_seed
        by_seed[variant] = {r["seed"]: r["expected_failure_rate"] for r in done}
        rates = np.array(list(per_seed.values()))
        out[variant] = {
            "expected_failure_rate": aggregate(rates),
            "iqm_expected_failure_rate": iqm(rates),
            "argmax_expected_failure_rate": aggregate([np.mean(r["argmax"]["expected_failure_rate"]) for r in done]),
            "share_of_oracle_gain": aggregate((uniform - rates) / (uniform - oracle)),
            "share_of_replanner_gain": aggregate((uniform - rates) / (uniform - base[REPLANNER].mean())),
            "runs_learned": int(np.sum(rates < uniform - 0.5 * (uniform - oracle))),  # more than half the oracle's gain over uniform
            "runs": len(rates),
            "diverged": sum(1 for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" in r),
            "H_action": float(np.mean([r["H_action"] for r in done])),
            "H_pushforward": float(np.mean([r["H_pushforward"] for r in done])),
            "policy_entropy": {k: float(np.mean([r["policy_entropy"][k] for r in done])) for k in ("action", "within_class")},
            "critic": {k: float(np.mean([r["critic"][k] for r in done])) for k in done[0]["critic"]} if done else {},
            "gap_attribution": {k: float(np.mean([r["gap_attribution"][k] for r in done])) for k in done[0]["gap_attribution"]} if done else {},
            "alpha_last_training_window": aggregate([r["alpha_path"][-1] for r in done if r["alpha_path"]]) if any(r["alpha_path"] for r in done) else None,
            "alpha_median_at_evaluation": aggregate([r["alpha_at_evaluation"]["q50"] for r in done]) if done else None,
            "minus_baseline_paired_by_episode": {
                name: aggregate([np.mean(np.asarray(r["expected_failure_rate"]) - base[name]) for r in done]) for name in base
            },
        }
        out["discounted_failure_rate"][variant] = aggregate([np.mean(r["discounted_failure_rate"]) for r in done])
    shared = sorted(set(per_variant["oc-sac"]) & set(per_variant["sac"]))
    diffs = [per_variant["oc-sac"][s] - per_variant["sac"][s] for s in shared]
    out["oc-sac minus sac (paired by seed)"] = aggregate(diffs)
    out["paired comparison"] = {
        "seeds": shared,
        "probability_oc_sac_fails_less": probability_of_improvement([per_variant["oc-sac"][s] for s in shared], [per_variant["sac"][s] for s in shared]),
        "seeds_where_oc_sac_fails_less": int(np.sum(np.asarray(diffs) < 0)),
        "bootstrap_ci95_of_mean_difference": bootstrap_ci95(diffs),
        "two_level_bootstrap_ci95_of_mean_difference": two_level_bootstrap_ci95(
            [np.asarray(by_seed["oc-sac"][s]) - np.asarray(by_seed["sac"][s]) for s in shared]
        ),
        "iqm_of_difference": iqm(diffs),
    }
    if equivalence_margin is not None and len(diffs) > 1:
        from stats import equivalent

        out["paired comparison"]["equivalence"] = equivalent(diffs, equivalence_margin)
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
    parser.add_argument("--delay", type=int, help="quarters before prescribing becomes selection pressure (ToyConfig's default when omitted)")
    parser.add_argument("--lam", type=float, help="occupancy memory per quarter (ToyConfig's default when omitted)")
    parser.add_argument("--quarters", type=int, default=12_000, help="agent decisions per training run")
    parser.add_argument("--episodes", type=int, default=10, help="held-out evaluation episodes per policy")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="local processes (--backend local)")
    parser.add_argument("--backend", choices=("local", "modal"), default="local", help="where the jobs run; see experiments/modal_backend.py")
    parser.add_argument("--equivalence-margin", type=float, help="margin of a two one-sided test on the paired difference (none by default)")
    parser.add_argument("--diagnose-gap", action="store_true", help="attribute each agent's gap to the planner (no noise only; about 3 plans per decision)")
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
    if args.equivalence_margin is not None and not args.equivalence_margin > 0:
        parser.error("--equivalence-margin must be positive")
    if (args.delay is not None and args.delay < 0) or (args.lam is not None and not 0.0 <= args.lam < 1.0):
        parser.error("--delay must be non-negative and --lam must lie in [0, 1)")
    overrides = _parse_overrides(parser, args.set)
    git = git_state()  # read at launch: the tree may change while the run is in progress

    from ocmdp.agents.oc_sac import SACConfig
    from ocmdp.eval.seeds import eval_seed

    config = SACConfig(**{"total_quarters": args.quarters, **overrides})
    sizes = tuple(args.class_sizes)
    instance = (sizes, args.delay, args.lam)
    jobs = [(v, s, seed, args.quarters, args.episodes, overrides, tier.start, instance, args.diagnose_gap)
            for s in args.sigmas for v in VARIANTS for seed in args.seeds]  # fmt: skip
    baseline_jobs = [(s, args.episodes, tier.start, instance, config.gamma) for s in args.sigmas]
    if args.backend == "modal":
        import modal_backend

        backend = modal_backend.description()
        calls = [("train_agent", "evaluate_baselines", job) for job in baseline_jobs]
        calls += [("train_agent", "train_and_evaluate", job) for job in jobs]
        results = modal_backend.run_jobs(calls)
        baseline_results, runs = results[: len(baseline_jobs)], results[len(baseline_jobs) :]
    else:
        from importlib.metadata import version

        backend = {"name": "local", "platform": platform.platform(), "workers": args.workers,
                   "packages": {name: version(name) for name in ("numpy", "torch", "gymnasium")}}  # fmt: skip
        with ProcessPoolExecutor(args.workers) as pool:
            baseline_futures = [pool.submit(evaluate_baselines, job) for job in baseline_jobs]
            runs = list(pool.map(train_and_evaluate, jobs))
            baseline_results = [f.result() for f in baseline_futures]
    baselines = {result["sigma"]: result["baselines"] for result in baseline_results}

    toy = make_env(0.0, *instance).unwrapped
    max_entropy = {"oc-sac": math.log(toy.coupling.n_mechanisms), "sac": math.log(toy.coupling.n_drugs)}
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git": git,
        "backend": backend,
        "evaluation_episode_seeds": [eval_seed(tier.start + e) for e in range(args.episodes)],
        "instance": {k: v for k, v in asdict(toy.config).items() if k != "resistance_noise"},
        "class_structure": {"class_sizes": list(sizes), "imbalance_kl_nats": imbalance_kl(sizes)},
        "entropy_target_nats": {v: [config.entropy_low * h, config.entropy_high * h] for v, h in max_entropy.items()},
        "sac_config": {v: asdict(SACConfig(**{"total_quarters": args.quarters, **overrides, **VARIANTS[v]})) for v in VARIANTS},
        "runs": runs,
        "baselines": {str(s): baselines[s] for s in args.sigmas},  # per-episode values, so every contrast can be recomputed
    }
    report["summary"] = {str(s): summarise(runs, baselines[s], s, args.equivalence_margin) for s in args.sigmas}
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
                f"   runs learned {row['runs_learned']}/{row['runs']}   diverged {row['diverged']}"
                f"   H_action {row['H_action']:.3f}  H_pushforward {row['H_pushforward']:.3f}"
                f"   policy entropy inside classes {row['policy_entropy']['within_class']:.3f}"
            )
        diff, paired = summary["oc-sac minus sac (paired by seed)"], summary["paired comparison"]
        boot = paired["bootstrap_ci95_of_mean_difference"]
        print(
            f"  oc-sac - sac (paired by seed): {diff['mean']:+.4f} ± {diff['ci95'] or 0:.4f}   bootstrap [{boot[0]:+.4f}, {boot[1]:+.4f}]"
            f"   IQM {paired['iqm_of_difference']:+.4f}   P(oc-sac fails less) {paired['probability_oc_sac_fails_less']:.2f}"
            f"   oc-sac lower on {paired['seeds_where_oc_sac_fails_less']}/{len(paired['seeds'])} seeds"
        )
        discounted = summary["discounted_failure_rate"]
        print(f"  failure rate discounted at {config.gamma} per quarter: " + "   ".join(f"{name} {agg['mean']:.4f}" for name, agg in discounted.items()))
        for variant in VARIANTS:
            critic, row = summary[variant]["critic"], summary[variant]
            alpha = row["alpha_median_at_evaluation"]["mean"] if row["alpha_median_at_evaluation"] else float("nan")
            versus = row["minus_baseline_paired_by_episode"][REPLANNER]
            if row["gap_attribution"]:
                g = row["gap_attribution"]
                print(
                    f"  {variant} gap to the planner {g['gap_to_planner']:+.4f} (closure error {g['closure_error']:.1e}): policy class"
                    f" {g['lost_to_policy_class']:+.4f}, own continuation ranks differently {g['lost_to_continuation_ranks_differently']:+.4f},"
                    f" critic misranks {g['lost_to_critic_misranks']:+.4f}, actor departs {g['lost_to_actor_departs_from_critic']:+.4f}"
                )
            if not critic:
                continue
            print(
                f"  {variant}: critic ranks the best class first in {critic['critic_ranks_best_class_first']:.0%} of states,"
                f" actor in {critic['actor_ranks_best_class_first']:.0%}; regret critic {critic['critic_choice_regret']:.3f},"
                f" actor {critic['actor_choice_regret']:.3f} (mean gap {critic['mean_gap_best_to_second']:.3f}); median alpha at evaluation {alpha:.2e};"
                f" minus {REPLANNER} {versus['mean']:+.4f} ± {versus['ci95'] or 0:.4f}"
            )
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
