"""Verification and validation harness (Section 5 of the proposal).

Verification asks whether the implementation does what the specification says. Each
check returns pass, fail, or pending (its component does not exist yet). pytest runs
every check (tests/test_verification.py), so a failing check fails the build.

Validation asks whether policies achieve the outcome. It returns measurements, not
verdicts: the comparator table with regret against the full-information oracle, and the
oracle's degradation when its occupancy memory lam is misspecified.

    python experiments/validate.py
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from ocmdp.baselines.heuristics import Cycling, FixedGuideline, HeuristicPolicy, StationaryMixing, uniform_mixing
from ocmdp.baselines.oracle import FullInformationOracle
from ocmdp.components.constraint_filter import sample_feasible
from ocmdp.core.occupancy import OccupancyMeasure, scale_effective_horizon
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.metrics import expected_failure_rate, summarize
from ocmdp.eval.rollout import rollout
from ocmdp.eval.seeds import eval_seed
from ocmdp.objectives.pushforward_entropy import categorical_entropy, pushforward_entropy, similarity_sensitive_entropy
from ocmdp.theory.mean_field import MeanFieldModel
from ocmdp.theory.two_drug import stationary_failure_rate

# Between the default toy's equilibria under uniform mixing (0.5) and a pure policy (0.67),
# so the thresholds separate the two regimes instead of sitting on one of them.
FIXATION_THRESHOLD = 0.6
EFFICACY_THRESHOLD = 0.6


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "pending"
    detail: str
    values: dict = field(default_factory=dict)


VERIFICATION: dict[str, Callable[[], CheckResult]] = {}


def verification(fn: Callable[[], CheckResult]) -> Callable[[], CheckResult]:
    VERIFICATION[fn.__name__] = fn
    return fn


def _within(name: str, error: float, tol: float, detail: str, **values) -> CheckResult:
    """Pass iff error is finite and at most tol. NaN fails, so a broken computation never passes silently."""
    ok = bool(np.isfinite(error)) and error <= tol
    return CheckResult(name, "pass" if ok else "fail", detail, values)


def run_verification() -> list[CheckResult]:
    return [check() for check in VERIFICATION.values()]


@verification
def occupancy_decays_at_lambda() -> CheckResult:
    lam = 0.8
    occ = OccupancyMeasure(2, lam, init=np.array([1.0, 0.0]))
    err = float(np.max([abs(occ.update(np.array([0.0, 1.0]))[0] - lam**n) for n in range(1, 61)]))
    return _within("occupancy_decays_at_lambda", err, 1e-12, f"max |mu_0 - lam^n| over 60 periods = {err:.1e}", max_error=err)


@verification
def constraint_filter_never_infeasible() -> CheckResult:
    rng = np.random.default_rng(0)
    violations = 0
    for _ in range(10_000):
        n = int(rng.integers(2, 21))
        mask = rng.random(n) < rng.random()
        if not mask.any():
            mask[rng.integers(n)] = True
        violations += not mask[sample_feasible(rng.dirichlet(np.full(n, 0.3)), mask, rng)]
    env = ToyOCMDPEnv(ToyConfig.paired(contraindication_rate=0.5))
    for policy in (FixedGuideline(4, 0), uniform_mixing(4)):
        violations += int(rollout(env, policy, seed=0).violations.sum())
    return _within(
        "constraint_filter_never_infeasible", violations, 0,
        f"{violations} infeasible actions over 10 000 random cases and 2 000 environment decisions",
        violations=violations,
    )


@verification
def embedding_does_not_collapse() -> CheckResult:
    return CheckResult("embedding_does_not_collapse", "pending", "needs C2 (ocmdp.components.action_encoder)")


@verification
def analytical_two_drug_reproduced() -> CheckResult:
    cfg = ToyConfig.two_drug(rho=0.25, horizon_quarters=600, decisions_per_quarter=50)
    env = ToyOCMDPEnv(cfg)
    errors = {}
    for p in (1.0, 0.5, 0.3):
        policy = np.array([p, 1.0 - p])
        log = rollout(env, StationaryMixing(policy), seed=0)
        simulated = float(np.mean(log.fail_prob[300:] @ policy))
        errors[p] = abs(simulated - float(stationary_failure_rate(policy, env.coupling, env.kappa, env.delta)))
    worst = float(np.max(list(errors.values())))
    return _within(
        "analytical_two_drug_reproduced", worst, 5e-3,
        f"max |simulated - closed form| over p in (1, 0.5, 0.3) = {worst:.1e}", errors=errors,
    )


@verification
def mean_field_model_matches_env() -> CheckResult:
    """Fed the realised per-quarter action shares, the differentiable model must replay the environment exactly:
    the per-quarter expected failure the oracle optimises, the resistance path, and the final occupancy and pending."""
    cfg = ToyConfig.paired(rho=0.2, kappa=(0.10, 0.15), delta=(0.05, 0.04), delay=2, mu0=(0.7, 0.1, 0.1, 0.1))
    env = ToyOCMDPEnv(cfg)
    log = rollout(env, uniform_mixing(4), seed=3)
    n, Q = cfg.decisions_per_quarter, cfg.horizon_quarters
    shares = np.stack([np.bincount(log.actions[q * n : (q + 1) * n], minlength=4) / n for q in range(Q)])
    env_fails = np.array([np.mean(log.fail_prob[q, log.actions[q * n : (q + 1) * n]]) for q in range(Q)])
    fails, resistance, final = MeanFieldModel(cfg).simulate(torch.from_numpy(shares))
    end = env.state()
    errors = {
        "failure": float(np.max(np.abs(fails.numpy() - env_fails))),
        "resistance": float(np.max(np.abs(resistance.numpy() - log.resistance))),
        "occupancy": float(np.max(np.abs(final.occupancy.numpy() - end["occupancy"]))),
        "pending": float(np.max(np.abs(torch.stack(final.pending).numpy() - np.stack(end["pending"])))),
    }
    worst = float(np.max(list(errors.values())))
    return _within(
        "mean_field_model_matches_env", worst, 1e-12,
        "max discrepancy " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()), errors=errors,
    )


@verification
def env_factors_through_pushforward() -> CheckResult:
    """Premise of claim 2, in the environment itself: policies that differ only in how mass splits between
    drugs with identical cross-resistance rows produce identical resistance paths and failure rates."""
    env = ToyOCMDPEnv(ToyConfig.paired(rho=0.3))
    pairs = [
        (FixedGuideline(4, 0), FixedGuideline(4, 1)),
        (StationaryMixing([0.5, 0.0, 0.5, 0.0]), StationaryMixing([0.0, 0.5, 0.0, 0.5])),
    ]
    diffs, vacuous = [], False
    for a, b in pairs:
        la, lb = rollout(env, a, seed=0), rollout(env, b, seed=0)
        vacuous |= np.array_equal(la.actions, lb.actions)  # a pair giving the same drugs would test nothing
        diffs += [np.abs(la.resistance - lb.resistance).max(), abs(expected_failure_rate(la) - expected_failure_rate(lb))]
    worst = np.inf if vacuous else float(np.max(diffs))
    return _within("env_factors_through_pushforward", worst, 1e-12, f"max difference within a class = {worst:.1e}")


@verification
def similarity_entropy_generalises_pushforward() -> CheckResult:
    probs = torch.softmax(torch.randn(64, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64), -1)
    S = torch.tensor(ToyConfig.paired().cross_resistance, dtype=torch.float64)
    to_cluster = (similarity_sensitive_entropy(probs, S @ S.T) - pushforward_entropy(probs, S)).abs().max().item()
    to_shannon = (similarity_sensitive_entropy(probs, torch.eye(4, dtype=torch.float64)) - categorical_entropy(probs)).abs().max().item()
    return _within(
        "similarity_entropy_generalises_pushforward", float(np.max([to_cluster, to_shannon])), 1e-12,
        f"H_Z vs cluster form (Z = SS^T): {to_cluster:.1e}; vs Shannon (Z = I): {to_shannon:.1e}",
    )


@verification
def oracle_plan_dominates_heuristic_schedules() -> CheckResult:
    """In the model it plans with, the oracle must do at least as well as every heuristic it is compared with."""
    rows = {}
    for name, cfg in (("TwoDrug", ToyConfig.two_drug()), ("PairedDrugs", ToyConfig.paired())):
        env = ToyOCMDPEnv(cfg)
        env.reset(seed=0)
        oracle = FullInformationOracle(env)
        oracle.probs(None, {"quarter": 0})
        model = MeanFieldModel(cfg)
        quarters = np.arange(cfg.horizon_quarters)
        values = []
        for policy in comparators(env)[:-1]:
            schedule = np.stack([policy.probs(None, {"quarter": int(q)}) for q in quarters])
            values.append(float(model.simulate(torch.from_numpy(schedule))[0].mean()))
        rows[name] = {"oracle": oracle.predicted_failure_rate, "best_heuristic": float(np.min(values))}
    excess = float(np.max([v["oracle"] - v["best_heuristic"] for v in rows.values()]))
    return _within(
        "oracle_plan_dominates_heuristic_schedules", excess, 0.0,
        "mean-field failure rate, oracle vs best heuristic: "
        + "; ".join(f"{k} {v['oracle']:.4f} vs {v['best_heuristic']:.4f}" for k, v in rows.items()),
        **rows,
    )


def comparators(env: gym.Env) -> list[HeuristicPolicy]:
    """Table 1 comparators that exist so far, for a toy environment. The oracle is always last."""
    n = env.action_space.n
    policies: list[HeuristicPolicy] = [FixedGuideline(n, 0)]
    if n == 4:
        policies += [
            StationaryMixing([0.5, 0.5, 0.0, 0.0], name="within-class"),
            StationaryMixing([0.5, 0.0, 0.5, 0.0], name="cross-class"),
        ]
    policies += [Cycling(n, period=4), uniform_mixing(n), FullInformationOracle(env)]
    return policies


def comparator_table(env: gym.Env, seeds: tuple[int, ...]) -> dict[str, dict[str, dict]]:
    """policy -> metric -> aggregate over seeds. Regret is expected failures per episode above the oracle's, paired by seed.

    `seeds` are protocol seeds; episodes run on their evaluation seeds (ocmdp.eval.seeds.eval_seed).
    """
    assignment = env.unwrapped.coupling.mechanism_assignment()
    per_policy = {
        policy.name: [
            summarize(rollout(env, policy, eval_seed(s)), assignment, FIXATION_THRESHOLD, EFFICACY_THRESHOLD) for s in seeds
        ]
        for policy in comparators(env)
    }
    oracle_rows = per_policy["oracle"]
    n_decisions = env.unwrapped.config.horizon_quarters * env.unwrapped.config.decisions_per_quarter
    table = {}
    for name, rows in per_policy.items():
        for row, ref in zip(rows, oracle_rows):
            row["regret"] = n_decisions * (row["expected_failure_rate"] - ref["expected_failure_rate"])
        table[name] = {k: aggregate([row[k] for row in rows]) for k in rows[0]}
    return table


def lambda_misspecification(
    env: gym.Env, seeds: tuple[int, ...], factors: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
) -> dict[float, dict]:
    """Oracle planning with lam' whose effective horizon is `factor` times the true one."""
    lam = env.unwrapped.config.lam
    out = {}
    for factor in factors:
        lam_hat = scale_effective_horizon(lam, factor)
        oracle = FullInformationOracle(env, lam=None if factor == 1.0 else lam_hat)
        rates = [expected_failure_rate(rollout(env, oracle, eval_seed(s))) for s in seeds]
        out[factor] = {"lam": lam_hat, "expected_failure_rate": aggregate(rates)}
    return out


def aggregate(values) -> dict:
    """Mean and sample std over the finite values; `never` counts +inf (seeds where no mechanism fixed).

    NaN or -inf means a metric is broken, not that an event never happened, so it raises.
    """
    v = np.asarray(values, dtype=float)
    if np.isnan(v).any() or np.isneginf(v).any():
        raise ValueError(f"metric values contain NaN or -inf: {v}")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"mean": None, "std": None, "never": int(v.size)}
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return {"mean": float(finite.mean()), "std": std, "never": int(v.size - finite.size)}
