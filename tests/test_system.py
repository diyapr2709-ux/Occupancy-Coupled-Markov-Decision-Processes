"""System tests: the whole stack end to end, under fuzzed configurations and hostile inputs."""

import copy
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import ocmdp.envs  # noqa: F401  registers the Gymnasium ids
from ocmdp.agents.lagrangian import DualVariable
from ocmdp.baselines.heuristics import Cycling, FixedGuideline, StationaryMixing, uniform_mixing
from ocmdp.baselines.oracle import FullInformationOracle
from ocmdp.components.constraint_filter import project_probs
from ocmdp.core.coupling import CrossResistanceCoupling
from ocmdp.core.occupancy import OccupancyMeasure, scale_effective_horizon
from ocmdp.core.selection import check_rates
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.rollout import rollout
from ocmdp.eval.vv import VERIFICATION
from ocmdp.theory.mean_field import MeanFieldModel

ROOT = Path(__file__).resolve().parents[1]


def random_config(rng: np.random.Generator, **overrides) -> ToyConfig:
    """A random valid configuration, including degenerate corners (one drug, one mechanism, delay > horizon)."""
    n_drugs, n_mech = int(rng.integers(1, 7)), int(rng.integers(1, 4))
    C = rng.random((n_drugs, n_mech)) * (rng.random((n_drugs, n_mech)) < 0.6)
    C[np.arange(n_drugs), rng.integers(n_mech, size=n_drugs)] = rng.uniform(0.2, 1.0, size=n_drugs)
    kappa = rng.uniform(0.0, 0.6, size=n_mech)
    fields = dict(
        cross_resistance=tuple(map(tuple, C)),
        kappa=tuple(kappa),
        delta=tuple(rng.uniform(0.01, 1.0 - kappa)),
        lam=float(rng.uniform(0.0, 0.99)),
        delay=int(rng.integers(0, 4)),
        r0=tuple(rng.uniform(0.0, 1.0, size=n_mech)),
        mu0=tuple(rng.dirichlet(np.ones(n_drugs))),
        decisions_per_quarter=int(rng.integers(1, 6)),
        horizon_quarters=int(rng.integers(1, 9)),
        contraindication_rate=float(rng.choice([0.0, rng.uniform(0.0, 0.95)])),
        expected_reward=bool(rng.integers(2)),
        markov_observation=bool(rng.integers(2)),
    )
    return ToyConfig(**(fields | overrides))


def test_fuzzed_configs_keep_every_invariant_and_replay_exactly():
    rng = np.random.default_rng(2026)
    for trial in range(200):
        cfg = random_config(rng)
        env = ToyOCMDPEnv(cfg)
        obs, info = env.reset(seed=trial)
        policy, policy_rng = uniform_mixing(env.action_space.n), np.random.default_rng(trial)
        actions, steps, truncated = [], 0, False
        while not truncated:
            assert env.observation_space.contains(obs), (trial, cfg)
            assert info["action_mask"].any()
            assert np.all((info["resistance"] >= 0) & (info["resistance"] <= 1))
            assert np.all(info["occupancy"] >= 0) and math.isclose(info["occupancy"].sum(), 1.0)
            assert np.all((info["fail_prob"] >= 0) & (info["fail_prob"] <= 1))
            obs, reward, terminated, truncated, info = env.step(policy.act(obs, info, policy_rng))
            steps += 1
            actions.append(info["action"])
            assert -1.0 <= reward <= 0.0 and not terminated and not info["violation"]
            assert info["quarter"] == steps // cfg.decisions_per_quarter
        assert steps == cfg.decisions_per_quarter * cfg.horizon_quarters

        n = cfg.decisions_per_quarter
        shares = np.stack([np.bincount(actions[q * n : (q + 1) * n], minlength=env.action_space.n) / n for q in range(cfg.horizon_quarters)])
        _, resistance, final = MeanFieldModel(cfg).simulate(torch.from_numpy(shares))
        np.testing.assert_allclose(resistance[-1].numpy(), env.state()["resistance"], atol=1e-12, err_msg=str(cfg))
        np.testing.assert_allclose(final.occupancy.numpy(), env.state()["occupancy"], atol=1e-12, err_msg=str(cfg))


def test_oracle_runs_on_fuzzed_configs_without_violations():
    rng = np.random.default_rng(7)
    for trial in range(15):
        env = ToyOCMDPEnv(random_config(rng))
        oracle = FullInformationOracle(env, replan_every=int(rng.integers(1, 5)), steps=15, refine_steps=5)
        log = rollout(env, oracle, seed=trial)
        assert not log.violations.any()
        assert np.isfinite(oracle.predicted_failure_rate)


@pytest.mark.parametrize(
    "bad",
    [
        {"kappa": float("nan")},
        {"delta": float("inf")},
        {"r0": float("nan")},
        {"kappa": 0.7, "delta": 0.4},
        {"lam": 1.0},
        {"lam": float("nan")},
        {"delay": 1.5},
        {"delay": -1},
        {"decisions_per_quarter": 2.5},
        {"decisions_per_quarter": 0},
        {"horizon_quarters": True},
        {"contraindication_rate": float("nan")},
        {"resistance_noise": -0.1},
        {"resistance_noise": float("nan")},
        {"resistance_noise": float("inf")},
        {"mu0": (0.5, 0.6)},
        {"mu0": (1.0,)},
        {"cross_resistance": ((float("nan"), 1.0), (0.0, 1.0))},
        {"cross_resistance": ((0.0, 0.0), (0.0, 1.0))},
    ],
)
def test_invalid_configs_fail_at_construction(bad):
    with pytest.raises(ValueError):
        ToyOCMDPEnv(ToyConfig.two_drug(**bad) if "cross_resistance" not in bad else ToyConfig(**bad))


def test_step_is_guarded_outside_an_episode():
    env = ToyOCMDPEnv(ToyConfig.two_drug(horizon_quarters=1, decisions_per_quarter=1))
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    _, _, _, truncated, _ = env.step(0)
    assert truncated
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=1)
    env.step(np.int64(1))


@pytest.mark.parametrize("action", [1.5, -1, 2, "0", None, np.array([0, 1])])
def test_invalid_actions_are_rejected(action):
    env = ToyOCMDPEnv(ToyConfig.two_drug())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize(
    "call",
    [
        lambda: check_rates(float("nan"), 0.1),
        lambda: CrossResistanceCoupling([[float("nan"), 1.0]]),
        lambda: DualVariable(0.1, 1.0).observe(float("nan")),
        lambda: DualVariable(float("nan"), 1.0),
        lambda: project_probs([float("nan"), 1.0], [True, True]),
        lambda: project_probs([-0.5, 1.5], [True, True]),
        lambda: scale_effective_horizon(0.8, 0.0),
        lambda: scale_effective_horizon(0.8, float("nan")),
        lambda: OccupancyMeasure(2, 0.5, delay=1.0),
        lambda: FixedGuideline(2, drug=2),
        lambda: Cycling(2, period=0),
        lambda: Cycling(2, drugs=(0, 5)),
        lambda: StationaryMixing([float("nan"), 1.0]),
    ],
)
def test_hostile_inputs_raise_instead_of_propagating(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_vectorised_environments_run_full_episodes(mode):
    envs = gym.make_vec("ocmdp/PairedDrugs-v0", num_envs=3, vectorization_mode=mode)
    try:
        obs, _ = envs.reset(seed=0)
        steps_per_episode = 40 * 25
        for _ in range(steps_per_episode + 5):  # crosses an autoreset
            obs, rewards, terminated, truncated, _ = envs.step(envs.action_space.sample())
            assert envs.observation_space.contains(obs)
            assert np.all((rewards <= 0) & (rewards >= -1))
    finally:
        envs.close()


def test_rollouts_are_deterministic_given_the_seed():
    env = gym.make("ocmdp/PairedDrugs-v0")
    for make_policy in (lambda: uniform_mixing(4), lambda: FullInformationOracle(env, steps=30, refine_steps=10)):
        a, b = rollout(env, make_policy(), seed=5), rollout(env, make_policy(), seed=5)
        np.testing.assert_array_equal(a.actions, b.actions)
        np.testing.assert_array_equal(a.failures, b.failures)
        np.testing.assert_array_equal(a.resistance, b.resistance)
    assert not np.array_equal(rollout(env, uniform_mixing(4), seed=5).actions, rollout(env, uniform_mixing(4), seed=6).actions)


def test_deepcopied_and_pickled_envs_continue_identically():
    env = ToyOCMDPEnv(ToyConfig.paired(contraindication_rate=0.3, markov_observation=True))
    env.reset(seed=0)
    for a in [0, 1, 2, 3] * 10:
        env.step(a)
    clones = [copy.deepcopy(env), pickle.loads(pickle.dumps(env))]
    for a in [3, 2, 1, 0] * 20:
        expected = env.step(a)
        for clone in clones:
            got = clone.step(a)
            np.testing.assert_array_equal(got[0], expected[0])
            assert got[1:4] == expected[1:4]


def test_scales_to_the_proposal_action_set():
    """Around twenty classes and a handful of mechanisms, as in Section 4 of the proposal."""
    rng = np.random.default_rng(0)
    C = (rng.random((20, 5)) < 0.3) * rng.uniform(0.3, 1.0, (20, 5))
    C[np.arange(20), np.arange(20) % 5] = 1.0
    cfg = ToyConfig(cross_resistance=tuple(map(tuple, C)), kappa=0.1, delta=0.05)
    env = ToyOCMDPEnv(cfg)
    oracle = FullInformationOracle(env, steps=60, refine_steps=15)
    log = rollout(env, oracle, seed=0)
    uniform = rollout(env, uniform_mixing(20), seed=0)
    assert len(log.actions) == 1000
    assert log.fail_prob[log.decision_quarter, log.actions].mean() < uniform.fail_prob[uniform.decision_quarter, uniform.actions].mean()


def test_validate_cli_writes_a_complete_report(tmp_path):
    result = subprocess.run(
        [sys.executable, "experiments/validate.py", "--verification-only", "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (report,) = tmp_path.glob("vv_*.json")
    checks = {c["name"]: c["status"] for c in json.loads(report.read_text())["verification"]}
    assert set(checks) == set(VERIFICATION)
    assert "fail" not in checks.values()
