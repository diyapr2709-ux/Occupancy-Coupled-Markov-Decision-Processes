"""System tests: the whole stack end to end, under fuzzed configurations and hostile inputs."""

import copy
import csv
import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from ocmdp.core.selection import check_rates, failure_probability
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
        {"r0_spread": -0.1},
        {"mu0_spread": 1.5},
        {"mu0_spread": float("nan")},
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


def test_scales_to_twenty_drugs_and_five_mechanisms():
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


def _train_cli(*args, timeout=300):
    return subprocess.run([sys.executable, "experiments/train_agent.py", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _training_script():
    sys.path.insert(0, str(ROOT))
    try:
        from experiments import train_agent
    finally:
        sys.path.remove(str(ROOT))
    return train_agent


def test_training_env_takes_the_coupling_and_keeps_the_defaults_otherwise():
    script = _training_script()
    default = script.make_env(0.0).unwrapped.config
    assert (default.delay, default.lam) == (ToyConfig().delay, ToyConfig().lam)
    changed = script.make_env(0.2, (3, 1), delay=4, lam=0.5).unwrapped.config
    assert (changed.delay, changed.lam, changed.resistance_noise) == (4, 0.5, 0.2)
    assert changed.cross_resistance == ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def test_training_env_observes_the_pending_occupancy():
    env = _training_script().make_env(0.0)
    toy = env.unwrapped
    delay, n_mech, per_quarter = toy.config.delay, toy.coupling.n_mechanisms, toy.config.decisions_per_quarter
    assert toy.config.markov_observation and delay >= 1
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 * n_mech + 1 + n_mech * (1 + delay),)
    np.testing.assert_array_equal(obs[-delay * n_mech :], 0.0)  # nothing prescribed yet
    for _ in range(per_quarter):
        obs, *_ = env.step(0)
    # the quarter just closed is not yet in the occupancy the kernel sees, so it must be in the observation
    np.testing.assert_allclose(obs[-n_mech:], toy.coupling.matrix[0])


def test_committed_per_quarter_gives_one_drug_per_quarter_and_is_the_oracles_choice():
    from ocmdp.baselines.oracle import FullInformationOracle
    from ocmdp.eval.rollout import rollout

    script = _training_script()
    env = script.make_env(0.0)
    base = FullInformationOracle(env, steps=20, refine_steps=5, n_random=0)
    committed = script.committed_per_quarter(FullInformationOracle(env, steps=20, refine_steps=5, n_random=0))
    assert committed.name == "oracle[one drug per quarter]"
    log = rollout(env, committed, seed=10_000)
    given = []
    for q in np.unique(log.decision_quarter):
        drugs = np.unique(log.actions[log.decision_quarter == q])
        assert len(drugs) == 1
        given.append(int(drugs[0]))

    class Recording(FullInformationOracle):
        def probs(self, obs, info):
            p = super().probs(obs, info)
            self.choices.append(int(np.argmax(p)))
            return p

    recording = Recording(env, steps=20, refine_steps=5, n_random=0)
    recording.choices = []
    replay = script.committed_per_quarter(recording)
    rollout(env, replay, seed=10_000)
    assert given == recording.choices and len(set(given)) > 1  # the oracle's own choice each quarter, and it rotates


def test_robust_summaries_match_their_definitions():
    script = _training_script()
    assert script.iqm([5.0, 1.0, 2.0, 3.0, 100.0]) == pytest.approx(10 / 3)  # sorted 1 2 3 5 100, drops one from each end
    assert script.iqm([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) == pytest.approx(4.5)  # drops two of eight from each end
    assert script.probability_of_improvement([0.1, 0.2], [0.2, 0.3]) == pytest.approx((1 + 1 + 0.5 + 1) / 4)
    lo, hi = script.bootstrap_ci95([0.01, 0.02, 0.03, 0.04])
    assert 0.01 <= lo < 0.025 < hi <= 0.04
    assert script.bootstrap_ci95([0.5, 0.5, 0.5]) == [0.5, 0.5]
    seed_only = script.bootstrap_ci95([0.01, 0.03])
    both = script.two_level_bootstrap_ci95([[0.0, 0.02], [0.02, 0.04]])  # seed means 0.01 and 0.03, with spread inside each seed
    assert both[0] <= seed_only[0] and both[1] >= seed_only[1]
    assert script.two_level_bootstrap_ci95([[0.2, 0.2], [0.2, 0.2]]) == [0.2, 0.2]
    # seeds agree and only a shared episode effect varies: the interval must carry that effect at full width,
    # as an episode bootstrap of it does, not shrunk by averaging independent episode draws over seeds
    effect = np.random.default_rng(1).normal(0.0, 0.01, size=20)
    crossed = script.two_level_bootstrap_ci95(np.tile(effect, (10, 1)))
    episodes_only = script.bootstrap_ci95(effect)
    assert 0.8 < (crossed[1] - crossed[0]) / (episodes_only[1] - episodes_only[0]) < 1.25
    assert script.bootstrap_ci95(np.tile(effect, (10, 1)).mean(axis=1)) == [pytest.approx(effect.mean())] * 2


def test_soft_returns_discount_later_rewards_and_entropy_bonuses_and_stop_at_the_horizon():
    reward, bonus = np.array([[1.0, 2.0, 3.0]]), np.array([[9.0, 0.5, 0.25]])
    # the bonus of the deciding quarter itself is not part of the value of the drug chosen there
    np.testing.assert_allclose(_training_script().soft_returns(reward, bonus, 0.5), [[3.0625, 3.625, 3.0]])


def test_quarterly_and_discounted_failure_follow_the_drugs_given():
    script = _training_script()
    log = SimpleNamespace(
        fail_prob=np.array([[0.1, 0.5], [0.2, 0.6], [0.9, 0.9]]),
        decision_quarter=np.array([0, 0, 1, 1]),
        actions=np.array([0, 1, 1, 1]),
    )
    failure = script.quarterly_failure(log)
    np.testing.assert_allclose(failure, [0.3, 0.6])
    assert script.discounted_failure_rate(failure, 1.0) == pytest.approx(0.45)
    assert script.discounted_failure_rate(failure, 0.5) == pytest.approx((0.3 + 0.5 * 0.6) / 1.5)


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "--tier"),
        (["--tier", "dev", "--seeds", "100"], "inside the dev tier"),
        (["--tier", "val", "--seeds", "100", "100"], "distinct"),
        (["--tier", "dev", "--episodes", "101"], "--episodes"),
        (["--tier", "test", "--seeds", "1000"], "--prereg"),
        (["--tier", "test", "--seeds", "1000", "--prereg", "preregistrations/missing.yaml"], "--prereg"),
        (["--tier", "val", "--seeds", "100"], "--platform"),
        (["--tier", "dev", "--checkpoint-every", "6000"], "--checkpoint-every"),
        (["--tier", "val", "--seeds", "100", "--platform", "x", "--checkpoint-every", "1000"], "--checkpoint-every"),
        (["--tier", "dev", "--class-sizes", "4"], "--class-sizes"),
        (["--tier", "dev", "--class-sizes", "3", "0"], "--class-sizes"),
        (["--tier", "dev", "--lam", "1.0"], "--lam"),
        (["--tier", "dev", "--delay", "-1"], "--delay"),
    ],
)
def test_train_cli_rejects_runs_outside_the_seed_protocol(args, message, tmp_path):
    result = _train_cli(*args, "--out", str(tmp_path))
    assert result.returncode == 2 and message in result.stderr
    assert not list(tmp_path.iterdir())


def test_train_cli_refuses_a_preregistration_outside_the_repository(tmp_path):
    prereg = tmp_path / "outside.yaml"
    prereg.write_text("slug: outside\n")
    result = _train_cli("--tier", "test", "--seeds", "1000", "--prereg", str(prereg), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "inside the repository" in result.stderr
    assert not (tmp_path / "out" / "launches").exists()


def _protocol():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import check_dev_protocol
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    return check_dev_protocol


def _step1_report(sizes, quarters, rates, same_weights=True, learned=10):
    """A fixed-temperature-0 report whose agents fail at `rates[seed]` on two episodes; uniform 0.40, replanner 0.30."""
    runs = []
    for seed, rate in enumerate(rates):
        for variant in ("oc-sac", "sac"):
            final = f"final-{seed}-{quarters}" + ("" if same_weights or variant == "oc-sac" else "-other")
            runs.append({"variant": variant, "sigma": 0.0, "seed": seed, "expected_failure_rate": [rate, rate],
                         "parameter_sha256": {"initial": f"init-{seed}", "after_first_window": f"first-{seed}", "final": final}})  # fmt: skip
    config = {"temperature": "fixed", "alpha": 0.0, "total_quarters": quarters}
    return {"_path": f"train_{quarters}.json", "class_structure": {"class_sizes": list(sizes)}, "runs": runs,
            "sac_config": {"oc-sac": config, "sac": config},
            "baselines": {"0.0": {"uniform": {"expected_failure_rate": [0.40, 0.40]}, "oracle[replans every quarter]": {"expected_failure_rate": [0.30, 0.30]}}},
            "summary": {"0.0": {"oc-sac": {"runs_learned": learned}}}}  # fmt: skip


def test_step1_picks_the_first_converged_length_and_fails_without_one():
    protocol = _protocol()
    noise = [0.001 * ((s % 3) - 1) for s in range(10)]
    flat = {q: [0.32 + n for n in noise] for q in (12_000, 24_000, 48_000)}
    reports = [_step1_report(sizes, q, flat[q]) for sizes in ((2, 2), (8, 5, 4, 2, 1)) for q in flat]
    lines, result = protocol.step1(reports)
    assert result == {"failed": False, "bitwise": True, "length": 12_000}, lines
    improving = {12_000: [0.36 + n for n in noise], 24_000: [0.32 + n for n in noise], 48_000: [0.32 + n for n in noise]}
    reports = [_step1_report(sizes, q, improving[q]) for sizes in ((2, 2), (8, 5, 4, 2, 1)) for q in improving]
    assert protocol.step1(reports)[1]["length"] == 24_000  # 12k -> 24k still gains 0.4 of the replanner's share
    still = {12_000: [0.38 + n for n in noise], 24_000: [0.35 + n for n in noise], 48_000: [0.32 + n for n in noise]}
    reports = [_step1_report(sizes, q, still[q]) for sizes in ((2, 2), (8, 5, 4, 2, 1)) for q in still]
    lines, result = protocol.step1(reports)
    assert result["failed"] and result["length"] is None and "STOP" in lines[-1]
    reports = [_step1_report((2, 2), q, flat[q], same_weights=q != 24_000) for q in flat]
    lines, result = protocol.step1(reports)
    assert result["failed"] and not result["bitwise"] and "FAIL" in lines[0]


def _step2_report(alpha_path, alpha_eval, entropy_last, fixed=None):
    rng = {"floor": 0.0, "maximum": math.log(2), "target": 0.89 * math.log(2)}
    cfg = {"temperature": "fixed" if fixed is not None else "auto", "alpha": fixed or 0.0, "log_alpha_range": [-12.0, 2.0], "total_quarters": 12_000}
    runs = [{"variant": v, "sigma": 0.0, "seed": 0, "alpha_path": alpha_path, "alpha_at_evaluation": alpha_eval, "entropy_path": [entropy_last],
             "entropy_range": rng, "policy_entropy": {"action": 0.05 + math.log(2), "within_class": math.log(2)}} for v in ("oc-sac", "sac")]  # fmt: skip
    for run in runs:
        if run["variant"] == "sac":
            run["policy_entropy"] = {"action": 0.05, "within_class": 0.0}
    return {"_path": "train_dev.json", "class_structure": {"class_sizes": [2, 2]}, "runs": runs, "sac_config": {"oc-sac": cfg, "sac": cfg}}


def test_step2_holds_learned_temperatures_to_their_clamp_target_and_settling():
    protocol = _protocol()
    target = 0.89 * math.log(2)
    settled = protocol.step2([_step2_report([0.2, 0.15, 0.12, 0.11], 0.11, target)])[1]
    assert settled == {"failed": False, "widen_upward": False}
    assert protocol.step2([_step2_report([0.2, 0.15, 0.12, math.exp(-12.0)], 0.11, target)])[1]["failed"]  # reached the clamp
    assert protocol.step2([_step2_report([0.2, 0.15, 0.12, 0.11], 0.11, 0.3)])[1]["failed"]  # far from its target
    assert protocol.step2([_step2_report([0.2, 0.15, 0.12, 0.11, 0.11, 0.11, 2.0, 0.11], 0.11, target)])[1]["failed"]  # swinging in the last quarter
    assert not protocol.step2([_step2_report([0.1] * 4, 0.1, 0.0, fixed=0.1)])[1]["failed"]
    assert protocol.step2([_step2_report([0.1, 0.1, 0.2, 0.1], 0.1, 0.0, fixed=0.1)])[1]["failed"]
    assert protocol.step2([_step2_report([3.0] * 4, 3.0, 0.0, fixed=3.0)])[1]["widen_upward"]  # class entropy 0.05 < 0.9 log 2


REPORTS_WITH_A_HOME_PATH = {  # committed before paths were recorded relative to the repository; runs/ is never edited
    "runs/phase/phase_20261123_184302_13496.json": "d89bc092d7eeed6b3519904eec858530d3d8d4e27158182a61b980fe9e44c345",
    "runs/phase/phase_20261123_190013_15752.json": "b1bda038b5aaa3bd780265e10f36f3e4836e1cbc8e4c2822048f6785207891fd",
    "runs/train/train_20261116_190634_8365.json": "2549945dff723963c918f747ddbb68c7d8bf6677ee2a610b28b418907a3c1dca",
    "runs/train/train_20261116_192212_10213.json": "a7c6272ae21875216a9fb184ba7e2cd036481594d988701b3429a23fca5bca31",
    "runs/train/train_20261116_233400_32604.json": "074d75abfc79551ce3de497fcc28b5dbdf2e258d7f1f99abd8fe59c41b004290",
    "runs/train/train_20261123_184321_12020.json": "cc2b74fe66fc5ac0a6828e410612fc50502b5833c8f0746e75f0a9c44bda8263",
    "runs/train/train_20261123_184953_15052.json": "412c81f0b2d59775aca56433be30a7ff47d2c2a2de619ecbd1758d3ff36cc314",
    "runs/train/train_20261123_190717_20498.json": "2ea2673aaa33e3220465407fca3e833a46ff913d52808851c3fc62ec86d93785",
    "runs/train/train_20261123_190750_20538.json": "bacadf036cf15d0f635b46a6e293716feac200cfa4959205ccb7c6a5f545fa89",
    "runs/train/train_20261123_194806_29133.json": "3e658011cf5c6e6bfea147afe0eeeea3601c8875a0fbc1bf6104e5862b1d5a06",
}


def test_no_file_carries_a_home_directory_path_beyond_the_known_reports():
    # a home directory names its owner; both the path form and the dashed form a tool may record are caught
    import re

    pattern = re.compile(r"(/Users/|/home/)[A-Za-z0-9._-]+/|-Users-[A-Za-z0-9._]+-")
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True)
    offending = []
    for name in listed.stdout.split():
        path = ROOT / name
        if path.suffix == ".pt" or not path.is_file():
            continue
        if pattern.search(path.read_text(errors="ignore")) and REPORTS_WITH_A_HOME_PATH.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            offending.append(name)
    assert offending == []


def test_a_seed_file_cannot_relabel_the_test_seeds(tmp_path):
    relabelled = tmp_path / "seeds.yaml"
    relabelled.write_text("dev: [1000, 1099]\nval: [100, 199]\ntest: [2000, 2099]\n")
    result = _train_cli("--tier", "dev", "--seeds", "1000", "--seeds-file", str(relabelled), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "overlaps the repository's test seeds" in result.stderr
    result = _train_cli("--tier", "test", "--seeds", "2000", "--seeds-file", str(relabelled), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "the repository's seeds.yaml" in result.stderr
    assert not list((tmp_path / "out").iterdir())  # refused before any launch record or report


def _refusal(capsys, check, *args) -> str:
    import argparse

    with pytest.raises(SystemExit):
        check(argparse.ArgumentParser(), *args)
    return capsys.readouterr().err


def test_test_tier_checks_refuse_in_order_and_cannot_borrow_an_approval(tmp_path, capsys, monkeypatch):
    script = _training_script()
    repo = tmp_path / "repo"
    (repo / "preregistrations").mkdir(parents=True)
    (repo / "preregistrations" / "planned.yaml").write_text("slug: planned\nplatform: test-host\n")
    (repo / "preregistrations" / "unplaced.yaml").write_text("slug: unplaced\n")
    (repo / "notes.yaml").write_text("not a preregistration\n")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr(script, "ROOT", repo)
    planned, out = repo / "preregistrations" / "planned.yaml", tmp_path / "out"
    clean, dirty = {"dirty": False}, {"dirty": True}
    assert "file in preregistrations/" in _refusal(capsys, script.check_test_launch, repo / "notes.yaml", clean, out, "test-host")
    assert "clean, committed tree" in _refusal(capsys, script.check_test_launch, planned, dirty, out, "test-host")
    assert "must name the platform" in _refusal(capsys, script.check_test_launch, repo / "preregistrations" / "unplaced.yaml", clean, out, "test-host")
    assert "preregistered for platform test-host, not rtx-1" in _refusal(capsys, script.check_test_launch, planned, clean, out, "rtx-1")
    assert "approvals/planned.ok" in _refusal(capsys, script.check_test_launch, planned, clean, out, "test-host")
    (out / "launches").mkdir(parents=True)
    (out / "launches" / "train_earlier.json").write_text(json.dumps({"launch_id": "train_earlier", "tier": "test", "prereg_slug": "planned"}))
    assert script.earlier_test_launches("planned", out) == ["train_earlier"]
    assert "already used by launch train_earlier" in _refusal(capsys, script.check_test_launch, planned, clean, out, "test-host")
    link = repo / "preregistrations" / "approved.yaml"
    link.symlink_to(planned)
    assert "not a link" in _refusal(capsys, script.check_test_launch, link, clean, out, "test-host")
    planned.write_text("slug: planned, edited after the commit\n")
    assert "committed, unmodified" in _refusal(capsys, script.check_test_launch, planned, clean, out, "test-host")


def test_training_jobs_keep_both_agents_of_a_seed_in_one_job():
    script = _training_script()
    args = SimpleNamespace(quarters=200, episodes=2, diagnose_gap=False, checkpoint_every=None, sigmas=[0.0, 0.2], seeds=[0, 1])
    assert [(job[0], job[1]) for job in script.training_jobs(args, {}, 0, ((2, 2), None, None))] == [(s, seed) for s in (0.0, 0.2) for seed in (0, 1)]
    calls = []
    original = script.train_and_evaluate
    script.train_and_evaluate = lambda job: calls.append(job[:3]) or {"variant": job[0]}
    try:
        runs = script.train_and_evaluate_pair((0.2, 7, 200, 2, {}, 0, ((2, 2), None, None), False, None))
    finally:
        script.train_and_evaluate = original
    assert calls == [("oc-sac", 0.2, 7), ("sac", 0.2, 7)] and [r["variant"] for r in runs] == ["oc-sac", "sac"]


def test_train_cli_rejects_overlapping_tiers(tmp_path):
    seeds_file = tmp_path / "seeds.yaml"
    seeds_file.write_text("dev: [0, 99]\nval: [50, 199]\ntest: [1000, 1099]\n")
    result = _train_cli("--tier", "dev", "--seeds-file", str(seeds_file), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "overlap" in result.stderr


def test_train_cli_writes_a_manifest_on_the_tiers_own_episodes(tmp_path):
    small = ["--sigmas", "0.0", "--quarters", "200", "--episodes", "2", "--workers", "2",
             "--set", "learning_starts=64", "batch_size=32", "hidden_dim=16", "dual_update_every=40"]  # fmt: skip
    result = _train_cli("--tier", "val", "--seeds", "100", "--platform", "test-host", *small, "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    (path,) = tmp_path.glob("train_*.json")
    report = json.loads(path.read_text())
    assert report["args"]["tier"] == "val" and report["args"]["seeds"] == [100]
    assert report["report_schema"] == 2 and report["launch_id"] == path.stem and report["platform"] == "test-host"
    assert report["args"]["seeds_file"] == "seeds.yaml" and platform.node() not in path.read_text()  # no machine's names or directories
    assert report["args"]["out"] == f"<outside the repository>/{tmp_path.name}"
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import verify_checkpoints
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    assert verify_checkpoints.verify(path, load=True) == []
    first = tmp_path / report["runs"][0]["checkpoints"][0]["path"]
    intact = first.read_bytes()
    first.write_bytes(intact[:-1] + b"x")
    assert verify_checkpoints.verify(path) == [f"size or SHA-256 differs from the report: {report['runs'][0]['checkpoints'][0]['path']}"]
    first.write_bytes(intact)
    launch = json.loads((tmp_path / "launches" / f"{path.stem}.json").read_text())
    assert launch["launch_id"] == path.stem and launch["tier"] == "val" and launch["platform"] == "test-host"
    assert launch["git"] == report["git"] and launch["prereg_slug"] is None and launch["prereg_sha256"] is None
    oc_sac, sac = (next(r for r in report["runs"] if r["variant"] == v) for v in ("oc-sac", "sac"))
    assert oc_sac["parameter_sha256"]["initial"] == sac["parameter_sha256"]["initial"]  # one seed, identical initial weights
    assert [r["variant"] for r in report["runs"]] == ["oc-sac", "sac"]
    assert report["evaluation_episode_seeds"] == [10_100, 10_101]
    assert set(report["git"]) == {"commit", "dirty", "code_diff_sha256", "untracked_code"}
    assert report["sac_config"]["oc-sac"]["entropy"] == "pushforward" and report["sac_config"]["sac"]["entropy"] == "action"
    assert set(report["baselines"]["0.0"]) >= {"uniform", "oracle", "oracle[replans every quarter]", "oracle[one drug per quarter]"}
    assert report["backend"]["name"] == "local" and report["backend"]["workers"] == 2
    assert report["instance"]["markov_observation"] is True and "resistance_noise" not in report["instance"]
    assert {r["variant"] for r in report["runs"]} == {"oc-sac", "sac"}
    assert report["class_structure"] == {"class_sizes": [2, 2], "imbalance_kl_nats": pytest.approx(0.0, abs=1e-12)}
    assert report["instance"]["cross_resistance"] == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

    # every agent is saved next to the report, and the saved agent is the one that was evaluated
    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, load_checkpoint, parameter_sha256

    script = _training_script()
    for run in report["runs"]:
        (checkpoint,) = run["checkpoints"]
        blob = (tmp_path / checkpoint["path"]).read_bytes()
        assert checkpoint["path"] == f"checkpoints/val/{path.stem}/{run['variant']}_sigma0.0_seed100_q200.pt"
        assert checkpoint["decisions"] == 200 and checkpoint["bytes"] == len(blob)
        assert checkpoint["sha256"] == hashlib.sha256(blob).hexdigest()
        agent, decisions = load_checkpoint(blob)
        config = {k: tuple(v) if isinstance(v, list) else v for k, v in report["sac_config"][run["variant"]].items()}
        assert decisions == 200 and agent.cfg == SACConfig(**config)
        assert parameter_sha256(agent) == run["parameter_sha256"]["final"]
        assert run["entropy_range"] == pytest.approx({"floor": agent.entropy_floor, "maximum": agent.entropy_max, "target": agent.entropy_target})
        env = script.make_env(0.0)
        greedy = QuarterlyChoicePolicy(lambda obs, rng: agent.choose(obs, rng, greedy=True), env.action_space.n, run["variant"])
        again = script._episode_stats(env, script._rollouts(env, greedy, 2, 100), agent.cfg.gamma)
        assert again["expected_failure_rate"] == run["argmax"]["expected_failure_rate"]


def test_modal_backend_pins_the_local_package_versions_and_ships_the_code_it_runs():
    pytest.importorskip("modal")
    from importlib.metadata import version

    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import modal_backend
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    assert modal_backend.PINNED == {name: version(name) for name in ("numpy", "torch", "gymnasium")}
    assert modal_backend.description()["name"] == "modal" and "train_agent" in modal_backend.SCRIPTS
    with pytest.raises(ValueError, match="must call into"):
        modal_backend.run_jobs([("os", "system", ("true",))])
    assert modal_backend.PYTHON == ".".join(map(str, sys.version_info[:2]))


def test_train_cli_builds_unequal_classes_and_logs_the_entropy_spent_inside_them(tmp_path):
    small = ["--sigmas", "0.0", "--quarters", "200", "--episodes", "2", "--workers", "2",
             "--set", "learning_starts=64", "batch_size=32", "hidden_dim=16", "dual_update_every=40"]  # fmt: skip
    result = _train_cli("--tier", "val", "--seeds", "100", "--platform", "test-host", "--class-sizes", "3", "1", "--checkpoint-every", "100", *small, "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    (path,) = tmp_path.glob("train_*.json")
    report = json.loads(path.read_text())
    assert report["instance"]["cross_resistance"] == [[1.0, 0.0]] * 3 + [[0.0, 1.0]]
    kl = math.log(2) + 0.75 * math.log(0.75) + 0.25 * math.log(0.25)
    assert report["class_structure"] == {"class_sizes": [3, 1], "imbalance_kl_nats": pytest.approx(kl)}
    ranges = {r["variant"]: r["entropy_range"] for r in report["runs"]}
    # three tied drugs and a singleton: drug entropy can reach 0 here, class entropy reaches at most log 2
    assert ranges["sac"]["floor"] == pytest.approx(0.0, abs=1e-6) and ranges["sac"]["maximum"] == pytest.approx(math.log(4), abs=1e-4)
    assert ranges["oc-sac"]["floor"] == pytest.approx(0.0, abs=1e-6) and ranges["oc-sac"]["maximum"] == pytest.approx(math.log(2), abs=1e-4)
    for run in report["runs"]:
        assert [c["decisions"] for c in run["checkpoints"]] == [100, 200]
        spent = run["policy_entropy"]
        # tied drugs: the entropy inside classes is q_big log 3, at most log 3 whatever mass the big class gets
        assert 0.0 <= spent["within_class"] <= min(spent["action"], math.log(3)) + 1e-6
    summary = report["summary"]["0.0"]
    paired = summary["paired comparison"]
    assert paired["seeds"] == [100] and paired["seeds_where_oc_sac_fails_less"] in (0, 1)
    assert 0.0 <= paired["probability_oc_sac_fails_less"] <= 1.0
    lo, hi = paired["two_level_bootstrap_ci95_of_mean_difference"]
    assert lo <= summary["oc-sac minus sac (paired by seed)"]["mean"] <= hi
    for variant in ("oc-sac", "sac"):
        assert set(summary[variant]["policy_entropy"]) == {"action", "within_class"}
        critic = summary[variant]["critic"]
        assert 0.0 <= critic["critic_ranks_best_class_first"] <= 1.0 and critic["critic_choice_regret"] >= -1e-9
        assert critic["mean_gap_best_to_second"] >= 0.0
        versus = summary[variant]["minus_baseline_paired_by_episode"]
        assert set(versus) == set(summary["baselines"])
        agent = [r for r in report["runs"] if r["variant"] == variant]
        for name, baseline in report["baselines"]["0.0"].items():
            expected = np.mean([np.mean(np.asarray(r["expected_failure_rate"]) - baseline["expected_failure_rate"]) for r in agent])
            assert versus[name]["mean"] == pytest.approx(expected, abs=1e-12)
    assert set(summary["discounted_failure_rate"]) == set(summary["baselines"]) | {"oc-sac", "sac"}
    for run in report["runs"]:
        assert len(run["discounted_failure_rate"]) == len(run["expected_failure_rate"]) == 2 == len(run["final_fail_prob"])
        assert run["critic"]["states"] == 2 * 40 and run["critic"]["actor_choice_regret"] >= -1e-9
        assert len(run["alpha_path"]) == len(run["xi_path"]) == len(run["entropy_path"]) == (200 - 64 + 1) // 40  # one per window
        assert all(run["entropy_range"]["floor"] - 1e-6 <= h <= run["entropy_range"]["maximum"] + 1e-6 for h in run["entropy_path"])
        assert math.exp(-12.0) * (1 - 1e-6) <= run["alpha_at_evaluation"] <= math.exp(2.0) * (1 + 1e-6)
        assert set(run["host"]) == {"machine", "cpu", "simd", "numpy_cpu_features", "torch_threads", "cuda_visible_devices",
                                    "cuda_available", "pid", "job_in_process", "modal"}  # fmt: skip
        assert run["host"]["torch_threads"] == 1 and run["host"]["job_in_process"] >= 1 and run["host"]["numpy_cpu_features"]
        assert set(run["parameter_sha256"]) == {"initial", "after_first_window", "final"}


def _phase_cli(*args, timeout=300):
    return subprocess.run([sys.executable, "experiments/sweep_phase_diagram.py", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "--tier"),
        (["--tier", "dev", "--lams", "1.0"], "--lams"),
        (["--tier", "dev", "--delays", "-1"], "--delays"),
        (["--tier", "dev", "--episodes", "1"], "--episodes"),
        (["--tier", "test"], "--prereg"),
        (["--tier", "dev", "--kappa", "0.6", "--delta", "0.5"], "--kappa/--delta"),
    ],
)
def test_phase_cli_rejects_invalid_grids(args, message, tmp_path):
    result = _phase_cli(*args, "--out", str(tmp_path))
    assert result.returncode == 2 and message in result.stderr
    assert not list(tmp_path.iterdir())


def test_phase_cli_pairs_every_policy_on_the_tiers_episodes_and_its_contrasts_recompute(tmp_path):
    grid = ["--delays", "0", "2", "--lams", "0.5", "--sigmas", "0.0", "0.3", "--horizon", "6", "--episodes", "2", "--workers", "4",
            "--kappa", "0.2", "--delta", "0.1"]  # fmt: skip
    result = _phase_cli("--tier", "val", "--platform", "test-host", *grid, "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    (path,) = tmp_path.glob("phase_*.json")
    report = json.loads(path.read_text())
    assert report["evaluation_episode_seeds"] == [10_100, 10_101] and report["backend"]["name"] == "local"
    assert report["launch_id"] == path.stem and (tmp_path / "launches" / f"{path.stem}.json").is_file()
    assert report["instance"]["horizon_quarters"] == 6 and report["instance"]["cross_resistance"] == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert (report["instance"]["kappa"], report["instance"]["delta"]) == (0.2, 0.1)
    assert len(report["cells"]) == 4 and len(report["results"]) == 4 * len(report["policies"])
    assert all(len(r["final_fail_prob"]) == 2 and len(r["final_fail_prob"][0]) == 4 for r in report["results"])
    rates = {(r["delay"], r["lam"], r["sigma"], r["policy"]): r["expected_failure_rate"] for r in report["results"]}
    for cell in report["cells"]:
        key = (cell["delay"], cell["lam"], cell["sigma"])
        for name, (a, b) in report["contrasts"].items():
            diff = np.array(rates[(*key, a)]) - np.array(rates[(*key, b)])
            assert cell[name]["mean"] == pytest.approx(diff.mean(), abs=1e-12)
        if cell["sigma"] == 0.0:
            # no noise: the model is exact, so replanning every quarter cannot lose to committing at the start
            assert cell["value_of_feedback"]["mean"] >= -1e-9


def test_phase_plot_draws_every_cell_with_its_interval_and_tabulates_it(tmp_path):
    cells = [
        {"delay": d, "lam": lam, "sigma": s, "value_of_feedback": {"mean": 0.01 * s + 0.001 * d, "ci95": 0.002}}
        for d in (0, 2) for lam in (0.5, 0.9) for s in (0.0, 0.2)
    ]  # fmt: skip
    results = [
        {"delay": c["delay"], "lam": c["lam"], "sigma": c["sigma"], "policy": policy, "expected_failure_rate": rates}
        for c in cells
        for policy, rates in (("planned-once", [0.40, 0.42, 0.41]), ("replanned-every-1", [0.39, 0.40, 0.40]))
    ]  # fmt: skip
    report = {"cells": cells, "results": results, "contrasts": {"value_of_feedback": ["planned-once", "replanned-every-1"]},
              "evaluation_episode_seeds": [10_000, 10_001, 10_002], "args": {"tier": "dev"}}  # fmt: skip
    (tmp_path / "phase_x.json").write_text(json.dumps(report))
    predictions = [{"delay": c["delay"], "lam": c["lam"], "sigma": c["sigma"], "value_of_feedback": 0.0} for c in cells]
    (tmp_path / "pred.json").write_text(json.dumps(predictions))
    result = subprocess.run(
        [sys.executable, "experiments/plot_phase_diagram.py", str(tmp_path / "phase_x.json"), "--predictions", str(tmp_path / "pred.json"), "--out", str(tmp_path / "fig")],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    svg = (tmp_path / "fig" / "phase_x_value_of_feedback.svg").read_text()
    assert svg.count("<circle") == 2 + 2 * len(cells)  # legend, then a filled marker and a hollow prediction per cell
    assert svg.count('stroke-width="1.5"') == len(cells)  # one interval whisker per measured cell
    with open(tmp_path / "fig" / "phase_x_value_of_feedback.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(cells) and {float(r["predicted"]) for r in rows} == {0.0}
    # recomputed from the raw episodes, not copied from the summaries: differences 0.01, 0.02, 0.01
    assert float(rows[0]["mean"]) == pytest.approx(0.04 / 3) and float(rows[0]["ci95"]) == pytest.approx(4.302653 * (0.01 / 3**0.5) / 3**0.5, rel=1e-6)


def _stats():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import stats
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    return stats


@pytest.mark.parametrize(
    "probability, df, quantile",
    [(0.975, 1, 12.706205), (0.975, 4, 2.776445), (0.975, 9, 2.262157), (0.975, 39, 2.022691), (0.975, 200, 1.971896), (0.95, 4, 2.131847)],
)
def test_t_quantile_matches_tables(probability, df, quantile):
    assert _stats().t_quantile(probability, df) == pytest.approx(quantile, abs=2e-6)


def test_t_quantile_reaches_heavy_tails_and_refuses_lower_probabilities():
    stats = _stats()
    assert stats.t_quantile(0.9999, 1) == pytest.approx(math.tan(math.pi * (0.9999 - 0.5)), rel=1e-6)  # Cauchy: about 3183
    for probability in (0.4, 0.5, 1.0):
        with pytest.raises(ValueError):
            stats.t_quantile(probability, 5)


def test_equivalence_needs_the_whole_90_percent_interval_inside_the_margin():
    stats = _stats()
    close = stats.equivalent([0.001, -0.001, 0.0005, -0.0005, 0.0], margin=0.005)
    assert close["equivalent"] and -0.005 < close["ci90"][0] < close["ci90"][1] < 0.005
    assert not stats.equivalent([0.02, 0.021, 0.019, 0.02], margin=0.005)["equivalent"]
    assert not stats.equivalent([0.004, -0.004, 0.004, -0.004], margin=0.005)["equivalent"]  # centred on zero, too wide


def test_counterfactual_values_tie_inside_a_class_and_stop_at_the_horizon():
    import copy as copy_module

    from ocmdp.agents.oc_sac import Agent, SACConfig

    script = _training_script()
    env = script.make_env(0.0, (3, 1))
    toy = env.unwrapped
    cfg = SACConfig(total_quarters=240, learning_starts=48, batch_size=32, hidden_dim=16)
    obs, _ = env.reset(seed=10_000)
    torch.manual_seed(0)
    agent = Agent(obs.shape[0], torch.tensor(toy.coupling.matrix, dtype=torch.float32),
                  torch.tensor(toy.coupling.mechanism_assignment(), dtype=torch.float32), toy.coupling.n_mechanisms, cfg)  # fmt: skip
    rng = np.random.default_rng(0)
    start = [script._continuation_return(env, d, agent, rng) for d in range(4)]
    assert start[0] == start[1] == start[2] != start[3]  # drugs 0-2 share a mechanism
    last = copy_module.deepcopy(env)
    while last.unwrapped.state()["quarter"] < toy.config.horizon_quarters - 1:
        last.step(0)
    assert last.unwrapped.state()["quarter"] == toy.config.horizon_quarters - 1
    fail_prob = failure_probability(last.unwrapped.state()["resistance"], toy.coupling.matrix)[3]
    # one quarter left: its reward and nothing after it, no continuation and no entropy bonus
    assert script._continuation_return(last, 3, agent, rng) == pytest.approx(-cfg.reward_scale * fail_prob)


def _predictor():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import predict_entropy_gap
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    return predict_entropy_gap


def test_predicted_entropy_gap_vanishes_where_the_two_objectives_coincide_and_favours_large_classes_otherwise():
    predictor = _predictor()
    fast = dict(gamma=0.97, reward_scale=10.0, steps=40, n_random=1, seed=0)
    both = lambda a: {"class entropy": a, "drug entropy": a}  # noqa: E731
    equal = predictor.predict((2, 2), both(0.5), **fast)
    assert equal["class entropy minus drug entropy"] == 0.0  # the objectives differ by the constant alpha log 2
    cold = predictor.predict((3, 1), both(0.0), **fast)
    assert cold["class entropy minus drug entropy"] == 0.0
    warm = predictor.predict((3, 1), both(2.0), **fast)
    assert warm["drug entropy"]["mean_mass_by_class"][0] > warm["class entropy"]["mean_mass_by_class"][0]


def _synthetic_phase_report(kappa, crossing_at, constant_sign=None, sigmas=(0.0, 0.1, 0.2, 0.3, 0.4), episodes=6):
    """planned-once minus greedy equals sigma - crossing_at (or a constant of the given sign), plus small episode noise."""
    results = []
    for s in sigmas:
        offset = (s - crossing_at) if constant_sign is None else constant_sign * (0.05 + s)
        noise = [0.001 * ((e % 3) - 1) for e in range(episodes)]
        results.append({"delay": 1, "lam": 0.8, "sigma": s, "policy": "greedy", "expected_failure_rate": [0.4] * episodes})
        results.append({"delay": 1, "lam": 0.8, "sigma": s, "policy": "planned-once", "expected_failure_rate": [0.4 + offset + n for n in noise]})
    return {"instance": {"kappa": kappa, "delta": 0.05, "horizon_quarters": 40}, "results": results}


def _scorer():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import score_phase_predictions
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    return score_phase_predictions


@pytest.mark.parametrize("predicted, expected_error, verdict", [(0.2, 0.0, "pass"), (0.05, math.log(2), "fail")])  # 0.05 is censored to 0.1
def test_scorer_measures_the_crossing_and_rules_against_the_margin(predicted, expected_error, verdict):
    scorer = _scorer()
    lines, episodes = scorer.measured_lines([_synthetic_phase_report(0.1, 0.2)], "planned_once_minus_greedy")
    out = scorer.score(lines, episodes, {(0.1, 0.05, 40, 1, 0.8): (predicted, 1)}, None, [], 1, math.log(math.sqrt(2)), 200, 0)
    assert out["lines"][0]["measured_crossing"] == pytest.approx(0.2, abs=2e-3)
    assert out["E"] == pytest.approx(expected_error, abs=0.02) and out["verdict"] == verdict


def test_scorer_censors_crossings_outside_the_grid_by_the_contrasts_orientation():
    scorer = _scorer()
    sigmas = np.array([0.0, 0.1, 0.2, 0.4])
    # beyond the crossing the contrast is positive: a line positive everywhere crossed below the grid, negative everywhere above it
    assert scorer.line_error(sigmas, (None, 1), (None, 1), beyond=1) == 0.0
    assert scorer.line_error(sigmas, (None, -1), (None, -1), beyond=1) == 0.0
    assert scorer.line_error(sigmas, (None, 1), (None, -1), beyond=1) == pytest.approx(math.log(0.4 / 0.1))
    assert scorer.line_error(sigmas, (0.2, 1), (None, -1), beyond=1) == pytest.approx(math.log(0.4 / 0.2))
    assert scorer.line_error(sigmas, (None, 1), (0.2, 1), beyond=1) == pytest.approx(math.log(0.2 / 0.1))
    # crossings the grid cannot resolve are clamped, measured and predicted alike
    assert scorer.line_error(sigmas, (None, 1), (0.05, 1), beyond=1) == 0.0
    assert scorer.line_error(sigmas, (0.025, 1), (None, 1), beyond=1) == 0.0
    assert scorer.line_error(sigmas, (None, -1), (0.9, 1), beyond=1) == 0.0


def test_crossing_is_where_the_contrast_settles_into_its_sign():
    scorer = _scorer()
    sigmas = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert scorer.crossing(sigmas, np.array([0.1, -0.1, -0.1, 0.1, 0.2]), 1) == (pytest.approx(0.25), 1)  # not the first dip at 0.05
    assert scorer.crossing(sigmas, np.array([0.1, 0.2, 0.1, 0.3, 0.2]), 1) == (None, 1)
    assert scorer.crossing(sigmas, np.array([0.1, -0.2, 0.1, -0.3, -0.2]), 1) == (None, -1)
    assert scorer.crossing(sigmas, np.zeros(5), 1) == (None, 0)
    assert scorer.line_error(sigmas, scorer.crossing(sigmas, np.zeros(5), 1), (None, -1), beyond=1) == 0.0  # never shows the sign: above


def test_scorer_refuses_reports_evaluated_on_different_episodes():
    scorer = _scorer()
    a, b = _synthetic_phase_report(0.1, 0.2), _synthetic_phase_report(0.05, 0.2)
    a["evaluation_episode_seeds"], b["evaluation_episode_seeds"] = [10_000, 10_001], [30_000, 30_001]
    with pytest.raises(ValueError, match="same evaluation episodes"):
        scorer.measured_lines([a, b], "planned_once_minus_greedy")


def test_scorer_flags_a_shift_in_the_wrong_direction_and_compares_with_a_null():
    scorer = _scorer()
    reports = [_synthetic_phase_report(0.1, 0.2), _synthetic_phase_report(0.2, 0.3)]
    lines, episodes = scorer.measured_lines(reports, "planned_once_minus_greedy")
    theory = {(0.1, 0.05, 40, 1, 0.8): (0.2, 1), (0.2, 0.05, 40, 1, 0.8): (0.3, 1)}
    null = {(0.1, 0.05, 40, 1, 0.8): (0.25, 1), (0.2, 0.05, 40, 1, 0.8): (0.25, 1)}
    direction = {"from": {"kappa": 0.1, "delta": 0.05, "horizon": 40}, "to": {"kappa": 0.2, "delta": 0.05, "horizon": 40}, "sign": -1}
    out = scorer.score(lines, episodes, theory, null, [direction], 1, math.log(math.sqrt(2)), 200, 0)
    (shift,) = out["directions"]
    assert shift["log_shift"] == pytest.approx(math.log(1.5), abs=0.02) and shift["wrong_sign"]
    assert out["E_minus_E_null_ci95"][1] < 0  # the theory places both crossings better than the shared null


def test_null_predictions_carry_each_dev_crossing_to_every_held_out_setting():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import null_phase_predictions
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    dev = _synthetic_phase_report(0.1, 0.2)
    settings = [{"kappa": 0.05, "delta": 0.025, "horizon": 40}, {"kappa": 0.1, "delta": 0.05, "horizon": 80}]
    null = null_phase_predictions.null_predictions(dev, "planned_once_minus_greedy", 1, settings)
    assert len(null["lines"]) == 2 and {row["horizon"] for row in null["lines"]} == {40, 80}
    assert all(row["sigma_star"] == pytest.approx(0.2, abs=2e-3) and row["sign_at_max_sigma"] == 1 for row in null["lines"])


def test_gap_attribution_closes_on_the_measured_gap_and_its_parts_are_consistent():
    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, train
    from ocmdp.envs.toy import ToyOCMDPEnv as Env
    from ocmdp.envs.wrappers import MechanismObservation

    script = _training_script()
    config = ToyConfig(cross_resistance=script.cross_resistance((2, 1)), horizon_quarters=5, decisions_per_quarter=3, **script.ENV_KWARGS)
    env = MechanismObservation(Env(config))
    agent = train(env, SACConfig(total_quarters=120, learning_starts=40, batch_size=32, hidden_dim=16, dual_update_every=40), seed=0).agent
    observations, snapshots, chosen = [], [], []

    def choose(obs, rng):
        observations.append(np.array(obs, dtype=np.float32))
        snapshots.append(copy.deepcopy(env))
        chosen.append(agent.choose(obs, rng))
        return chosen[-1]

    logs = script._rollouts(env, QuarterlyChoicePolicy(choose, 3, "oc-sac"), 2, 10_000)
    out = script._gap_attribution(agent, snapshots, np.stack(observations), np.array(chosen), logs, env.unwrapped.coupling.identical_row_classes(), planner_steps=20)
    assert out["closure_error"] < 1e-9  # the per-decision losses telescope to the gap
    parts = sum(out[k] for k in ("lost_to_policy_class", "lost_to_continuation_ranks_differently", "lost_to_critic_misranks", "lost_to_actor_departs_from_critic"))
    assert parts == pytest.approx(out["gap_to_planner"], abs=1e-9)
    assert min(out[k] for k in ("lost_to_continuation_ranks_differently", "lost_to_critic_misranks", "lost_to_actor_departs_from_critic")) >= 0.0



def test_continuation_value_matches_a_step_by_step_recomputation():
    from ocmdp.agents.oc_sac import SACConfig, train

    script = _training_script()
    env = script.make_env(0.0)
    cfg = SACConfig(total_quarters=240, learning_starts=48, batch_size=32, hidden_dim=16, dual_update_every=40)
    agent = train(env, cfg, seed=0).agent
    obs, _ = env.reset(seed=10_000)
    first = int(np.argmin(agent.probs(obs)))  # a drug the agent would not choose, so the continuation has to switch
    copied = copy.deepcopy(env)
    rewards, bonuses, drug, done = [], [], first, False
    while not done:
        quarter_rewards = []
        quarter = copied.unwrapped.state()["quarter"]
        while not done and copied.unwrapped.state()["quarter"] == quarter:
            obs, reward, terminated, truncated, _ = copied.step(drug)
            quarter_rewards.append(reward)
            done = terminated or truncated
        rewards.append(cfg.reward_scale * np.mean(quarter_rewards))
        if not done:
            with torch.no_grad():
                probs = agent.policy(torch.as_tensor(obs, dtype=torch.float32)[None])
                bonuses.append(float(agent.temperature() * agent.entropy(probs)[0]))
            drug = int(probs[0].argmax())
    assert any(b > 0 for b in bonuses) and drug != first
    expected = rewards[0] + sum(cfg.gamma**k * (bonuses[k - 1] + rewards[k]) for k in range(1, len(rewards)))
    assert script._continuation_return(env, first, agent, np.random.default_rng(0)) == pytest.approx(expected, rel=1e-9)


def test_git_state_hashes_code_changes_only_including_names_with_spaces(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "ocmdp").mkdir()
    (tmp_path / "ocmdp" / "x.py").write_text("a = 1\n")
    git("init", "-q")
    git("add", "ocmdp/x.py")
    git("commit", "-q", "-m", "fixture")
    script = _training_script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    clean = script.git_state()
    assert clean["dirty"] is False and clean["code_diff_sha256"] is None
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "note.md").write_text("not code\n")
    assert script.git_state()["dirty"] is False
    (tmp_path / "experiments").mkdir()
    spaced = tmp_path / "experiments" / "a b.py"
    spaced.write_text("b = 1\n")
    first = script.git_state()
    assert first["dirty"] and first["untracked_code"] == ["experiments/a b.py"]
    spaced.write_text("b = 2\n")
    second = script.git_state()
    assert second["code_diff_sha256"] != first["code_diff_sha256"]
    (tmp_path / "ocmdp" / "x.py").write_text("a = 2\n")
    assert script.git_state()["code_diff_sha256"] not in (first["code_diff_sha256"], second["code_diff_sha256"])
