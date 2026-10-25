import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import ocmdp.envs  # noqa: F401  registers the Gymnasium ids
from ocmdp.baselines.heuristics import uniform_mixing
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.rollout import rollout


@pytest.mark.parametrize("markov", [False, True])
def test_passes_gymnasium_env_checker(markov):
    cfg = ToyConfig.paired(contraindication_rate=0.3, delay=2, markov_observation=markov)
    check_env(ToyOCMDPEnv(cfg), skip_render_check=True)


def test_markov_observation_carries_counts_and_pending_occupancy():
    cfg = ToyConfig.two_drug(delay=2, decisions_per_quarter=4, markov_observation=True)
    env = ToyOCMDPEnv(cfg)
    env.reset(seed=0)
    for a in (0, 0, 0, 1, 1):  # one full quarter (3:1), then one decision into the next
        obs, *_ = env.step(a)
    elapsed, counts, pending = obs[4], obs[5:7], obs[7:].reshape(2, 2)
    assert elapsed == pytest.approx(1 / cfg.horizon_quarters)
    np.testing.assert_allclose(counts, [0.0, 0.25])
    # one quarter recorded with delay 2: it enters mu at the 2nd next close, so it sits in row 1
    np.testing.assert_allclose(pending, [[0.0, 0.0], [0.75, 0.25]])
    assert obs.shape == env.observation_space.shape
    for a in (1, 1, 1):  # close the second quarter (all drug 1); the first moves to row 0
        obs, *_ = env.step(a)
    np.testing.assert_allclose(obs[7:].reshape(2, 2), [[0.75, 0.25], [0.0, 1.0]])


def test_heterogeneous_rates_and_initial_conditions():
    cfg = ToyConfig.two_drug(kappa=(0.1, 0.3), delta=(0.05, 0.02), r0=(0.0, 0.2), mu0=(0.9, 0.1), delay=0)
    env = ToyOCMDPEnv(cfg)
    _, info = env.reset(seed=0)
    np.testing.assert_allclose(info["resistance"], [0.0, 0.2])
    np.testing.assert_allclose(info["occupancy"], [0.9, 0.1])
    for _ in range(cfg.decisions_per_quarter):
        _, _, _, _, info = env.step(0)
    mu = 0.8 * np.array([0.9, 0.1]) + 0.2 * np.array([1.0, 0.0])
    expected = np.array([0.0, 0.2]) + np.array([0.1, 0.3]) * mu * (1 - np.array([0.0, 0.2])) - np.array([0.05, 0.02]) * np.array([0.0, 0.2])
    np.testing.assert_allclose(info["resistance"], expected)


@pytest.mark.parametrize("bad", [{"kappa": (0.1, 0.1, 0.1)}, {"r0": 1.5}, {"contraindication_rate": 1.0}])
def test_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        ToyOCMDPEnv(ToyConfig.two_drug(**bad))


def test_state_exposes_full_simulator_state():
    env = ToyOCMDPEnv(ToyConfig.two_drug(delay=1, decisions_per_quarter=3))
    env.reset(seed=0)
    for _ in range(4):
        env.step(1)
    s = env.state()
    assert (s["quarter"], s["decision"]) == (1, 1)
    np.testing.assert_allclose(s["counts"], [0.0, 1.0])
    np.testing.assert_allclose(s["pending"][0], [0.0, 1.0])


@pytest.mark.parametrize("env_id,n_drugs", [("ocmdp/TwoDrug-v0", 2), ("ocmdp/PairedDrugs-v0", 4)])
def test_registered_ids(env_id, n_drugs):
    env = gym.make(env_id)
    assert env.action_space.n == n_drugs


def test_episode_is_horizon_quarters_long():
    cfg = ToyConfig.two_drug(horizon_quarters=7, decisions_per_quarter=5)
    log = rollout(ToyOCMDPEnv(cfg), uniform_mixing(2), seed=0)
    assert len(log.actions) == 35
    assert log.resistance.shape == (8, 2)


def test_kernel_is_fixed_within_a_quarter():
    env = ToyOCMDPEnv(ToyConfig.two_drug(decisions_per_quarter=10))
    _, info = env.reset(seed=0)
    for _ in range(9):
        _, _, _, _, step_info = env.step(0)
        np.testing.assert_array_equal(step_info["resistance"], info["resistance"])
    _, _, _, _, step_info = env.step(0)
    assert step_info["quarter"] == 1
    assert not np.array_equal(step_info["resistance"], info["resistance"])


def test_feasibility_mask_always_nonempty_and_violations_reported():
    env = ToyOCMDPEnv(ToyConfig.paired(contraindication_rate=0.9))
    _, info = env.reset(seed=0)
    violations = 0
    for _ in range(500):
        assert info["action_mask"].any()
        _, _, _, _, info = env.step(0)  # ignores the mask on purpose
        violations += info["violation"]
    assert violations > 0
