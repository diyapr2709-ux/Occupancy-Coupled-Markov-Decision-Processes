import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import ocmdp.envs  # noqa: F401  registers the Gymnasium ids
from ocmdp.baselines.heuristics import StationaryMixing, uniform_mixing
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.rollout import rollout
from ocmdp.theory.two_drug import stationary_failure_rate


def test_passes_gymnasium_env_checker():
    check_env(ToyOCMDPEnv(ToyConfig.paired(contraindication_rate=0.3)), skip_render_check=True)


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


@pytest.mark.parametrize("p", [1.0, 0.5, 0.3])
def test_reproduces_analytical_stationary_failure_rate(p):
    """The analytical two-drug result, reproduced numerically by the environment."""
    cfg = ToyConfig.two_drug(rho=0.25, horizon_quarters=600, decisions_per_quarter=50)
    env = ToyOCMDPEnv(cfg)
    policy = np.array([p, 1.0 - p])
    log = rollout(env, StationaryMixing(policy), seed=0)
    simulated = np.mean(log.fail_prob[300:] @ policy)
    predicted = stationary_failure_rate(policy, env.coupling, cfg.kappa, cfg.delta)
    assert simulated == pytest.approx(predicted, abs=5e-3)


def test_feasibility_mask_always_nonempty_and_violations_reported():
    env = ToyOCMDPEnv(ToyConfig.paired(contraindication_rate=0.9))
    _, info = env.reset(seed=0)
    violations = 0
    for _ in range(500):
        assert info["action_mask"].any()
        _, _, _, _, info = env.step(0)  # ignores the mask on purpose
        violations += info["violation"]
    assert violations > 0
