import math

import numpy as np
import pytest
import torch

from ocmdp.agents.oc_sac import Agent, QuarterlyChoicePolicy, SACConfig, _check_finite, train
from ocmdp.baselines.heuristics import MyopicGreedy
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.envs.wrappers import MechanismObservation
from ocmdp.eval.rollout import rollout

TINY = SACConfig(total_quarters=240, learning_starts=48, batch_size=32, hidden_dim=16, dual_update_every=40)


def tiny_env(**kwargs):
    cfg = ToyConfig.paired(horizon_quarters=6, decisions_per_quarter=3, expected_reward=True, markov_observation=True, **kwargs)
    return MechanismObservation(ToyOCMDPEnv(cfg))


@pytest.mark.parametrize("bad", [{"entropy": "shannon"}, {"learning_starts": 16, "batch_size": 32}, {"total_quarters": 48}])
def test_config_rejects_inconsistent_settings(bad):
    with pytest.raises(ValueError):
        SACConfig(**{**{"total_quarters": 240, "learning_starts": 48, "batch_size": 32}, **bad})


def test_quarterly_choice_holds_one_drug_per_quarter_through_the_filter():
    env = ToyOCMDPEnv(ToyConfig.paired(decisions_per_quarter=10, horizon_quarters=8, contraindication_rate=0.4))
    choices = iter([0, 2, 1, 3, 0, 2, 1, 3])
    policy = QuarterlyChoicePolicy(lambda obs, rng: next(choices), 4, "fixed-sequence")
    log = rollout(env, policy, seed=0)
    assert not log.violations.any()
    for q, drug in enumerate([0, 2, 1, 3, 0, 2, 1, 3]):
        given = log.actions[log.decision_quarter == q]
        assert np.mean(given == drug) > 0.4  # the chosen drug unless contraindicated (rate 0.4)


def test_training_runs_is_deterministic_and_finite():
    a, b = train(tiny_env(), TINY, seed=3), train(tiny_env(), TINY, seed=3)
    assert len(a.episode_failure) == 40
    assert a.episode_failure == b.episode_failure
    assert all(math.isfinite(x) for x in a.episode_failure + a.alpha + a.xi)
    for pa, pb in zip(a.agent.parameters(), b.agent.parameters()):
        torch.testing.assert_close(pa, pb)


def test_entropy_modes_train_different_policies():
    push = train(tiny_env(), TINY, seed=0).agent
    action = train(tiny_env(), SACConfig(**{**vars(TINY), "entropy": "action"}), seed=0).agent
    assert any(not torch.equal(p, q) for p, q in zip(push.actor.parameters(), action.actor.parameters()))
    probs = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    assert push.entropy(probs).item() == pytest.approx(0.0, abs=1e-6)  # one mechanism
    assert action.entropy(probs).item() == pytest.approx(math.log(2))
    assert push.max_entropy == pytest.approx(math.log(2)) and action.max_entropy == pytest.approx(math.log(4))


def test_agent_policy_is_a_distribution_and_alpha_starts_at_alpha_init():
    env = tiny_env()
    obs, _ = env.reset(seed=0)
    toy = env.unwrapped
    agent = Agent(
        obs.shape[0],
        torch.tensor(toy.coupling.matrix, dtype=torch.float32),
        torch.tensor(toy.coupling.mechanism_assignment(), dtype=torch.float32),
        toy.coupling.n_mechanisms,
        TINY,
    )
    probs, log_alpha = agent.policy(torch.as_tensor(obs)[None])
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    assert log_alpha.exp().item() == pytest.approx(TINY.alpha_init)
    assert 0 <= agent.choose(obs, np.random.default_rng(0)) < 4


def test_divergence_is_reported_not_propagated():
    with pytest.raises(FloatingPointError, match="diverged"):
        _check_finite("critic loss", torch.tensor(float("nan")))


def test_myopic_greedy_gives_the_lowest_failure_drug_and_splits_ties():
    np.testing.assert_allclose(MyopicGreedy().probs(None, {"fail_prob": np.array([0.3, 0.1, 0.1, 0.4])}), [0, 0.5, 0.5, 0])
