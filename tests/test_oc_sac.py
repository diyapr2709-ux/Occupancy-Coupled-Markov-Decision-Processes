import copy
import math

import numpy as np
import pytest
import torch

from ocmdp.agents import oc_sac
from ocmdp.agents.oc_sac import Agent, QuarterlyChoicePolicy, SACConfig, _check_finite, train
from ocmdp.baselines.heuristics import MyopicGreedy
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.envs.wrappers import MechanismObservation
from ocmdp.eval.rollout import rollout
from ocmdp.objectives.pushforward_entropy import within_class_entropy

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


def test_entropy_modes_share_initial_weights_reset_seeds_and_replay_order(monkeypatch):
    original_sample, original_update = oc_sac._Replay.sample, oc_sac._update

    def trace(entropy):
        resets, draws, initial = [], [], []
        env = tiny_env()
        env_reset = env.reset

        def reset(**kwargs):
            resets.append(kwargs["seed"])
            return env_reset(**kwargs)

        def sample(replay, n, rng):
            draws.append(copy.deepcopy(rng).integers(replay.size, size=n).tolist())
            return original_sample(replay, n, rng)

        def update(agent, *args):
            if not initial:  # nothing has been optimised before the first update
                initial.extend(p.clone() for p in agent.parameters())
            return original_update(agent, *args)

        monkeypatch.setattr(env, "reset", reset)
        monkeypatch.setattr(oc_sac._Replay, "sample", sample)
        monkeypatch.setattr(oc_sac, "_update", update)
        trained = train(env, SACConfig(**{**vars(TINY), "entropy": entropy}), seed=5).agent
        return resets, draws, initial, trained

    push_resets, push_draws, push_initial, push = trace("pushforward")
    action_resets, action_draws, action_initial, action = trace("action")
    assert push_resets == action_resets and len(set(push_resets)) == len(push_resets) == 40
    assert push_draws == action_draws and len(push_draws) == TINY.total_quarters - TINY.learning_starts + 1
    assert len(push_initial) == len(action_initial) > 0
    for p, q in zip(push_initial, action_initial):
        assert torch.equal(p, q)
    # the shared streams are not an artefact of two identical runs: the trained policies differ
    assert any(not torch.equal(p, q) for p, q in zip(push.actor.parameters(), action.actor.parameters()))


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


def test_training_rejects_an_observation_without_the_pending_occupancy():
    cfg = ToyConfig.paired(horizon_quarters=6, decisions_per_quarter=3, expected_reward=True)
    with pytest.raises(ValueError, match="markov_observation"):
        train(MechanismObservation(ToyOCMDPEnv(cfg)), TINY, seed=0)


@pytest.mark.parametrize("class_sizes", [(2, 2), (3, 1)])
def test_drugs_sharing_a_mechanism_are_indistinguishable_to_the_networks(class_sizes):
    # The actor and critics score a drug through its row of C, and MechanismObservation maps prescribing through C,
    # so drugs with the same row get the same probability and value from any weights. The entropy a policy spends
    # inside a class is therefore pinned at its maximum, sum_j q_j log n_j, whatever the entropy bonus is taken on.
    rows = tuple(tuple(float(j == m) for j in range(len(class_sizes))) for m, n in enumerate(class_sizes) for _ in range(n))
    env = MechanismObservation(ToyOCMDPEnv(ToyConfig(cross_resistance=rows, markov_observation=True, expected_reward=True)))
    toy = env.unwrapped
    obs, info = env.reset(seed=0)
    for drug in range(toy.action_space.n):
        obs, *_ = env.step(drug)
    torch.manual_seed(0)
    agent = Agent(obs.shape[0], torch.tensor(toy.coupling.matrix, dtype=torch.float32), torch.tensor(toy.coupling.mechanism_assignment(), dtype=torch.float32), toy.coupling.n_mechanisms, TINY)
    with torch.no_grad():
        for p in agent.parameters():
            p.add_(torch.randn_like(p))
        probs, _ = agent.policy(torch.as_tensor(obs)[None])
        q = agent.q(torch.as_tensor(obs)[None])
    classes = toy.coupling.identical_row_classes()
    for members in classes:
        assert torch.equal(probs[0, list(members)], probs[0, members[0]].expand(len(members)))
        assert torch.equal(q[0, list(members)], q[0, members[0]].expand(len(members)))
    pinned = sum(probs[0, list(c)].sum() * math.log(len(c)) for c in classes)
    assert within_class_entropy(probs[0], classes).item() == pytest.approx(pinned.item(), abs=1e-6)
