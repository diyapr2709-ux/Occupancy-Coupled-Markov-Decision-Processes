import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from ocmdp.agents.distill import HeadPolicy, collect, dagger, distill
from ocmdp.baselines.oracle import FullInformationOracle
from ocmdp.components.policy_head import MixingPolicyHead
from ocmdp.components.reserve_critic import ReserveCritic, reserve_cost, reserve_td_target
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.envs.wrappers import MechanismObservation
from ocmdp.eval.rollout import rollout

torch.manual_seed(0)


def test_policy_head_outputs_a_masked_distribution_and_per_state_temperature():
    head = MixingPolicyHead(state_dim=6, embed_dim=3, hidden_dim=32)
    mask = torch.rand(8, 5) < 0.5
    mask[:, 0] = True
    probs, log_alpha = head(torch.randn(8, 6), torch.randn(5, 3), mask)
    torch.testing.assert_close(probs.sum(-1), torch.ones(8))
    assert torch.all(probs[~mask] == 0)
    assert log_alpha.shape == (8,)


def test_policy_head_is_permutation_equivariant_and_ties_identical_embeddings():
    head = MixingPolicyHead(state_dim=4, embed_dim=2, hidden_dim=32)
    state, emb = torch.randn(3, 4), torch.randn(5, 2)
    emb[1] = emb[0]
    probs, _ = head(state, emb)
    torch.testing.assert_close(probs[:, 0], probs[:, 1])
    perm = torch.tensor([4, 2, 0, 1, 3])
    torch.testing.assert_close(head(state, emb[perm])[0], probs[:, perm])


def test_policy_head_gradient_reaches_parameters_and_embeddings():
    head = MixingPolicyHead(state_dim=4, embed_dim=2, hidden_dim=16)
    emb = torch.randn(3, 2, requires_grad=True)
    probs, log_alpha = head(torch.randn(5, 4), emb)
    (probs[:, 0].sum() + log_alpha.sum()).backward()
    assert emb.grad.abs().sum() > 0
    assert all(p.grad is not None for p in head.parameters())


def test_reserve_critic_shape_and_equivariance():
    critic = ReserveCritic(state_dim=4, embed_dim=2, hidden_dim=16)
    state, emb = torch.randn(3, 4), torch.randn(5, 2)
    q = critic(state, emb)
    assert q.shape == (3, 5)
    perm = torch.tensor([1, 0, 4, 3, 2])
    torch.testing.assert_close(critic(state, emb[perm]), q[:, perm])


def test_reserve_cost_is_the_net_drop_in_effective_drugs():
    before = np.array([0.2, 0.5, 0.7, 0.3])
    after = np.array([0.65, 0.4, 0.5, 0.61])  # drugs 0 and 3 lost, drug 2 regained: 3 effective -> 2
    assert reserve_cost(before, after, threshold=0.6) == 1.0
    assert reserve_cost(after, before, threshold=0.6) == 0.0  # a net gain costs nothing


@pytest.mark.parametrize("before,after", [([np.nan, 0.5], [0.9, 0.9]), (0.5, [0.7] * 4), ([0.1, 0.2], [0.1])])
def test_reserve_cost_rejects_malformed_inputs(before, after):
    with pytest.raises(ValueError):
        reserve_cost(before, after, threshold=0.6)


def test_reserve_td_target_stops_at_episode_end():
    cost = torch.tensor([1.0, 0.0])
    next_probs = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    next_q = torch.tensor([[2.0, 4.0], [3.0, 9.0]])
    target = reserve_td_target(cost, next_probs, next_q, gamma=0.9, done=torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(target, torch.tensor([1.0 + 0.9 * 3.0, 0.0]))


@pytest.mark.parametrize("markov", [False, True])
def test_mechanism_observation_pushes_drug_blocks_forward(markov):
    cfg = ToyConfig.paired(rho=0.3, delay=2, markov_observation=markov)
    raw = ToyOCMDPEnv(cfg)
    env = MechanismObservation(ToyOCMDPEnv(cfg))
    check_env(env.unwrapped, skip_render_check=True)
    raw_obs, _ = raw.reset(seed=0)
    obs, _ = env.reset(seed=0)
    for a in [0, 2, 3, 1, 0] * 12:
        raw_obs, *_ = raw.step(a)
        obs, *_ = env.step(a)
        assert env.observation_space.contains(obs)
    A = env.mechanism_map
    np.testing.assert_allclose(obs[2:4], raw_obs[2:6] @ A, atol=1e-6)
    if markov:
        np.testing.assert_allclose(obs[4], raw_obs[6])
        np.testing.assert_allclose(obs[7:].reshape(2, 2), raw_obs[11:].reshape(2, 4) @ A, atol=1e-6)


def test_mechanism_observation_cannot_tell_cross_resistant_duplicates_apart():
    env = MechanismObservation(ToyOCMDPEnv(ToyConfig.paired(rho=0.2, markov_observation=True)))
    env.reset(seed=0)
    first = [env.step(0)[0] for _ in range(60)]
    env.reset(seed=0)
    second = [env.step(1)[0] for _ in range(60)]
    np.testing.assert_allclose(np.array(first), np.array(second), atol=1e-7)  # float32 obs, summation-order rounding


@pytest.mark.parametrize("bad", [np.full((4, 2), -0.5), np.full((4, 2), 1.5), np.ones((3, 2)), np.full((4, 2), np.nan)])
def test_mechanism_observation_rejects_bad_map(bad):
    with pytest.raises(ValueError):
        MechanismObservation(ToyOCMDPEnv(ToyConfig.paired()), mechanism_map=bad)


def test_mechanism_observation_refuses_to_stack_on_layout_changing_wrappers():
    import gymnasium as gym

    inner = MechanismObservation(ToyOCMDPEnv(ToyConfig.paired()))
    with pytest.raises(ValueError):
        MechanismObservation(inner)
    with pytest.raises(ValueError):
        MechanismObservation(gym.wrappers.NormalizeObservation(ToyOCMDPEnv(ToyConfig.paired())))


def test_mechanism_observation_keeps_proportional_drugs_apart():
    """Rows (1,) and (0.2,) are one mechanism at fivefold different pressure: still distinguishable."""
    env = MechanismObservation(ToyOCMDPEnv(ToyConfig(cross_resistance=((1.0,), (0.2,)), delay=0, decisions_per_quarter=1)))
    env.reset(seed=0)
    a = env.step(0)[0]
    env.reset(seed=0)
    b = env.step(1)[0]
    assert not np.allclose(a[1], b[1])


def test_distillation_reduces_cross_entropy_and_yields_a_valid_policy():
    env = ToyOCMDPEnv(ToyConfig.two_drug(horizon_quarters=8, decisions_per_quarter=5))
    emb = torch.tensor(env.config.cross_resistance, dtype=torch.float32)
    demos = collect(env, FullInformationOracle(env, steps=30, refine_steps=10), seeds=(0,))
    assert demos.obs.shape == (40, 4) and demos.probs.shape == (40, 2)
    head = MixingPolicyHead(4, 2, hidden_dim=32)
    losses = distill(head, emb, demos, steps=200)
    assert np.mean(losses[-20:]) < losses[0]
    p = HeadPolicy(head, emb).probs(demos.obs[0].numpy(), {})
    assert p.shape == (2,) and np.isclose(p.sum(), 1.0)


def test_dagger_aggregates_student_states():
    env = ToyOCMDPEnv(ToyConfig.paired(horizon_quarters=4, decisions_per_quarter=3))
    emb = torch.tensor(env.config.cross_resistance, dtype=torch.float32)
    teacher = FullInformationOracle(env, steps=20, refine_steps=5)
    student = dagger(env, teacher, MixingPolicyHead(env.observation_space.shape[0], 2, 16), emb, seeds=(0,), rounds=2, steps=20)
    log = rollout(env, student, seed=0)
    assert len(log.actions) == 12
