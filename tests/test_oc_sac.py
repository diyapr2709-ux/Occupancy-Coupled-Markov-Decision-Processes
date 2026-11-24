import copy
import math

import numpy as np
import pytest
import torch

from ocmdp.agents import oc_sac
from ocmdp.agents.entropy_range import reachable_entropy_range, tied_groups
from ocmdp.agents.oc_sac import Agent, QuarterlyChoicePolicy, SACConfig, _check_finite, checkpoint_bytes, load_checkpoint, train
from ocmdp.baselines.heuristics import MyopicGreedy
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.envs.wrappers import MechanismObservation
from ocmdp.eval.rollout import rollout
from ocmdp.objectives.pushforward_entropy import pushforward_entropy, within_class_entropy

TINY = SACConfig(total_quarters=240, learning_starts=48, batch_size=32, hidden_dim=16, dual_update_every=40)


def tiny_env(**kwargs):
    cfg = ToyConfig.paired(horizon_quarters=6, decisions_per_quarter=3, expected_reward=True, markov_observation=True, **kwargs)
    return MechanismObservation(ToyOCMDPEnv(cfg))


@pytest.mark.parametrize(
    "bad",
    [
        {"entropy": "shannon"},
        {"learning_starts": 16, "batch_size": 32},
        {"total_quarters": 48},
        {"temperature": "rising"},
        {"target_fraction": 1.0},
        {"alpha": -0.1},
        {"alpha_init": 0.0},
        {"alpha_lr": -1e-3},
    ],
)
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
    # with two classes of two tied drugs, class entropy reaches [0, log 2] and drug entropy only [log 2, log 4]
    assert (push.entropy_floor, push.entropy_max) == pytest.approx((0.0, math.log(2)), abs=1e-4)
    assert (action.entropy_floor, action.entropy_max) == pytest.approx((math.log(2), math.log(4)), abs=1e-4)
    for agent in (push, action):
        assert agent.entropy_floor < agent.entropy_target < agent.entropy_max


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


def _agent(cfg: SACConfig, class_sizes=(2, 2)) -> tuple[Agent, np.ndarray]:
    rows = tuple(tuple(float(j == m) for j in range(len(class_sizes))) for m, n in enumerate(class_sizes) for _ in range(n))
    env = MechanismObservation(ToyOCMDPEnv(ToyConfig(cross_resistance=rows, markov_observation=True, expected_reward=True)))
    obs, _ = env.reset(seed=0)
    toy = env.unwrapped
    matrix, assignment = toy.coupling.matrix, toy.coupling.mechanism_assignment()
    agent = Agent(obs.shape[0], torch.tensor(matrix, dtype=torch.float32), torch.tensor(assignment, dtype=torch.float32), toy.coupling.n_mechanisms, cfg)
    return agent, obs


def _class_env(class_sizes):
    rows = tuple(tuple(float(j == m) for j in range(len(class_sizes))) for m, n in enumerate(class_sizes) for _ in range(n))
    cfg = ToyConfig(cross_resistance=rows, horizon_quarters=6, decisions_per_quarter=3, expected_reward=True, markov_observation=True)
    return MechanismObservation(ToyOCMDPEnv(cfg))


@pytest.mark.parametrize("class_sizes", [(2, 2), (8, 5, 4, 2, 1)])
@pytest.mark.parametrize("seed", [0, 1])
def test_agents_are_bitwise_identical_without_the_entropy_term(class_sizes, seed):
    # at temperature 0 both agents optimise the same objective from the same weights, so training must not tell them apart
    cfg = {**vars(TINY), "total_quarters": 2000, "temperature": "fixed", "alpha": 0.0}
    trained = [train(_class_env(class_sizes), SACConfig(**{**cfg, "entropy": e}), seed=seed) for e in ("pushforward", "action")]
    assert trained[0].episode_failure == trained[1].episode_failure
    assert set(trained[0].parameter_sha256) == {"initial", "after_first_window"} and trained[0].parameter_sha256 == trained[1].parameter_sha256
    assert oc_sac.parameter_sha256(trained[0].agent) == oc_sac.parameter_sha256(trained[1].agent)


def _matched_agents(class_sizes, alpha):
    """OC-SAC and SAC on one instance at the same fixed temperature, with identical weights, and a batch of states."""
    agents = []
    for entropy in ("pushforward", "action"):
        torch.manual_seed(0)
        agents.append(_agent(SACConfig(**{**vars(TINY), "temperature": "fixed", "alpha": alpha, "entropy": entropy}), class_sizes)[0])
    oc, sac = agents
    sac.load_state_dict(oc.state_dict())
    _, obs = _agent(TINY, class_sizes)
    batch = torch.as_tensor(obs)[None].repeat(16, 1) + 0.3 * torch.randn(16, obs.shape[0], generator=torch.Generator().manual_seed(1))
    return oc, sac, batch


def _actor_gradients(agent, batch, extra=None):
    from ocmdp.agents.losses import actor_loss

    agent.zero_grad()
    probs = agent.policy(batch)
    with torch.no_grad():  # the critics held fixed: only the actor's objective differs between the agents
        q, reserve_q = agent.q(batch), agent.reserve(batch, agent.embeddings)
    loss = actor_loss(probs, q, reserve_q, agent.temperature().expand(len(probs)), 0.0, agent.entropy(probs))
    (loss if extra is None else extra(probs)).backward()
    return [p.grad.clone() for p in agent.actor.parameters()]


@pytest.mark.parametrize("class_sizes, alpha", [((2, 2), 0.1), ((2, 2), 3.0), ((3, 1), 0.1), ((8, 5, 4, 2, 1), 3.0)])
def test_matched_temperature_differs_only_by_the_named_class_size_bonus(class_sizes, alpha):
    # with tied drugs, drug entropy = class entropy + sum_c q_c log n_c, so at one temperature SAC's actor objective is
    # OC-SAC's plus alpha * sum_c q_c log n_c, and its soft critic target exceeds OC-SAC's by gamma * alpha * that sum
    oc, sac, batch = _matched_agents(class_sizes, alpha)
    sizes = torch.tensor(class_sizes, dtype=torch.float32)
    with torch.no_grad():
        probs = oc.policy(batch)
        masses = probs @ oc.assignment
        torch.testing.assert_close(sac.entropy(probs) - oc.entropy(probs), (masses * sizes.log()).sum(-1), atol=1e-5, rtol=0)

    def bonus(p):
        return -(alpha * ((p @ oc.assignment) * sizes.log()).sum(-1)).mean()

    gap = [s - o for s, o in zip(_actor_gradients(sac, batch), _actor_gradients(oc, batch))]
    for g, expected in zip(gap, _actor_gradients(oc, batch, extra=bonus)):
        torch.testing.assert_close(g, expected, atol=1e-5, rtol=1e-4)
    if len(set(class_sizes)) == 1:  # equal classes: the bonus is a constant, so the actor gradients coincide
        assert all(g.abs().max() < 1e-5 for g in gap)


@pytest.mark.parametrize("class_sizes", [(2, 2), (3, 1)])
def test_isolation_identities_detect_a_scaled_drug_entropy(class_sizes, monkeypatch):
    # the identities must be able to fail: a drug entropy off by 1% breaks them
    original = oc_sac.categorical_entropy
    monkeypatch.setattr(oc_sac, "categorical_entropy", lambda p: 1.01 * original(p))
    with pytest.raises(AssertionError):
        test_matched_temperature_differs_only_by_the_named_class_size_bonus(class_sizes, 0.1)


@pytest.mark.parametrize("quarters, expected", [(12_000, 3e-3), (24_000, 2e-3), (48_000, 1e-3)])
def test_temperature_learning_rate_follows_the_reachability_rule(quarters, expected):
    cfg = SACConfig(total_quarters=quarters)
    assert cfg.alpha_lr == expected
    updates = quarters - cfg.learning_starts + 1
    assert cfg.alpha_lr * updates >= 3 * (math.log(cfg.alpha_init) - cfg.log_alpha_range[0])  # three crossings to the far edge
    assert SACConfig(total_quarters=quarters, alpha_lr=3e-4).alpha_lr == 3e-4  # an explicit rate overrides the rule


def test_reachable_entropy_range_follows_the_tied_drugs():
    embeddings = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]], dtype=torch.float64)
    assert tied_groups(embeddings) == [[3], [0, 1, 2]]
    agent, _ = _agent(TINY, class_sizes=(3, 1))
    # drug entropy: least on the singleton class (log 1), greatest uniform over all four drugs
    floor, maximum = reachable_entropy_range(lambda p: -(p * p.clamp_min(1e-300).log()).sum(-1), embeddings)
    assert (floor, maximum) == pytest.approx((0.0, math.log(4)), abs=1e-4)
    assert (agent.entropy_floor, agent.entropy_max) == pytest.approx((0.0, math.log(2)), abs=1e-4)


def test_fixed_temperature_has_no_learned_parameter_and_never_moves():
    cfg = SACConfig(**{**vars(TINY), "temperature": "fixed", "alpha": 0.3})
    agent, _ = _agent(cfg)
    assert not hasattr(agent, "log_alpha") and agent.entropy_target is None
    assert agent.temperature().item() == pytest.approx(0.3)
    result = train(tiny_env(), cfg, seed=0)
    assert result.alpha and all(a == pytest.approx(0.3) for a in result.alpha)


def test_learned_temperature_is_clamped_to_its_range():
    agent, _ = _agent(TINY)
    low, high = TINY.log_alpha_range
    with torch.no_grad():
        agent.log_alpha.fill_(high + 3.0)
        assert agent.temperature().item() == pytest.approx(math.exp(high))
        agent.log_alpha.fill_(low - 3.0)
        assert agent.temperature().item() == pytest.approx(math.exp(low))


def test_reachable_maximum_is_found_away_from_both_starts():
    # four distinguishable drugs, three selecting the first mechanism and one the second: uniform over drugs
    # (both starts) puts 3/4 on the first mechanism, while the maximum, log 2, needs half the mass on the fourth drug
    assignment = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]], dtype=torch.float64)
    floor, maximum = reachable_entropy_range(lambda p: pushforward_entropy(p, assignment), torch.eye(4, dtype=torch.float64))
    assert floor == pytest.approx(0.0, abs=1e-9) and maximum == pytest.approx(math.log(2), abs=1e-4)
    start = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert maximum - start > 0.1


def test_checkpoints_do_not_change_training_and_rebuild_the_agent():
    plain = train(tiny_env(), TINY, seed=2)
    saved = train(tiny_env(), TINY, seed=2, checkpoint_every=100)
    assert saved.episode_failure == plain.episode_failure and saved.alpha == plain.alpha
    assert [d for d, _ in saved.checkpoints] == [100, 200]  # every 100 decisions before the last (240)
    torch.manual_seed(11)
    before = torch.get_rng_state()
    agent, decisions = load_checkpoint(checkpoint_bytes(saved.agent, TINY.total_quarters))
    assert torch.equal(torch.get_rng_state(), before)  # loading draws nothing from the global generator
    assert decisions == TINY.total_quarters and agent.cfg == TINY
    for (name, a), (_, b) in zip(agent.state_dict().items(), saved.agent.state_dict().items()):
        assert torch.equal(a, b), name
    obs, _ = tiny_env().reset(seed=7)
    np.testing.assert_array_equal(agent.probs(obs), saved.agent.probs(obs))
    assert agent.temperature() == saved.agent.temperature()
    early, at = load_checkpoint(saved.checkpoints[0][1])
    assert at == 100 and any(not torch.equal(p, q) for p, q in zip(early.actor.parameters(), agent.actor.parameters()))
    with pytest.raises(ValueError, match="checkpoint_every"):
        train(tiny_env(), TINY, seed=2, checkpoint_every=0)


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
    probs = agent.policy(torch.as_tensor(obs)[None])
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    assert agent.temperature().item() == pytest.approx(TINY.alpha_init)
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
        probs = agent.policy(torch.as_tensor(obs)[None])
        q = agent.q(torch.as_tensor(obs)[None])
    classes = toy.coupling.identical_row_classes()
    for members in classes:
        assert torch.equal(probs[0, list(members)], probs[0, members[0]].expand(len(members)))
        assert torch.equal(q[0, list(members)], q[0, members[0]].expand(len(members)))
    pinned = sum(probs[0, list(c)].sum() * math.log(len(c)) for c in classes)
    assert within_class_entropy(probs[0], classes).item() == pytest.approx(pinned.item(), abs=1e-6)
