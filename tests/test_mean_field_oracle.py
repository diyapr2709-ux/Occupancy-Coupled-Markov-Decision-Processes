import numpy as np
import pytest
import torch

from ocmdp.baselines.heuristics import Cycling, uniform_mixing
from ocmdp.baselines.oracle import FullInformationOracle, plan_schedule, starting_logits
from ocmdp.core.occupancy import scale_effective_horizon
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.metrics import expected_failure_rate
from ocmdp.eval.rollout import rollout
from ocmdp.theory.mean_field import MeanFieldModel
from ocmdp.theory.two_drug import solve_two_drug


def test_constant_schedule_converges_to_stationary_result():
    cfg = ToyConfig.two_drug(rho=0.25)
    schedule = torch.full((600, 2), 0.5, dtype=torch.float64)
    fails, _, _ = MeanFieldModel(cfg).simulate(schedule)
    assert fails[-1].item() == pytest.approx(solve_two_drug(rho=0.25).f_opt, abs=1e-9)


def test_batched_simulation_matches_unbatched():
    model = MeanFieldModel(ToyConfig.paired(delay=2))
    schedules = torch.softmax(torch.randn(3, 10, 4, dtype=torch.float64), -1)
    batched, _, _ = model.simulate(schedules)
    for i in range(3):
        torch.testing.assert_close(batched[i], model.simulate(schedules[i])[0])


def test_failure_rate_is_differentiable_in_schedule_and_lambda():
    model = MeanFieldModel(ToyConfig.two_drug())
    model.lam = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    # not zeros: the uniform schedule is a symmetric saddle with exactly zero gradient
    logits = torch.randn(12, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    model.simulate(torch.softmax(logits, -1))[0].mean().backward()
    assert logits.grad.abs().sum() > 0 and model.lam.grad.abs() > 0


def test_uniform_schedule_is_a_saddle_of_the_symmetric_problem():
    logits = torch.zeros(12, 2, dtype=torch.float64, requires_grad=True)
    MeanFieldModel(ToyConfig.two_drug()).simulate(torch.softmax(logits, -1))[0].mean().backward()
    assert logits.grad.abs().max() < 1e-15


def test_state_from_env_requires_quarter_boundary():
    env = ToyOCMDPEnv(ToyConfig.two_drug())
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(ValueError):
        MeanFieldModel.state_from_env(env)


def test_planner_never_worse_than_its_best_start():
    cfg = ToyConfig.two_drug(horizon_quarters=16)
    model = MeanFieldModel(cfg)
    init = starting_logits(16, 2, n_random=2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        start = model.simulate(torch.softmax(init, -1))[0].mean(-1).min().item()
    _, planned = plan_schedule(model, model.initial_state(), init, steps=100)
    assert planned <= start + 1e-9


def test_oracle_respects_feasibility():
    cfg = ToyConfig.paired(horizon_quarters=12, decisions_per_quarter=10, contraindication_rate=0.3)
    env = ToyOCMDPEnv(cfg)
    log = rollout(env, FullInformationOracle(env, steps=100, refine_steps=20), seed=0)
    assert not log.violations.any()
    assert len(log.actions) == 120


@pytest.mark.parametrize("factor,expected", [(1.0, 0.8), (2.0, 0.9), (0.5, 0.6), (4.0, 0.95)])
def test_scale_effective_horizon(factor, expected):
    lam = scale_effective_horizon(0.8, factor)
    assert lam == pytest.approx(expected)
    assert 1 / (1 - lam) == pytest.approx(factor * 5.0)


def test_scale_effective_horizon_rejects_out_of_range():
    with pytest.raises(ValueError):
        scale_effective_horizon(0.8, 0.1)


def test_oracle_beats_cycling_and_uniform_in_expectation():
    env = ToyOCMDPEnv(ToyConfig.two_drug(horizon_quarters=20))
    oracle = expected_failure_rate(rollout(env, FullInformationOracle(env, steps=150), seed=0))
    for policy in (Cycling(2, period=4), uniform_mixing(2)):
        assert oracle < expected_failure_rate(rollout(env, policy, seed=0))


class _RecordingOracle(FullInformationOracle):
    def _replan(self, quarter):
        self.replanned_at.append(quarter)
        super()._replan(quarter)

    def reset(self):
        super().reset()
        self.replanned_at = []


def test_oracle_replans_every_period_including_a_short_last_one():
    env = ToyOCMDPEnv(ToyConfig.two_drug(horizon_quarters=10, decisions_per_quarter=3))
    oracle = _RecordingOracle(env, replan_every=4, steps=10, refine_steps=5)
    rollout(env, oracle, seed=0)
    assert oracle.replanned_at == [0, 4, 8]


def test_oracle_restarts_cleanly_when_reset_is_skipped():
    env = ToyOCMDPEnv(ToyConfig.two_drug(horizon_quarters=6, decisions_per_quarter=2))
    oracle = FullInformationOracle(env, replan_every=4, steps=10, refine_steps=5)
    for seed in (0, 1):
        obs, info = env.reset(seed=seed)  # deliberately no oracle.reset()
        truncated = False
        while not truncated:
            obs, _, _, truncated, info = env.step(oracle.act(obs, info, np.random.default_rng(seed)))


def test_oracle_rejects_nonpositive_replan_period():
    with pytest.raises(ValueError):
        FullInformationOracle(ToyOCMDPEnv(), replan_every=0)
