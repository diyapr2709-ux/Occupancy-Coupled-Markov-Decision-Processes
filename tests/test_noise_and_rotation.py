import numpy as np
import pytest
import torch

from ocmdp.baselines.heuristics import Cycling, ScheduledPolicy, uniform_mixing
from ocmdp.baselines.oracle import FullInformationOracle, best_rotation, tuned_cycling
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv, reflect
from ocmdp.eval.rollout import rollout
from ocmdp.eval.vv import noise_robustness
from ocmdp.theory.mean_field import MeanFieldModel


def test_noise_free_default_draws_nothing_extra_from_the_rng():
    """Turning the noise feature into the code must not change any existing seeded trajectory."""
    a = rollout(ToyOCMDPEnv(ToyConfig.two_drug()), uniform_mixing(2), seed=3)
    b = rollout(ToyOCMDPEnv(ToyConfig.two_drug(resistance_noise=0.0)), uniform_mixing(2), seed=3)
    np.testing.assert_array_equal(a.failures, b.failures)
    np.testing.assert_array_equal(a.resistance, b.resistance)


def test_noise_is_seeded_and_keeps_frequencies_in_range():
    env = ToyOCMDPEnv(ToyConfig.paired(resistance_noise=5.0))
    a, b = rollout(env, uniform_mixing(4), seed=0), rollout(env, uniform_mixing(4), seed=0)
    np.testing.assert_array_equal(a.resistance, b.resistance)
    assert np.all((a.resistance >= 0) & (a.resistance <= 1))
    assert not np.array_equal(a.resistance, rollout(env, uniform_mixing(4), seed=1).resistance)


def test_noise_vanishes_at_the_boundary():
    """Drift scales with sqrt(r (1 - r)): an absent mechanism with no acquisition stays absent."""
    cfg = ToyConfig.two_drug(kappa=(0.0, 0.1), r0=(0.0, 0.3), resistance_noise=1.0)
    log = rollout(ToyOCMDPEnv(cfg), uniform_mixing(2), seed=0)
    assert np.all(log.resistance[:, 0] == 0.0)


def test_tuned_cycling_is_the_best_rotation_on_the_model():
    cfg = ToyConfig.paired()
    model = MeanFieldModel(cfg)
    q = np.arange(cfg.horizon_quarters)

    def value(policy):
        schedule = np.stack([policy.probs(None, {"quarter": int(k)}) for k in q])
        return float(model.simulate(torch.from_numpy(schedule))[0].mean())

    best = value(tuned_cycling(cfg))
    for period in range(1, cfg.horizon_quarters + 1):
        assert best <= value(Cycling(4, period=period)) + 1e-12
        assert best <= value(Cycling(4, period=period, drugs=(0, 2, 1, 3))) + 1e-12
    # a phase-offset rotation (7-quarter first block, then 13) beats every phase-0 one, and the search finds it
    offset = np.eye(4)[np.where(q < 7, 0, np.where(((q - 7) // 13) % 2 == 0, 2, 0))]
    assert best <= value(ScheduledPolicy(offset)) + 1e-12 < min(value(Cycling(4, p, drugs=(0, 2))) for p in range(1, 41))


def test_best_rotation_is_identical_in_any_chunk_size():
    cfg = ToyConfig.paired(horizon_quarters=12)
    model = MeanFieldModel(cfg)
    classes = ToyOCMDPEnv(cfg).coupling.identical_row_classes()
    a = best_rotation(model, model.initial_state(), classes, 12, chunk=7)
    b = best_rotation(model, model.initial_state(), classes, 12, chunk=100_000)
    assert a[2] == b[2] and a[1] == b[1]


@pytest.mark.parametrize("cfg", [ToyConfig.two_drug(), ToyConfig.paired(), ToyConfig.two_drug(r0=(0.2, 0.05))])
def test_oracle_plan_is_never_worse_than_the_best_rotation_in_its_model(cfg):
    env = ToyOCMDPEnv(cfg)
    obs, info = env.reset(seed=0)
    oracle = FullInformationOracle(env, steps=30)
    oracle.probs(obs, info)
    model = MeanFieldModel(cfg)
    classes = env.coupling.identical_row_classes()
    _, rotation_value, _ = best_rotation(model, model.initial_state(), classes, cfg.horizon_quarters)
    assert oracle.predicted_failure_rate <= rotation_value + 1e-9


def test_oracle_splits_mass_evenly_between_interchangeable_drugs():
    env = ToyOCMDPEnv(ToyConfig.paired())
    obs, info = env.reset(seed=0)
    oracle = FullInformationOracle(env, steps=20)
    for _ in range(100):
        p = oracle.probs(obs, info)
        assert p[0] == pytest.approx(p[1]) and p[2] == pytest.approx(p[3])
        obs, _, _, _, info = env.step(oracle.act(obs, info, np.random.default_rng(0)))


def test_reflection_keeps_shocks_in_range_symmetric_and_mean_zero():
    x = np.random.default_rng(0).normal(0.0, 1.0, 400_000)
    bound = np.full_like(x, 0.3)
    y = reflect(x, bound)
    assert np.all(np.abs(y) <= 0.3 + 1e-12)
    np.testing.assert_allclose(reflect(-x, bound), -y, atol=1e-12)
    assert abs(y.mean()) < 3 * y.std() / np.sqrt(len(y))
    np.testing.assert_array_equal(reflect(x[:5], np.zeros(5)), np.zeros(5))
    np.testing.assert_allclose(reflect(np.array([0.1, -0.2]), np.array([0.3, 0.3])), [0.1, -0.2])  # inside: unchanged


def test_jittered_starts_differ_by_seed_and_default_starts_do_not():
    jittered = ToyOCMDPEnv(ToyConfig.paired(r0_spread=0.04, mu0_spread=0.5))
    starts = [jittered.reset(seed=s)[1] for s in range(3)]
    assert len({tuple(i["resistance"]) for i in starts}) == 3 and len({tuple(i["occupancy"]) for i in starts}) == 3
    for i in starts:
        assert np.all(np.abs(i["resistance"] - 0.05) <= 0.04) and i["occupancy"].sum() == pytest.approx(1.0)
    plain = ToyOCMDPEnv(ToyConfig.paired())
    assert len({tuple(plain.reset(seed=s)[1]["resistance"]) for s in range(3)}) == 1


def test_noise_robustness_reports_paired_gains():
    cfg = ToyConfig.two_drug(horizon_quarters=8, decisions_per_quarter=4)
    study = noise_robustness(cfg, seeds=(0, 1), noise_levels=(0.0, 0.3))
    for row in study.values():
        assert {"uniform", "tuned_rotation", "oracle[open-loop]", "oracle[closed-loop]", "replanning_gain", "gain_over_rotation"} <= set(row)
        assert np.isfinite(row["replanning_gain"]["mean"]) and np.isfinite(row["gain_over_rotation"]["mean"])
