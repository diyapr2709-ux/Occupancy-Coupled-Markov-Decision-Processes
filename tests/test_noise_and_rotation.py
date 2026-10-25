import numpy as np
import torch

from ocmdp.baselines.heuristics import Cycling, uniform_mixing
from ocmdp.baselines.oracle import tuned_cycling
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
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


def test_noise_robustness_reports_paired_gains():
    cfg = ToyConfig.two_drug(horizon_quarters=8, decisions_per_quarter=4)
    study = noise_robustness(cfg, seeds=(0, 1), noise_levels=(0.0, 0.3))
    for row in study.values():
        assert {"uniform", "oracle[open-loop]", "oracle[closed-loop]", "replanning_gain", "gain_over_rotation"} <= set(row)
        assert np.isfinite(row["replanning_gain"]["mean"]) and np.isfinite(row["gain_over_rotation"]["mean"])
