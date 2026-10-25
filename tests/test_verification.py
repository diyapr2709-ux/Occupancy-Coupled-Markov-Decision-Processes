"""Every V&V verification check must pass (pending is allowed until its component exists)."""

import numpy as np
import pytest

from ocmdp.baselines.heuristics import FixedGuideline
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.eval.metrics import expected_failure_rate
from ocmdp.eval.rollout import rollout
from ocmdp.eval.seeds import eval_seed
from ocmdp.eval.vv import VERIFICATION, aggregate, comparator_table, comparators


@pytest.mark.parametrize("name", list(VERIFICATION))
def test_verification_check(name):
    result = VERIFICATION[name]()
    assert result.status in ("pass", "pending"), f"{name}: {result.detail}"


def test_comparator_table_regret_is_paired_by_seed_and_positive_when_worse():
    env = ToyOCMDPEnv(ToyConfig.paired(horizon_quarters=6, decisions_per_quarter=5, contraindication_rate=0.2))
    seeds = (0, 1)
    table = comparator_table(env, seeds)
    assert list(table)[-1] == "oracle"
    assert table["oracle"]["regret"]["mean"] == 0.0
    # recompute fixed-guideline regret by hand, pairing each seed with the oracle's episode on the same seed
    oracle = comparators(env)[-1]
    regrets = [
        30 * (expected_failure_rate(rollout(env, FixedGuideline(4, 0), eval_seed(s))) - expected_failure_rate(rollout(env, oracle, eval_seed(s))))
        for s in seeds
    ]
    assert table["fixed[0]"]["regret"]["mean"] == pytest.approx(np.mean(regrets))
    assert table["fixed[0]"]["regret"]["mean"] > 0


def test_aggregate_uses_sample_std_and_counts_only_positive_infinity_as_never():
    agg = aggregate([1.0, 2.0, 3.0, 4.0, 5.0])
    assert agg["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert aggregate([np.inf, 2.0, 4.0]) == {"mean": 3.0, "std": pytest.approx(np.sqrt(2)), "never": 1}
    assert aggregate([np.inf, np.inf])["mean"] is None


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_aggregate_raises_on_broken_metric(bad):
    with pytest.raises(ValueError):
        aggregate([1.0, bad])
