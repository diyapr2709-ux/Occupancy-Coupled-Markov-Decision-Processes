import numpy as np
import pytest

from ocmdp.core.occupancy import OccupancyMeasure


def test_decays_at_configured_rate():
    lam = 0.8
    occ = OccupancyMeasure(2, lam, init=np.array([1.0, 0.0]))
    for n in range(1, 40):
        assert occ.update(np.array([0.0, 1.0]))[0] == pytest.approx(lam**n)


def test_recursion_matches_explicit_weighted_sum():
    rng = np.random.default_rng(0)
    lam, init = 0.7, np.array([0.2, 0.3, 0.5])
    ds = rng.dirichlet(np.ones(3), size=25)
    occ = OccupancyMeasure(3, lam, init=init)
    for d in ds:
        mu = occ.update(d)
    T = len(ds)
    explicit = (1 - lam) * sum(lam**k * ds[T - 1 - k] for k in range(T)) + lam**T * init
    np.testing.assert_allclose(mu, explicit)
    assert mu.sum() == pytest.approx(1.0)


def test_delay_holds_back_each_period():
    occ = OccupancyMeasure(2, 0.5, delay=2, init=np.array([1.0, 0.0]))
    np.testing.assert_allclose(occ.update([0.0, 1.0]), [1.0, 0.0])
    np.testing.assert_allclose(occ.update([0.0, 1.0]), [1.0, 0.0])
    np.testing.assert_allclose(occ.update([0.0, 1.0]), [0.5, 0.5])


def test_reset_discards_pending_history():
    occ = OccupancyMeasure(2, 0.5, delay=1, init=np.array([1.0, 0.0]))
    occ.update([0.0, 1.0])
    occ.reset()
    np.testing.assert_allclose(occ.update([1.0, 0.0]), [1.0, 0.0])
    np.testing.assert_allclose(occ.update([1.0, 0.0]), [1.0, 0.0])


@pytest.mark.parametrize("bad", [[0.5, 0.6], [-0.1, 1.1], [1.0]])
def test_rejects_non_distributions(bad):
    with pytest.raises(ValueError):
        OccupancyMeasure(2, 0.5).update(bad)


@pytest.mark.parametrize("lam", [-0.1, 1.0])
def test_rejects_lambda_outside_unit_interval(lam):
    with pytest.raises(ValueError):
        OccupancyMeasure(2, lam)
