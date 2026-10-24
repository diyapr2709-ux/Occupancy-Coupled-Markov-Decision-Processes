import numpy as np
import pytest

from ocmdp.core.coupling import CrossResistanceCoupling
from ocmdp.core.selection import check_rates, failure_probability, resistance_step, stationary_resistance


def test_independent_coupling_is_identity():
    mu = np.array([0.1, 0.6, 0.3])
    np.testing.assert_allclose(CrossResistanceCoupling.independent(3)(mu), mu)


def test_selection_pressure_lies_in_unit_interval():
    rng = np.random.default_rng(0)
    coupling = CrossResistanceCoupling(rng.random((6, 3)))
    pressure = coupling(rng.dirichlet(np.ones(6), size=100))
    assert pressure.shape == (100, 3)
    assert np.all((pressure >= 0) & (pressure <= 1))


def test_mechanism_assignment_is_row_stochastic():
    coupling = CrossResistanceCoupling([[1.0, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(coupling.mechanism_assignment().sum(axis=1), 1.0)


def test_resistance_converges_to_fixed_point_and_stays_bounded():
    kappa, delta = 0.3, 0.1
    pressure = np.array([0.0, 0.4, 1.0])
    r = np.array([0.9, 0.0, 0.0])
    for _ in range(500):
        r = resistance_step(r, pressure, kappa, delta)
        assert np.all((r >= 0) & (r <= 1))
    np.testing.assert_allclose(r, stationary_resistance(pressure, kappa, delta), atol=1e-9)


def test_failure_probability_without_cross_resistance_is_frequency():
    r = np.array([0.2, 0.7])
    np.testing.assert_allclose(failure_probability(r, np.eye(2)), r)


def test_failure_probability_with_cross_resistance():
    r = np.array([0.2, 0.5])
    C = np.array([[1.0, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(failure_probability(r, C), [1 - 0.8 * 0.75, 0.5])


@pytest.mark.parametrize("kappa,delta", [(0.8, 0.3), (-0.1, 0.1), (0.1, 0.0)])
def test_check_rates_rejects_unstable_or_degenerate(kappa, delta):
    with pytest.raises(ValueError):
        check_rates(kappa, delta)
