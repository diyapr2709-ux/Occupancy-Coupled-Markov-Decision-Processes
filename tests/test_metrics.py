import math

import numpy as np

from ocmdp.eval.metrics import effective_dimensionality, entropy, pushforward_occupancy_entropy, time_to_fixation


def test_effective_dimensionality_detects_collapse():
    rng = np.random.default_rng(0)
    assert effective_dimensionality(rng.normal(size=(500, 8))) > 6.5
    assert math.isclose(effective_dimensionality(np.outer(rng.normal(size=50), rng.normal(size=8))), 1.0)
    assert effective_dimensionality(np.ones((5, 3))) == 0.0


def test_time_to_fixation():
    r = np.array([[0.1, 0.1], [0.4, 0.2], [0.6, 0.3], [0.9, 0.3]])
    np.testing.assert_array_equal(time_to_fixation(r, 0.5), [2, np.inf])


def test_entropy_handles_zero_probabilities():
    np.testing.assert_allclose(entropy(np.array([[1.0, 0.0], [0.5, 0.5]])), [0.0, math.log(2)])


def test_pushforward_entropy_of_within_class_mixing_is_zero():
    assignment = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    occupancy = np.tile([0.5, 0.5, 0.0, 0.0], (10, 1))
    assert pushforward_occupancy_entropy(occupancy, assignment) == 0.0
