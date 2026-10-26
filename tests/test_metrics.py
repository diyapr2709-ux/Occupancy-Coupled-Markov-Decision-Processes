import math

import numpy as np
import pytest

from ocmdp.eval.metrics import (
    entropy,
    expected_failure_rate,
    pushforward_occupancy_entropy,
    time_to_fixation,
)
from ocmdp.eval.rollout import EpisodeLog


def test_time_to_fixation():
    r = np.array([[0.1, 0.1], [0.4, 0.2], [0.6, 0.3], [0.9, 0.3]])
    np.testing.assert_array_equal(time_to_fixation(r, 0.5), [2, np.inf])


def test_entropy_handles_zero_probabilities():
    np.testing.assert_allclose(entropy(np.array([[1.0, 0.0], [0.5, 0.5]])), [0.0, math.log(2)])


def test_pushforward_entropy_of_within_class_mixing_is_zero():
    assignment = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    occupancy = np.tile([0.5, 0.5, 0.0, 0.0], (10, 1))
    assert pushforward_occupancy_entropy(occupancy, assignment) == 0.0


def test_expected_failure_rate_indexes_quarter_and_action():
    log = EpisodeLog(
        actions=np.array([0, 1, 1, 0]),
        decision_quarter=np.array([0, 0, 1, 1]),
        failures=np.zeros(4, dtype=bool),
        violations=np.zeros(4, dtype=bool),
        resistance=np.zeros((3, 2)),
        occupancy=np.zeros((3, 2)),
        fail_prob=np.array([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]),
    )
    assert expected_failure_rate(log) == pytest.approx((0.1 + 0.2 + 0.4 + 0.3) / 4)
