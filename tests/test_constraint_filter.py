import numpy as np
import pytest
import torch

from ocmdp.components.constraint_filter import NoFeasibleAction, mask_logits, project_probs


def test_falls_back_to_uniform_over_feasible_set():
    np.testing.assert_allclose(project_probs([1.0, 0.0, 0.0], [False, True, True]), [0.0, 0.5, 0.5])


def test_batched_projection_rows_sum_to_one():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(5), size=32)
    mask = rng.random((32, 5)) < 0.5
    mask[:, 0] = True
    out = project_probs(probs, mask)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0)
    assert np.all(out[~mask] == 0)


def test_empty_feasible_set_raises():
    with pytest.raises(NoFeasibleAction):
        project_probs([0.5, 0.5], [False, False])
    with pytest.raises(NoFeasibleAction):
        mask_logits(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_masked_logits_put_exactly_zero_mass_on_infeasible_actions():
    torch.manual_seed(0)
    logits = torch.randn(64, 20)
    mask = torch.rand(64, 20) < 0.4
    mask[:, 3] = True
    probs = torch.softmax(mask_logits(logits, mask), dim=-1)
    assert torch.all(probs[~mask] == 0)
    torch.testing.assert_close(probs.sum(-1), torch.ones(64))
