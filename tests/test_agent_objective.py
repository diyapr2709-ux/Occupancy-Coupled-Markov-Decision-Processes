import pytest
import torch

from ocmdp.agents.lagrangian import DualVariable
from ocmdp.agents.losses import actor_loss, entropy_target, temperature_loss
from ocmdp.objectives.pushforward_entropy import pushforward, pushforward_entropy

PAIRED = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def _optimise(logits, assignment, steps=600, **loss_kwargs):
    logits = logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        probs = torch.softmax(logits, dim=-1)
        actor_loss(probs, entropy=pushforward_entropy(probs, assignment), **loss_kwargs).backward()
        opt.step()
    return torch.softmax(logits, dim=-1).detach()


def test_actor_spreads_mass_across_mechanisms():
    """With flat Q, the pushforward objective balances mechanisms, not drugs."""
    zeros = torch.zeros(1, 4)
    probs = _optimise(
        torch.tensor([[3.0, 2.0, 0.0, 0.0]]),
        PAIRED, q_values=zeros, reserve_q=zeros, alpha=torch.ones(1), xi=0.0,
    )
    torch.testing.assert_close(pushforward(probs, PAIRED), torch.tensor([[0.5, 0.5]]), atol=0.02, rtol=0)


def test_reserve_price_moves_mass_off_the_depleting_drug():
    probs = _optimise(
        torch.zeros(1, 2),
        torch.eye(2),
        q_values=torch.zeros(1, 2), reserve_q=torch.tensor([[1.0, 0.0]]),
        alpha=torch.full((1,), 0.05), xi=1.0,
    )
    assert probs[0, 0] < 0.1


@pytest.mark.parametrize("entropy_shape,alpha_shape", [((4, 1), (4,)), ((4,), (4, 1))])
def test_actor_loss_rejects_shapes_that_would_broadcast(entropy_shape, alpha_shape):
    probs = torch.full((4, 2), 0.5)
    with pytest.raises(ValueError):
        actor_loss(probs, torch.zeros(4, 2), torch.zeros(4, 2), torch.ones(alpha_shape), 0.0, torch.zeros(entropy_shape))


def test_temperature_rises_when_entropy_is_below_target():
    log_alpha = torch.zeros(3, requires_grad=True)
    temperature_loss(log_alpha, entropy=torch.full((3,), 0.2), target=torch.full((3,), 0.8)).backward()
    assert torch.all(log_alpha.grad < 0)  # gradient descent increases log alpha


def test_entropy_target_rises_as_a_mechanism_nears_fixation():
    r = torch.tensor([[0.05, 0.05], [0.05, 0.45], [0.05, 0.95]])
    target = entropy_target(r, h_low=0.2, h_high=1.0)
    assert torch.all(target[1:] > target[:-1])
    assert 0.2 <= target.min() and target.max() <= 1.0


def test_dual_variable_moves_on_slow_clock_and_stays_non_negative():
    xi = DualVariable(budget=0.1, lr=1.0, update_every=3)
    assert xi.observe(0.5) == 0.0
    assert xi.observe(0.5) == 0.0
    assert xi.observe(0.5) == pytest.approx(0.4)
    for _ in range(30):
        xi.observe(0.0)
    assert xi.value == 0.0
