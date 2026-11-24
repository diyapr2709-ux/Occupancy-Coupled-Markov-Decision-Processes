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


RANGE = (-12.0, 2.0)


def _temperature_grad(log_alpha: float, entropy: float, target: float) -> float:
    raw = torch.tensor(log_alpha, requires_grad=True)
    temperature_loss(raw, torch.tensor(entropy), target, RANGE).backward()
    return raw.grad.item()


def test_temperature_rises_below_target_and_falls_above_it():
    assert _temperature_grad(0.0, entropy=0.2, target=0.8) < 0  # gradient descent increases log alpha
    assert _temperature_grad(0.0, entropy=0.9, target=0.8) > 0


@pytest.mark.parametrize(
    "log_alpha, entropy, inward",
    [(-13.0, 0.2, True), (-13.0, 0.9, False), (3.0, 0.9, True), (3.0, 0.2, False)],
)
def test_temperature_outside_its_range_is_only_pushed_back_in(log_alpha, entropy, inward):
    grad = _temperature_grad(log_alpha, entropy, target=0.8)
    if inward:
        assert grad != 0 and (grad < 0) == (log_alpha < RANGE[0])
    else:
        assert grad == 0


def test_temperature_loss_takes_scalars_only():
    with pytest.raises(ValueError):
        temperature_loss(torch.zeros(3), torch.tensor(0.5), 0.8, RANGE)


def test_entropy_target_lies_strictly_inside_the_reachable_range():
    assert entropy_target(0.5, 1.5, 0.89) == pytest.approx(1.39)
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError):
            entropy_target(0.5, 1.5, fraction)
    with pytest.raises(ValueError):
        entropy_target(1.5, 1.5, 0.5)


def test_dual_variable_moves_on_slow_clock_and_stays_non_negative():
    xi = DualVariable(budget=0.1, lr=1.0, update_every=3)
    assert xi.observe(0.5) == 0.0
    assert xi.observe(0.5) == 0.0
    assert xi.observe(0.5) == pytest.approx(0.4)
    for _ in range(30):
        xi.observe(0.0)
    assert xi.value == 0.0
