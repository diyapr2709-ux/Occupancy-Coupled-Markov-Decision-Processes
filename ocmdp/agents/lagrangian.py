"""Dual variable xi for the option-depletion constraint E[Q_reserve] <= budget.

Projected dual ascent on a slower clock than the actor: observations accumulate, and xi
moves once every `update_every` of them, on their mean. A price that moves as fast as
the policy destabilises both.
"""


class DualVariable:
    def __init__(self, budget: float, lr: float, init: float = 0.0, update_every: int = 1, max_value: float | None = None):
        if lr <= 0 or update_every < 1 or init < 0:
            raise ValueError("need lr > 0, update_every >= 1 and init >= 0")
        self.budget = budget
        self.lr = lr
        self.update_every = update_every
        self.max_value = max_value
        self._xi = init
        self._buffer: list[float] = []

    @property
    def value(self) -> float:
        return self._xi

    def observe(self, constraint_value: float) -> float:
        self._buffer.append(float(constraint_value))
        if len(self._buffer) >= self.update_every:
            violation = sum(self._buffer) / len(self._buffer) - self.budget
            self._xi = max(0.0, self._xi + self.lr * violation)
            if self.max_value is not None:
                self._xi = min(self._xi, self.max_value)
            self._buffer.clear()
        return self._xi
