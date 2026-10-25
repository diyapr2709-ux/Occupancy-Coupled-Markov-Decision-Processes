"""Dual variable xi for the option-depletion constraint E[Q_reserve] <= budget.

Projected dual ascent on a slower clock than the actor: observations accumulate, and xi
moves once every `update_every` of them, on their mean. A price that moves as fast as
the policy destabilises both.
"""

import math


class DualVariable:
    def __init__(self, budget: float, lr: float, init: float = 0.0, update_every: int = 1, max_value: float | None = None):
        if not (math.isfinite(budget) and math.isfinite(lr) and math.isfinite(init)):
            raise ValueError("budget, lr and init must be finite")
        if lr <= 0 or init < 0 or not isinstance(update_every, int) or update_every < 1:
            raise ValueError("need lr > 0, init >= 0 and an integer update_every >= 1")
        if max_value is not None and not (math.isfinite(max_value) and max_value >= init):
            raise ValueError(f"max_value must be finite and at least init, got {max_value}")
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
        value = float(constraint_value)
        if not math.isfinite(value):
            # max(0, xi + lr * nan) is 0 in Python, which would silently reset the price
            raise ValueError(f"constraint value must be finite, got {constraint_value}")
        self._buffer.append(value)
        if len(self._buffer) >= self.update_every:
            violation = sum(self._buffer) / len(self._buffer) - self.budget
            stepped = self._xi + self.lr * violation
            if not math.isfinite(stepped):
                raise OverflowError(f"dual step overflowed (xi={self._xi}, lr={self.lr}, violation={violation})")
            self._xi = max(0.0, stepped)
            if self.max_value is not None:
                self._xi = min(self._xi, self.max_value)
            self._buffer.clear()
        return self._xi
