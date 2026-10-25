"""The coupling operator Phi: occupancy over drugs -> selection pressure per resistance mechanism.

Cross-resistance is why Phi is not the identity. Row a of the cross-resistance matrix C
says how strongly drug a selects for each mechanism, so drugs with the same row are a
single selection pressure however the policy splits its mass between them. This is the
equivalence the pushforward entropy H(psi#pi) is defined over.
"""

import numpy as np


class CrossResistanceCoupling:
    def __init__(self, cross_resistance: np.ndarray):
        C = np.array(cross_resistance, dtype=np.float64)
        if C.ndim != 2:
            raise ValueError(f"cross_resistance must be a (drugs, mechanisms) matrix, got shape {C.shape}")
        if not np.isfinite(C).all() or np.any(C < 0) or np.any(C > 1):
            raise ValueError("cross-resistance entries must lie in [0, 1]")
        if np.any(C.max(axis=1) == 0):
            raise ValueError("every drug must select for at least one mechanism")
        C.setflags(write=False)
        self.matrix = C

    @classmethod
    def independent(cls, n: int) -> "CrossResistanceCoupling":
        """One mechanism per drug and no cross-resistance."""
        return cls(np.eye(n))

    @property
    def n_drugs(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_mechanisms(self) -> int:
        return self.matrix.shape[1]

    def selection_pressure(self, mu: np.ndarray) -> np.ndarray:
        """pi_j = sum_a C[a, j] mu_a, in [0, 1] whenever mu is a distribution. Batches over leading axes."""
        return np.asarray(mu, dtype=np.float64) @ self.matrix

    __call__ = selection_pressure

    def mechanism_assignment(self) -> np.ndarray:
        """Row-normalised C: the ground-truth drug -> mechanism map a learned embedding psi should recover."""
        return self.matrix / self.matrix.sum(axis=1, keepdims=True)
