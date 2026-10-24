"""Evolutionary layer: genotype dynamics over an empirical fitness landscape.

Owner: Nikhila. Not implemented yet.

Built directly from published β-lactamase fitness landscape data (see data/README.md).
Each drug defines a fitness value per genotype; under drug mix mu the population moves
on the mu-weighted landscape. The cross-resistance matrix used by Phi and by the C2
graph should be derived from the same data (fitness correlations between drugs).
"""

import numpy as np


class FitnessLandscape:
    def __init__(self, fitness: np.ndarray, mutation_rate: float):
        """fitness: (drugs, genotypes) growth rates; genotypes indexed as bitstrings of mutations."""
        raise NotImplementedError

    @classmethod
    def from_data(cls, path: str) -> "FitnessLandscape":
        raise NotImplementedError

    def step(self, genotype_freqs: np.ndarray, drug_mix: np.ndarray) -> np.ndarray:
        """Advance genotype frequencies one quarter under the given drug mix."""
        raise NotImplementedError

    def cross_resistance(self) -> np.ndarray:
        """(drugs, drugs) fitness correlation, used to initialise the C2 graph."""
        raise NotImplementedError
