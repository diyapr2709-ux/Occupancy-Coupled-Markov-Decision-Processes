"""Ward transmission layer: lightweight agent-based model of patients on a ward.

Owner: Nikhila. Not implemented yet.

Patients are admitted carrying a genotype drawn from the community/hospital reservoir
(supplied by the evolutionary layer), acquire genotypes by contact on the ward, are
treated with the drug the policy selects, and are discharged. Census, culture events and
isolation status feed the ward-state input of Figure 1.
"""

import numpy as np


class WardModel:
    def __init__(self, n_beds: int, contact_rate: float, length_of_stay: float):
        raise NotImplementedError

    def admit(self, reservoir_freqs: np.ndarray, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def treat(self, patient: int, drug: int, rng: np.random.Generator) -> bool:
        """Treat one patient; returns True on treatment failure."""
        raise NotImplementedError

    def ward_state(self) -> np.ndarray:
        """Census, culture and isolation summary for the policy's ward-state input."""
        raise NotImplementedError
