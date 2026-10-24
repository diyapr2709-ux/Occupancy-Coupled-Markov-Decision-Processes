"""Transfer instantiation: deploying safety filters against an adapting attack population.

Owner: Diya. Not implemented yet.

Same OC-MDP structure as stewardship. Actions are filter configurations. The attack
population adapts to the time-averaged mix of deployed filters, not to any single one.
Filters that share a detection mechanism are cross-resistant: an attack that evades one
evades its relatives. Implement as a second ToyOCMDPEnv-compatible environment so the
agent and every comparator run on it unmodified.
"""
