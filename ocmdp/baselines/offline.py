"""CQL / IQL offline comparators. Owner: Soumitra. Not implemented yet.

Trained on logs from the fixed-guideline and cycling policies, to test whether logged
practice covers the counterfactual rotations an occupancy-coupled policy needs.

Stable-Baselines3 (and sb3-contrib) do not ship CQL or IQL. Candidates are d3rlpy
(DiscreteCQL; its IQL is continuous-action only) or a single-file implementation in
CleanRL style. Decide before the comparator sweep is scheduled.
"""
