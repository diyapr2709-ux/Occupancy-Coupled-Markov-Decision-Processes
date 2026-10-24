# Data

Fitness landscape data for the evolutionary layer (`ocmdp/envs/evolution.py`) goes here: per-drug growth rates over β-lactamase genotypes. The proposal's sequential-therapy figure comes from analysis of these landscapes. Record the source, licence and any preprocessing next to each file.

Expected layout:

- `fitness_landscape.csv`, with rows = drugs, columns = genotypes (bitstring of mutations), values = growth rate
- `SOURCE.md`, with the citation and how the file was produced
