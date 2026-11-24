# Informed switching of antibiotics in hospital wards

- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3048378/ (DOI: https://doi.org/10.1371/journal.pcbi.1001094)
- Access date: 2026-11-24
- Source read: PMC full-text page, through the fetch tool's extraction.
- Bibliographic: Kouyos, Abel zur Wiesch, Bonhoeffer. "Informed Switching Strongly Decreases the Prevalence of Antibiotic Resistance in Hospital Wards". PLoS Computational Biology 7(3): e1001094, 2011.

## Exact claims (verbatim abstract fragments)

- "We find that in most cases such an informed switching strategy outperforms both periodic cycling and mixing,"
- "the success of this strategy is essentially a stochastic phenomenon" "taking advantage of the small population sizes in hospital wards."
- Performance "can be improved substantially if information on resistance tests is integrated over a period of one to two weeks."

## Setting (as described on the page)

- Model: a stochastic compartmental model of a single 20-bed ward, simulated with Gillespie's direct algorithm. A deterministic version showed little gain over mixing.
- Drugs and strains: two broad-spectrum drugs, A and B, with four genotypes (wild type, A-resistant, B-resistant, doubly resistant).
- Information: microbiological tests on symptomatically infected patients, who are a small fraction of carriers. Results arrive after about 2 days on average.
- Decision rules:
  - ISSK: a resistance is treated as extinct if it was not detected in the past K days.
  - ISSLast: use the drug whose resistance was last detected longest ago.
  - Snapshot variants.
- Comparators: mixing (random use of both drugs), which is the reference; periodic cycling with a 90-day period; and optimal mixing in an asymmetric scenario. No optimal-control or planned-schedule comparator.
- Reported best: ISSLast, and ISSK with K = 7 or 14 days. These are "reported by" values, not measured here.

## Tag: BASELINE

Reason: informed switching is a state-feedback drug-choice rule that has already been shown to beat fixed cycling and mixing. It is the published feedback-vs-schedule comparison in this domain, so it is a required comparator. It is also prior evidence that feedback beats fixed schedules because of stochasticity, which overlaps the qualitative direction of Claim A. It does not give a phase diagram over delay, occupancy memory and noise, and it has no planner, CE or open-loop-optimal comparator.

- Tuned configuration: the paper reports its best rules (ISSLast, and ISSK with K = 7 or 14 days) and a 90-day cycling period.
- Reference implementation: none seen on the page. Not checked further.
