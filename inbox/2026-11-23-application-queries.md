# Application query group (literature/queries.txt, last group), filed 2026-11-23

Access date: 2026-11-23. This pass stopped at the turn limit after the first query; coverage is incomplete (see below).

## Run

- abs:"antibiotic" AND (abs:"reinforcement learning" OR abs:"drug cycling") via the arXiv API: 5 results, all seen in the feed only. None uses RL to choose drug sequences or cycles against resistance.
  - https://arxiv.org/abs/2603.11369 (abx_amr_simulator): opened and filed in inbox/2026-11-23-reviewer-leads.md.
  - https://arxiv.org/abs/2509.18153 (deep RL platform for antibiotic peptide discovery). RELATED: RL for molecule design, not treatment sequencing.
  - https://arxiv.org/abs/2505.02722 (LLM clinical reasoning from a sepsis registry). RELATED: no sequential drug policy.
  - https://arxiv.org/abs/2502.14919 (RL for selecting gene tests for resistance prediction). RELATED: diagnostics, not treatment.
  - https://arxiv.org/abs/2004.07782 (generative design of drug combinations). RELATED: no cycling or resistance dynamics.

## Not run (turn limit)

- Rewordings of group 1 (antibiotic cycling/rotation/mixing; antibiotic + optimal control or Markov decision; drug sequencing + resistance).
- Group 2 in all wordings: abs:"antimicrobial resistance" AND abs:"sequential treatment", and its rewordings.
- Group 3 in all wordings: abs:"collateral sensitivity" AND abs:"reinforcement learning", and its rewordings.
- Web (non-arXiv) searches for journal-only RL drug-cycling work.

## Not opened (cited by another source only)

- Weaver et al. (2024), "Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance", PNAS. Cited in the abx_amr_simulator paper as a pathogen-level model with treatment cycling. Not opened; no URL, numeric results or code status verified. Candidate rung-1 reproduction target pending a check.
