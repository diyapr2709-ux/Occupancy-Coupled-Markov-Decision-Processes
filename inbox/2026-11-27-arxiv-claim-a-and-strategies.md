# Literature pass: Claim A noise crossing, the two flagged sources, strategies

Summary:
1. PRE-EMPTS: no confirmed hit. Two possible threats could not be opened (they have no tag; a human with a browser should read both before Claim A's test): the APS 2026 SHEPHERD abstract (MDP on Wright-Fisher dynamics, "strong dependence on the drug-update interval"), and bioRxiv 10.64898/2026.09.17.752270 (closed-loop steering in E. coli).
2. B: "Equivalent Optima, Different Policies" (Wright, AI Engineering, 21 Oct 2026). Abstract and reference list read; the full text PDF could not be read as text. On the abstract: one-shot, no learned agents, but it has the class-level (decoded-fibre) corrected objective. RELATED. The APS "Information and Optimization" abstract was not reachable (HTTP 403).
3. Strategy 1: split "committed open loop minus greedy" through open-loop feedback control (OLFC), i.e. CE replanning. The value of feedback (OL to OLFC) is then kept apart from greedy's myopia loss (OLFC to greedy).
4. Strategy 2: add the decision or monitoring interval as an axis beside sigma, because the drug-control sources tie the value of feedback to update frequency.
5. Strategy 3: give greedy a delay-compensated (predictor) variant that acts on projected post-delay occupancy, so that "greedy loses at large delay" is not just a weak-baseline effect.

Access date for every entry: 2026-11-27. Previously filed items checked against this pass: none is superseded. Three files the brief named were not in the tree on 2026-11-27, so they could not be read: `inbox/2026-11-26-claimA-B1-B2-prior-art-2.md`, `inbox/2026-11-26-claimA-feedback-boundary-search.md` and `arena/2026-11-26-after-step1-stop-outcome.md`. The four questions below are answered as the brief stated them.

## Possible PRE-EMPTS: not reachable, no tag

- **SHEPHERD: Markov Decision Processes on Wright-Fisher Dynamics to Mitigate Therapeutic Resistance in Evolving Populations** (APS Global Physics Summit 2026). Cited in search extracts as Chen, Pachter, Scott, Hinczewski, "Stochastic Evolutionary Control of Therapeutic Resistance in Heterogeneous Populations", in preparation.
  - URL: https://meetings-archive.aps.org/smt/2026/mar-f58/4/ (HTTP 403; not reachable)
  - Claim, from search-engine extracts only: an MDP protocol keeps mean fitness "substantially lower" than any constant or periodic two-drug regimen on 3-, 4- and 8-genotype landscapes. Sensitivity analyses show "a strong dependence on the drug-update interval".
  - No arXiv preprint was found: an author search on Hinczewski for 2024-2026 returned nothing on this topic.
  - Why it matters: this is feedback against open-loop or periodic schedules in drug control, with the feedback value varying along an update-interval axis. If the full text sweeps the update interval and noise and reports a crossing, it comes close to Claim A. Must be read before the Claim A test.
- **Closed-loop steering locates an evolutionary barrier to multidrug resistance in Escherichia coli** (bioRxiv, Sept 2026)
  - URL: https://www.biorxiv.org/content/10.64898/2026.09.17.752270v1 (HTTP 429 on four attempts; not reachable)
  - Claim, from a search extract only: a daily feedback rule acts on the difference between a target resistance vector and the current estimate; progression to multidrug resistance was "constrained without being prevented".
  - Why it matters: it is experimental closed-loop drug steering. Whether it compares against fixed sequences, or treats noise or delay, is unknown.

## B. The two flagged sources

- **Wright, "Equivalent Optima, Different Policies: Certified Distribution-Preserving Reformulations for AI Decision Systems", AI Engineering 2(2), DOI 10.53941/aieng.2026.100020, published 21 Oct 2026**
  - URLs opened:
    - https://www.sciltp.com/journals/aieng/articles/2610005408 (abstract, keywords, references)
    - http://craigwright.net/equivalent-optima (author summary)
    - https://zenodo.org/records/23253837 (replication record)
  - Not readable: https://media.sciltp.com/articles/2610005408/2610005408.pdf was fetched, but no text could be extracted (no PDF renderer installed). The full text is therefore not reachable in readable form, and the four answers below come from the abstract, author page and Zenodo description.
  - Exact claims (abstract):
    - "Equality of the weighted residual-feature spectra across decoder fibres characterizes preservation of the decoded law."
    - "a spectrum with several atoms adds a nonnegative conditional-divergence penalty"
    - "an exact correction recovers the physical objective"
    - "preserving optimal values for every cost is stronger than preserving optimal physical policies"
    - The experiment is a designed six-period scheduling study with 20 cost vectors: "four tied auxiliary bits per period give a median physical-policy total variation of 0.245" (95% bootstrap 0.224-0.254). These are reported-by numbers, not ours.
  - The four questions:
    1. Sequential problems with state: no, as far as the readable material shows. It covers finite log-linear policies with deterministic decoding. The author page mentions "structured dynamic programmes", but no MDP or state transition is described. Reference 31 is "A Theory of Regularized Markov Decision Processes", cited but not visibly used.
    2. Gap between entropy-regularised optima in the physical objective: yes in substance, for the one-shot case. The extra term is a conditional-divergence penalty in the relative-entropy objective, and it is measured as physical-policy total variation. It is not stated as a gap in physical cost under stateful dynamics.
    3. Learned agents: no RL agents. Log-linear policies are fitted on designed instances with two optimisers and three initialisations, and four runs missed the gradient tolerance. Zenodo calls these "scheduling and contextual-learning experiments".
    4. Class-level corrected objective: yes, for the one-shot case. Encoded descriptions are grouped by decoded physical decision (the fibre), and "an exact correction recovers the physical objective". It cites Baram et al., Action Redundancy in RL (reference 33).
  - Tag: RELATED. It does not touch Claim A. For the toy claim B1, it is a second source, after MinRed, showing that the fibre decomposition and the corrected objective are known. That supports B1's existing "claimed as nothing new" wording; it does not open a new stop.
- **"Information and Optimization in Markov Decision Process Models of Drug Protocols"** (APS March Meeting 2026)
  - URL: https://meetings-archive.aps.org/smt/2026/mar-p63/11/ (HTTP 403 again; not reachable)
  - Further searches found no preprint: Semantic Scholar (HTTP 429), arXiv for Maxwell plus drug, arXiv for mutual information plus drug plus MDP, and arXiv for author Hinczewski.
  - Search extracts add only one point to the 26 Nov entry: biopsies happen at intervals, so information is imperfect.
  - The four questions: not reachable. The extracts point to a sequential MDP with a genetic state. They say nothing about an entropy-regularised gap in the physical objective, learned agents, or a class-level corrected objective.
  - Tag: unchanged from 26 Nov (RELATED, provisional); not re-tagged, because the page was not opened.

## A. Claim A search results opened

- **Engelhardt, "Dynamic Control of Stochastic Evolution: A Deep RL Approach to Adaptively Targeting Emergent Drug Resistance" (CelluDose), JMLR 21(203) 2020**
  - URL: https://arxiv.org/abs/1903.11373 (API record opened)
  - Claim: a deep RL feedback controller for drug dosing against stochastic heterogeneous proliferation reaches a 100% success rate and is robust to a key parameter. The abstract does not describe any sweep over noise, delay or memory against an open-loop schedule.
  - Tag: RELATED. Reason: feedback drug control under stochastic evolution, with no characterisation of a feedback-versus-open-loop boundary.
- **Magalang, Aguilar, Esguerra, Roldán, Sanchez-Taltavull, "Optimal switching strategies in multi-drug therapies for chronic diseases"**
  - URL: https://arxiv.org/abs/2411.16362 (API record opened)
  - Claim: therapy switches are modelled as "stochastic resets"; absorption times come from Langevin and master equations; switching protocols are optimised under cost constraints.
  - Tag: RELATED. Reason: a stochastic switching schedule, with no open-loop versus feedback comparison and no delay.
- **"Optimization of sequential therapies to maximize extinction of resistant bacteria through collateral sensitivity"**
  - URL: https://arxiv.org/abs/2510.01808 (seen in the arXiv API listing)
  - Claim: in a stochastic birth-death model, "fast sequential therapies are suboptimal"; extinction and resistance form a Pareto front over the switching period.
  - Tag: RELATED. Reason: periodic open-loop schedules only. Its period axis echoes the delay axis.
- **"Replacing Quantum Feedback with Open-Loop Control and Quantum Filtering"**
  - URL: https://arxiv.org/abs/0908.0198 (seen in the arXiv API listing)
  - Claim: "unavoidable delays in the feedback loop adversely affect their performance", so open-loop control with filtering is proposed instead.
  - Tag: RELATED. Reason: a cross-domain precedent for delay flipping the sign of open loop against feedback. It shows the qualitative sign flip is not new in general; Claim A's mechanism (an action-occupancy kernel) is not involved.
- Null arXiv API queries (abstract field, 0 hits): every combination of open-loop/closed-loop/feedback with noise and delay plus threshold/crossover/regime/phase diagram; receding horizon with open-loop, noise and delay or memory; certainty equivalent with open-loop and delay, memory or history; adaptive therapy with delay or monitoring and fixed schedules; open-loop with past actions or hysteresis and noise.

## C. Strategies (one line each on what it would change)

- **Open-loop feedback control (OLFC) and CE replanning as the feedback reference.** Lecture notes surfaced by search state that the OLFC cost is at most the optimal open-loop cost, citing Bertsekas 2007, Proposition 6.2.1. The same notes say a CE controller can be strictly worse than optimal open loop. Source: https://www.professeurs.polymtl.ca/jerome.le-ny/teaching/DP_fall09/notes/lec8_ADP.pdf (fetched; the PDF text could not be read; claim not verified, no tag).
  - What it would change: write the Claim A contrast as OL to OLFC (value of feedback, non-negative if the proposition holds) plus OLFC to greedy (myopia). A greedy-against-OL crossing then reads as myopia loss trading against feedback value, not as "feedback overtakes planning".
  - If OLFC is the "replanning variant" already in the code, report it beside greedy at every cell. Adding it as a comparator is the owner's call.
- **Decision-interval or monitoring axis** (SHEPHERD, not reachable; Fischer et al. 2015, already filed).
  - What it would change: a reviewer will ask whether sigma* moves with how often greedy re-decides. Either fix the interval and state it as a limitation, or add the axis under a new preregistration.
- **Delay-compensated greedy** (strategy proposal; no source opened).
  - What it would change: a predictor-style greedy scores each drug on the occupancy projected past the delay. Without it, greedy's loss at large delay may just reflect a weak baseline. This is the "strongest honest baseline" check on the feedback arm, and it needs owner approval as a comparator.
- **Blind-prediction reporting.** Pre-registration for predictive modelling (Hofman et al., arXiv 2311.18807) was seen in search only and not opened, so it has no tag.
  - What it would change: little. The project already freezes hashed predictions, scores them against a theory-free null and counts omitted lines as infinite error. The remaining gap is to report per-line sign agreement and the skill score against the null with intervals, under one pre-fixed aggregate so there is no multiplicity.
- **Template for the regime map.** Maynard-Zhang et al. (arXiv 2610.33412, already filed) stays the template for the OL/CE/optimal decomposition; nothing in this pass supersedes it.
