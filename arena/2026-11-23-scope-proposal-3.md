# Proposal 3

Proposal: put the problem-class result first, cut the entropy claim down to one structure-aware test, and get the comparators right

## 1. The recommendation

**Q1. Narrow the entropy claim; do not drop it.** Stop claiming "entropy over resistance mechanisms preserves drug efficacy better than entropy over drugs" in its crisp-class form.
- **Why:** with exactly interchangeable drugs and tied networks, the verified identity (`decisions/2026-11-23.md`) makes SAC equal to OC-SAC plus a reward bonus alpha(s)·log n_class(a). Baram et al. (UAI 2021, MinRed) already argue and show the unequal-class failure of action entropy, with classes given by identical next states.
- **What the crisp case becomes:** a stated proposition (the identity) plus a model prediction from `experiments/predict_entropy_gap.py`. It is checked once on dev as a verification, not claimed as a result.
- **The exact narrowed claim (E):** "On an instance whose cross-resistance matrix comes from a measured source, so that no two drugs are exactly interchangeable, SAC whose entropy is similarity-sensitive (H_Z, with the similarity taken from C) differs in expected failure rate from:
  - action-entropy SAC,
  - MinRed SAC,
  - SAC with no entropy (alpha = 0),

  at equal treatment cost, with each agent's temperature chosen by the rule the temperature arena fixes, tuned at parity on val seeds."
- **How it is decided:** two-sided, with a preregistered equivalence margin, so "equivalent" is a reportable outcome.
- **Wording:** a learning-aid claim, nothing more. The planner's optima are near-deterministic and both agents' evaluation temperatures sit at the clamp floor (`runs/train/train_20261123_190717_20498.json`, `train_20261123_190750_20538.json`). Any efficacy difference must therefore come from training dynamics, and the paper says so.

**Q2. What carries the paper, in order.**
1. **Primary: the control theory and phase diagram of occupancy-coupled MDPs.**
   - The formulation: a kernel driven by a delayed, exponentially weighted occupancy, with the pending occupancy in the observation.
   - The structural results already in `ocmdp/theory`.
   - The boundary in (delay, memory, noise) between committed schedules and state feedback, derived blind and confirmed at held-out (kappa, delta, horizon). This follows the superseding entry in `decisions/2026-11-23.md`, with the scoring code already written (`experiments/score_phase_predictions.py`, and `null_phase_predictions.py` as the theory-free competitor).
   - Learned agents placed on the diagram at a few preregistered cells, scored against the optimal references from Q5.
2. **Then the bridge (rung 3):** the Weaver et al. 2024 landscapes plus occupancy coupling. This is the only place results stand on shared ground with prior work. It waits for rung 1, whose route a separate arena decides.
3. **Then the narrowed entropy test E,** as one preregistration with one confirmatory run.
4. **Last, the completeness batch (Q3).**

On the claims ladder: the phase diagram uses planners and heuristics only, so I read it as "not new method work" and not blocked by rung 1. Rung-1 work runs in parallel with it. Every agent-versus-agent claim (E and the agents on the diagram) waits for rungs 1 and 2. That reading of CLAUDE.md is the owner's to confirm, not mine.

**Q3. Comparators.** No comparator below is weakened.

*For test E:*

| Comparator | Strongest published version | Reference code |
|---|---|---|
| Discrete SAC, action entropy | Christodoulou 2019 (arXiv 1910.07207); stabilisers from SD-SAC (arXiv 2209.10081: averaged target critics, Q-clip, entropy penalty) | Yes: CleanRL `sac_atari.py` at 004f8a0, SD-SAC repo (not inspected) |
| Its target rule | The dynamic rule of Mohammadpour, Frejinger and Bacon (ICLR 2024, arXiv 2406.05953), per-state log\|A(s)\| under masks, unless the temperature arena rules otherwise | Yes: github.com/SobhanMP/decoupled-soft-RL (licence unchecked) |
| MinRed SAC, the closest prior | Baram et al. UAI 2021, in both forms: learned inverse model, and known class map | No: build from the paper; SB3 hyperparameters in its Appendix B |
| Fixed-temperature SAC, and alpha = 0 | SD-SAC configuration (fixed alpha 0.05) as a grid point; alpha = 0 is the mandatory "does any entropy help" control | Yes (SD-SAC) |
| Annealed-target SAC | Xu et al. (arXiv 2112.02852, workshop) | No |
| Mechanism-level SAC (uniform within class) and two-level Hybrid-SAC (learned within-class choice, alpha_within = 0) | Delalleau et al. (arXiv 1912.11077) | No |
| Class-size-corrected drug SAC | Derived from the identity; no source | n/a |
| DQN, for rungs 2 and 3 | Weaver et al. configuration from evo_dm (GPL-3.0); CleanRL `dqn.py` | Yes |

*For the late completeness batch (plan ranks 3 and 7):*

| Comparator | Strongest published version | Reference code |
|---|---|---|
| PPO with entropy bonus | CleanRL `ppo.py` at e421c2e; SB3 Zoo configs as the grid's starting points | Yes |
| PPO-Lagrangian | OmniSafe `PPOLag.yaml` at 15603dd; safety-starter-agents at 4151a28 | Yes |
| PID-Lagrangian, the stronger multiplier | Stooke, Achiam and Abbeel, ICML 2020 (UNVERIFIED, not opened this session) | UNVERIFIED |
| SW-UCB and D-UCB | Garivier and Moulines (arXiv 0805.3415) | Yes: SMPyBandits at 012fc13 |
| EXP3.S | Same SW-UCB paper's benchmark | Yes: SMPyBandits family |
| Recovering-bandit planner | Pike-Burke and Grunewalder 2019 | No |
| Delay-aware RL | Derman et al. ICLR 2021 | Links github.com/galdl/rl_delay_basic (not opened) |
| Recurrent SAC without the pending-occupancy feature | No source | n/a |

The bandits are labelled as comparators that see no state.

*For the phase diagram, the policy-class references are what matter:*
- **Committed schedule:** the best open-loop schedule optimised for expected cost over the noise model (sample-average approximation over noise paths). The certainty-equivalent "planned-once" plan is not the best committed schedule under noise. "Feedback beats a committed schedule" is only a problem-class statement if the committed comparator is optimal in its own class.
- **Feedback:** the existing every-quarter replanner, which is certainty-equivalent MPC, and the optimal-under-noise reference from Q5.
- **Clinical heuristics:** reactive cycling and informed switching from Kouyos, Abel zur Wiesch and Bonhoeffer 2011 (PLoS Comput Biol 7:e1001094). Myopic greedy is the repository's version of these.

**Q4. The cross-resistance matrix C.** In the main experiments, agents do not get C in their networks.
- Drugs enter as one-hot identities. The observation is drug-indexed: the base environment without `MechanismObservation`, which needs no edit to `ocmdp/envs`.
- OC-SAC sees C only through its entropy map, because that map is the method. The true-C versions (embeddings and observation map) stay as a secondary "privileged" arm.
- **Why:** with C in tied networks, the comparison is settled analytically by the identity, so there is nothing to measure.
- **This is partly empirical, and one dev launch settles it.** Both agents on PairedDrugs, dev seeds 0–4, sigma 0, with raw K-way heads and drug-indexed observations:
  - The head is in `ocmdp/components` or `ocmdp/agents`, which are not locked.
  - Log within-class entropy, temperatures and the paired difference.
  - If within-class entropy no longer sits at log 2 by construction, the no-C design carries real content and is adopted. If it still does, the identity is not the reason.
  - The check is fixed before the run.

**Q5. A reference that is optimal under noise: yes, by dynamic programming. No scenario-tree MPC.**
- **Scope:** small cells only. Two classes, one class per quarter (the agents' policy class, with the existing `oracle[one drug per quarter]` as its noise-free counterpart), delay of at most 4.
- **State:** resistance per mechanism (2) on a grid, occupancy (1) on a grid, and the pending queue, which is exactly 2^d binary states. Gaussian quadrature over the resistance shock, backward induction over the 40-quarter horizon, interpolation between grid points. This should fit in numpy; I have not measured its cost.
- **Validation built in:** at sigma 0 it must reproduce the existing planner within its discretisation error. The tolerance is fixed before the run, and the grid is refined until the value changes by less than that tolerance.
- **Where it lives:** outside the locked directories (for example under `experiments/`). Putting it in `ocmdp/baselines` or `ocmdp/theory` is the owner's step.
- **Why it matters:** without it, every sigma > 0 contrast measures the certainty-equivalent planner, not the problem class.

**Q6. Instance family for test E.**
- **Primary source:** Oz et al. 2014 (Mol Biol Evol 31:2387; table S2, normalised resistance of evolved populations against 22 drugs; licence CC BY-NC 3.0, which permits derivatives).
- **C:** built from S2 by a rule fixed in the preregistration before any agent sees it: the mean over populations per drug pair, clipped to [0, 1], and row-normalised as `mechanism_assignment` does.
- **Grouping rule, fixed now:** the paper's own Table 1 target column as printed, giving K = 22, M = 6, n = (7, 5, 3, 3, 3, 1). The authors' grouping, not ours.
- **Replication instance:** Lazar et al. 2014 (CC BY, 12 drugs). Its near-equal classes make it the near-zero-imbalance anchor. Its drug list is UNVERIFIED and must be checked against its Table 1.
- **Excluded sources:**
  - Imamovic 2018: CC BY-NC-ND, which bars a released derived matrix.
  - Sakenova 2025: inferred, undirected categorical calls.
  - Maeda 2020: mostly non-antibiotic stressors.
- The S2 file format and completeness are UNVERIFIED; it was not downloaded.
- **Scope of the earlier separation sweep:** the synthetic two-factor crisp grid shrinks to a dev-tier check of the identity's prediction. Its imbalance axis is exactly the reward bias the identity names.

## 2. How the plan meets each criterion

1. **Novelty.**
   - *Phase diagram against Tennenholtz et al. ICML 2023 (DCMDP):* "DCMDP gives a general history-dependent model and regret bounds; we derive, for the delayed exponentially weighted special case, where committed schedules are optimal or near-optimal, and confirm it at parameters held out from the derivation."
   - *Against rebounding bandits:* cyclic optimality there is shown for identical, uncoupled, undelayed arms without a persistent state.
   - *Against Kouyos et al. 2011, the closest applied prior:* they found informed switching beats mixing largely because of ward stochasticity. We characterise that boundary jointly in delay, memory and noise, with optimal references for each policy class.
   - *Test E against Baram et al.:* "MinRed's equivalence is identical next states, so with no exactly redundant actions it reduces to action entropy; H_Z uses a measured, graded similarity." A short search found no prior similarity-kernel policy-entropy regulariser. The nearest were SAVGO (arXiv 2605.00787, similarity used in policy improvement) and VendiRL (skills). The search was cut short, so this is UNVERIFIED as a non-pre-emption.
2. **Falsifiability and power.**
   - *Phase diagram:* pass, fail or inconclusive per contrast against a margin, with the theory-free null as the competitor (code exists). One contrast (`planned_once_minus_greedy`) can fail in either sign.
   - *Test E:* logged paired standard deviations of OC-SAC minus SAC at sigma 0 on PairedDrugs, five seeds each:
     - 0.0023 (`runs/train/train_20261116_233400_32604.json`);
     - 0.0049 and 0.0047 (`train_20261123_184321_12020.json`, `train_20261123_184953_15052.json`);
     - 0.0177 (`train_20261123_190717_20498.json`);
     - 0.0024 (`train_20261123_190750_20538.json`).
   - *My arithmetic, not a result:* with n ≈ (2.8·sd/Δ)² + 2, a 0.005 effect at 80% power needs about 10 seeds at sd 0.0049 and about 100 at sd 0.0177. The 100 test seeds therefore cover the worst logged spread, but only just.
   - The paired spread on the K = 22 instance is not measured. It must be measured on dev and written into the preregistration before freezing. If the dev estimate needs more than 100 seeds, the preregistration states the resolution floor instead of running.
3. **Value per unit of work.** The phase-diagram planner cells need no training. The DP reference is days of work. Test E is one instance and one confirmatory run, not a sweep. PPO is deferred to one batch.
4. **Survives nulls.**
   - If the boundary is confirmed, it is a characterisation. If it fails, the theory-free null beating the theory is itself reported.
   - If E is null, the equivalence margin licenses "structure-aware entropy buys nothing as a learning aid here". Together with the identity, that is a coherent negative result.
   - The bridge stands on its own.
5. **Comparator strength.** Every comparator is in its strongest published form. The committed-schedule comparator is upgraded to its optimal form under noise. MinRed is included even though it erases the crisp-class story.
6. **Compliance.**
   - Confirmatory runs only under new dated preregistrations.
   - The instance and grouping rule are fixed now, from the source's own table, so there is no instance search.
   - The rung order is respected for agent claims.
   - Nothing touches the locked directories without the owner.

## 3. The strongest objection, and whether it is fatal

"Feedback beats an open-loop schedule under noise" is a textbook fact of stochastic control (the value of closed-loop over open-loop control, which certainty-equivalent and open-loop-feedback methods exist to exploit). A boundary that only restates "noise above zero helps feedback" would be a heatmap of the obvious. Kouyos et al. already showed the stochastic effect in a ward model.

This is not fatal if:
- the confirmed content is the boundary's location as a function of delay and memory at held-out parameters, scored against the theory-free null;
- both policy classes are represented by their optima (Q5, and the open-loop schedule optimised under noise).

It is fatal if the derivation only predicts the sign at sigma = 0 against sigma > 0. That is checkable when the derivation is delivered, and I make no prediction about it here.

Second objection: test E's novelty rests on an incomplete search. That is not fatal, but a pre-emption check (lit-scout on similarity-sensitive and kernel policy entropy) must clear before its preregistration is written.

## 4. What would show this proposal is wrong

- **The DP reference:** if at sigma 0 it fails to match the existing planner within its discretisation tolerance, the Q5 design is wrong.
- **The Q4 dev launch:** if within-class entropy still pins at log n under raw heads, the no-C rationale is wrong.
- **Test E's spread:** if the dev paired spread on the Oz instance needs more than 100 seeds for the preregistered margin, E is underpowered and should be dropped, not run.
- **The phase-diagram scoring:** if the theory-free null predicts the held-out crossings as well as the derivation (E minus E_null not below zero), the primary contribution reduces to a figure.
- **Prior art:** if lit-scout finds a similarity-kernel policy-entropy regulariser, E is pre-empted and dropped.

## 5. Prior work

- Kouyos, Abel zur Wiesch and Bonhoeffer 2011, "Informed Switching Strongly Decreases the Prevalence of Antibiotic Resistance in Hospital Wards", PLoS Comput Biol 7:e1001094 ([PMC3048378](https://pmc.ncbi.nlm.nih.gov/articles/PMC3048378)). The closest applied precedent for the phase diagram. This should be filed to inbox as RELATED and possibly PRE-EMPTS for the "noise flips the ranking" sentence.
- Uecker and Bonhoeffer 2021, J R Soc Interface 18:20210308 ([PMC8385374](https://pmc.ncbi.nlm.nih.gov/articles/PMC8385374/)). A review: reactive cycling results are inconclusive across models, and no information-delay modelling is reported.
- Beardmore et al. 2017, Mol Biol Evol ([PMC5400377](https://pmc.ncbi.nlm.nih.gov/articles/PMC5400377)), and a 2022 hospital switching-time study ([PMC8941386](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8941386/)), the latter not opened.
- SAVGO, arXiv 2605.00787 ([link](https://arxiv.org/pdf/2605.00787)); Diversity Actor-Critic, arXiv 2006.01419; weighted-entropy SAC, arXiv 2011.09083 ([link](https://ar5iv.labs.arxiv.org/html/2011.09083)). These were seen in search results only.
- Already filed in inbox: Baram et al. 2021, Tennenholtz et al. 2023, Rank et al. 2024, rebounding bandits and linear bandits with memory, abx_amr_simulator, Mohammadpour et al. 2024, Weaver et al. 2024, and the data sources (`inbox/2026-11-16-*.md`, `inbox/2026-11-23-*.md`).

## Contradictions and loose ends for the owner

- The criteria are not contradictory. Two things need the owner's ruling:
  - whether planner-only phase-diagram work counts as "new method work" under rung 1;
  - "equal treatment cost" in the project claim has no operational definition in any file I read, and test E needs one before its preregistration.
- Kouyos et al. 2011 should be filed to inbox per the gathering rule. I wrote nothing.

Files read: `arena/2026-11-23-scope-criteria.md`, `decisions/2026-11-23.md`, `decisions/2026-11-16.md`, every file in `inbox/`, `README.md`, the run reports named above, `ocmdp/core/coupling.py`, `experiments/train_agent.py`, and the plan at `<a scratch file outside the repository>`.
