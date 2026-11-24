# Proposal 2

## 1. Recommendation

**Target.** Weaver et al., PNAS 2024 (DOI 10.1073/pnas.2303165121), base case on the Mira 2015 landscapes (15 beta-lactams, N = 4, 20-step episodes, 500 training episodes). It is still the only published RL drug-cycling result I can find on measured landscapes with released code. Two web searches turned up no follow-up or competitor. The application search that was cut short (inbox/2026-11-23-application-queries.md, "Not run") should be finished before the preregistration is frozen. It is about an hour of work, and if it turns up a target with numeric tables and pinned code, this decision should be reopened.

**Route: clean-room reimplementation in this repository, specified from the paper, the SI and a read-only reading of the authors' code. The authors' code is not run inside the gate.**
- **Specification file.** Write a spec before any code. Every unstated detail gets one frozen reading and a source, in this order of precedence: published text, SI, `evo_dm` at 16335ce (the last commit before publication, after the delay feature 46ec357), then `rl_cycling` at d68cf07. A reading is chosen without running anything. Alternative readings are listed but never run to "see which matches".
- **Licensing.** Reading GPL code to learn facts (hyperparameter values, the algorithm) is fine. No code text is copied. `rl_cycling` has no licence, so it is read, never copied and never run.
- **Landscapes.** Transcribe Mira 2015 Table 4 (CC BY) twice, independently, and diff the result against the hard-coded list form in `evo_dm`. The list form is what the authors ran (`as_dict=False` default), so it governs. Every Table 4 disagreement, including ZOX genotype 2 (0.5 vs 0.805), is reported before any run and never "fixed".
- **Baselines computed exactly.** Random cycling, all 105 two-drug cycles and the 20-step backward-induction MDP are Markov-chain quantities. Compute them by propagating the state distribution, not by Monte Carlo.
- **DQN as coded at 16335ce:** plain DQN target, Conv1D(64,3) x2, MaxPool, Dropout 0.2, Dense 28, Adam, MSE, and the `evol_deepmind` defaults (lr 1e-4, minibatch 60, target update 310, gamma 0.99, min_epsilon 0.005, train_freq 100). The pre-freeze read must confirm how `practice()` applies train_freq.
- **Replicates.** 100 replicates on dev seeds 0–99 (seeds.yaml has exactly 100). The same seed list is used for RL-genotype and RL-fit, and each seed controls network init, replay sampling, epsilon draws and the environment.
- **Evaluation.** Greedy (epsilon 0) for 500 episodes after training, matching the harness's pre_trained rerun. The metric is mean true population fitness per episode.

**Facts I settled read-only today, which the spec must handle:**
- **Noise.** `evol_env` adds one N(0, (0.05 x noise_modifier)^2) draw to the sensor fitness only; the reward uses true fitness. "Noise modifier 40" therefore means a sensor standard deviation of 2.0.
- **Normalisation scales disagree.**
  - `evol_deepmind` defaults to `normalize_drugs=True`, which rescales each landscape to [0,1].
  - `policy_sweep` and `mdp_mira_sweep` default to False.
  - The published 1.41 and 1.88 are both above 1, so that panel must be unnormalised.
  - The spec must state which scale each comparison is on.
- **No harness for the noise-40 panel.** `rl_cycling/R/test_harness_noise.R` at HEAD runs only noise_modifier 100 with `iter = 10`. The 1.41/1.88 panel's exact settings are not in the release.
- **The SI is still unopened.** PMC's PDF link returned a download stub and PNAS returns 403. The user may have to fetch it by hand. It is a pre-freeze read, not a blocker.

**Conditional, only after a stop and only with the user's approval:** a differential run of the authors' code to tell whether a failure is ours or theirs.
- Clone `evo_dm` at 16335ce to a directory outside the repository and review all of its source.
- Run it in a container: Python 3.10, tensorflow-cpu 2.11, pinned dependencies, `--network none` at run time, source mounted read-only, output directory outside the repository.
- Drive it with our own small script that sets seeds and passes the harness's arguments. The unlicensed R code is not used.
- A repository script ingests the outputs into runs/ with file hashes.
- Nothing is installed into .venv.

**Tolerances, all fixed now.** Any failure means report and stop.

| ID | Reported quantity | Criterion |
|---|---|---|
| E1 | Landscape values | All 240 values agree with Table 4 to its printed precision. Every disagreement is listed before any run. This is an audit item and does not stop the run on its own. |
| E2 | Random cycling 1.88 at noise 40 | The exact expected value, unnormalised, from wild type, over 20 steps, lies in 1.88 ± 0.03. The margin allows for the authors' Monte Carlo error over 10 x 500 x 20 steps with a sensor standard deviation of 2.0. |
| E3 | MDP uses CTX in more than 50% of time steps | The exact CTX share along the optimal policy's trajectory distribution is above 0.50. |
| R1 | RL-genotype beats random and all two-drug cycles in 100/100 | At most 5 of 100 replicates fail. A replicate passes if its 500-episode mean is below the exact random value and below the exact value of every two-drug cycle. |
| R2 | RL-fit beats random in 98/100 | At most 5 of 100 replicates fail. |
| R3 | RL-fit mean 1.41 at noise 40 | The 95% bootstrap CI of our mean over 100 replicates lies entirely inside 1.41 ± 0.12 (0.12 is a quarter of the published 0.47 gap to random). |
| S1 | Edge-weight correlation 0.96 (RL-genotype vs MDP) | The 95% CI over replicates lies inside 0.96 ± 0.04. The edge definition is taken from `prep_figs.R` and frozen in the spec. |
| S2 | Edge-weight correlation 0.82 (RL-fit vs MDP) | The 95% CI lies inside 0.82 ± 0.10. |
| D | Delay: d ≤ 1 matches the base case, d > 1 is no better than random | Descriptive only, reported with intervals. The published claim is qualitative. |

My hand arithmetic, to be confirmed by a script before freezing (this is the derivation of the rule, not a measured result): against 100/100, a two-sided Fisher exact test at alpha 0.05 first rejects at 6 failures, so "at most 5" matches it. At a true rate of 0.98, the chance of 6 or more failures is about 0.02. At a true rate of 0.93 it is about 0.7. So R1 and R2 can fail.

**Negative controls, preregistered.** Two crippled agents run on the same 100 seeds: an untrained network acting greedily, and DQN trained on a reward shuffled across time steps. Both must fail R1 and R2. If either passes, the gate is declared non-discriminating and that is reported. It is never reported as a pass.

**If the numbers are not reproduced:**
- Report, for every criterion, our value with its interval, the published value, the difference and the spec reading used. Then stop.
- No hyperparameter, reading or landscape value changes after the first run.
- Rungs 2 and 3 do not start.
- Next steps are the user's call: (a) the conditional differential arm; (b) the user asks the authors for the figure commit and the unreleased `.Rda` outputs (that is a message, so it is not done by an agent); (c) if a spec error is found by a test written against the paper rather than by the outcome, a new dated preregistration with both runs reported.
- A partial outcome, where E1–E3 pass and R fails, still establishes the shared environment, MDP and baselines. Only the user decides whether that licenses rung 3.

**From rung 1 to rungs 2 and 3:**
- **Rung 2.** On the reproduced environment, unmodified:
  - OC-SAC and discrete SAC (Christodoulou 2019), with per-agent targets c·log M and c·log K. The SAC target follows the dynamic rule of Mohammadpour et al. 2024, the strongest honest form.
  - The reproduced DQN as reference, and the MDP as the upper bound at sigma 0.
  - Mechanism classes come from an external classification fixed before any run, for example WHO ATC level 4. My recollection is penicillins (AMP, AM), inhibitor combinations (AMC, SAM, TZP), 2nd-generation cephalosporins (CEC, CXM, CTT, CPR), 3rd-generation (CTX, ZOX, CRO, CAZ, CPD) and 4th-generation (FEP), giving M = 5 and K = 15. This must be checked against the ATC index.
  - A tie is the expected result, judged by a preregistered equivalence margin expressed as a fraction of the random-to-MDP gap.
- **Rung 3.** Exactly one added mechanism: fitness under drug a is modulated by the delayed, exponentially weighted occupancy of a's class. Setting the coupling strength to 0 must reproduce the rung-1 environment exactly on the same seeds, and that becomes a unit test. The observation carries the pending occupancy for every agent, asserted. The published DQN configuration is "the published approach"; a parity-tuned DQN is reported separately under its own label. The functional form and parameters go to their own arena and preregistration. Weaver's measurement delay stays at 0 so it is not confused with action-effect delay.

## 2. Against each criterion

1. **Shared ground.** Every gated quantity (E2, E3, R1–R3, S1, S2) is a number or count stated in the published text. Nothing is digitised from figures, because choosing which points to read is a forking path.
2. **A tolerance that can be fixed now and can fail.** Every rule is numeric and applies to 100 replicates that already exist as dev seeds 0–99. E2 and E3 are exact and cannot be rescued by seed noise. R1 and R2 have a stated chance of failing. Negative controls show whether the gate can discriminate at all. R3, S1 and S2 are equivalence rules, so the burden falls on showing agreement rather than on failing to show a difference.
3. **Auditability.** Every step reruns from this repository: the spec cites a commit and file for each fact, landscapes are transcribed from a CC BY source, and the baselines are exact. No GPL text enters the repository and the unlicensed R code is never used. The only external execution is optional, isolated and logged.
4. **Cost and stall risk.** About 4–5 working days: spec and transcription 1.5, environment and exact baselines 1, DQN 1, runs under a day (about 10^6 environment steps per condition, small networks, CPU), report 0.5. The three stall risks are the unopened SI, the missing noise-40 harness, and the normalisation ambiguity. Each is handled by a frozen reading rather than by iteration.
5. **Use for rungs 2 and 3.** Owning the code is what makes OC-SAC, SAC and occupancy coupling addable. Running the authors' package instead would mean extending GPL code, inside a fragile TF 2.11 environment.
6. **Security.** No downloaded code runs inside the gate. The conditional arm is reviewed, sandboxed, offline and needs approval.

## 3. Strongest objection

**The gate is shallow and the reading is underdetermined.**
- R1 and R2 sit at the ceiling. On a 16-state problem almost any working DQN beats random cycling, so passing them shows little.
- The only quantitative RL anchor, R3, comes from a panel whose harness is not released, whose replicate count the paper itself contradicts (100 in the text, 10 in the caption), and whose scale depends on a normalisation choice.
- So a pass may be close to automatic, and a fail may come from a wrong reading rather than a wrong implementation.

**Not fatal.**
- E2 and E3 are exact environment checks and genuinely severe.
- The negative controls detect a toothless gate.
- Reading choices are frozen before any run.

It does limit what can be claimed: the rung-1 sentence must name the quantities reproduced, not "the paper".

There is also a tension, short of a contradiction, between criteria 1 and 2. The reported quantities carry no dispersion, so every margin (0.03, 0.12, 0.04, 0.10) is a judgement made now. I state them as judgements.

A further caveat for rung 2: all 15 drugs act against one TEM beta-lactamase. Taken literally, "mechanism classes" collapses to M = 1, and the ATC classes are a proxy that a reviewer will question.

## 4. What would show I was wrong

- A negative control passes R1 or R2, meaning the tolerance cannot discriminate.
- After the gate run, the environment-only checks (E2, E3) are recomputed under each listed alternative reading, descriptively and with no rerun of RL. If outcomes flip often, the reproduction is underdetermined and the shared ground is weak.
- If the authors' own code is later run and misses its own numbers, the target was the wrong choice.
- The finished application search finds a target with numeric tables and pinned code.
- The spec is not frozen within 7 working days, which means the route has stalled.

## 5. Prior work

- **Same landscapes, earlier work.** Mira et al. 2015, PLoS ONE e0122283 (source of the landscapes), and Nichol et al. 2015, PLoS Comput Biol (sequential therapy as a Markov chain on the same landscapes). Neither is RL. Both could serve as an optional environment cross-check outside the gate.
- **Reproducing Weaver 2024.** None found.
- **Methodology.**
  - Henderson et al. 2018, "Deep RL that Matters" (AAAI): seed sensitivity.
  - Colas et al. 2018, "How many random seeds?"
  - Agarwal et al. 2021, rliable (NeurIPS): interval reporting and interquartile mean.
  - Lakens 2017, equivalence testing (TOST): the logic behind R3, S1 and S2.
  - Pineau et al. 2021, JMLR, reproducibility checklist.
  - ReScience C, venue for independent replications.

These are cited from memory, not opened in this session.

**Files:**
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/arena/2026-11-23-reproduction-criteria.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/seeds.yaml
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/budget.yaml

Nothing was written or executed.

Sources:
- [PNAS 2024 (PMC11032439)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/)
- [Preprint (PMC9882109)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9882109)
- [evo_dm evol_game.py at 16335ce](https://raw.githubusercontent.com/DavisWeaver/evo_dm/16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0/evodm/evol_game.py)
- [evo_dm exp.py at 16335ce](https://raw.githubusercontent.com/DavisWeaver/evo_dm/16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0/evodm/exp.py)
- [rl_cycling test_harness_noise.R](https://raw.githubusercontent.com/DavisWeaver/rl_cycling/master/R/test_harness_noise.R)
- [Nichol et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567305/)
- [Mira et al. 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0122283)
