# Criteria: the Claim A preregistration's decision rule, episode count and exclusions

Written 2026-11-27, before any run at the held-out settings and by a session that has read no value of any dev map. The owner's instruction of 26 November makes the arena the decision maker for choices that would otherwise wait for the owner.

## The decision

Claim A tests a blind theory of where the contrast `planned_once_minus_greedy` (committed plan minus myopic greedy; positive means greedy fails less) changes sign along the resistance-noise axis sigma, on 20 (delay, lam) lines at two held-out instance settings (`experiments/claim_a_heldout_settings.json`: kappa 0.2 and delta 0.1, one rate doubled at a time; 40 scored lines, 160 cells, sigma grid {0, 0.1, 0.2, 0.4}). The predictions were delivered and hashed before any held-out data existed (`predictions/claim-a-2026-11-26/stage2/`). The scorer is `experiments/score_phase_predictions.py`; the theory-free null is `experiments/null_phase_predictions.py`, which carries each dev line's measured crossing to the held-out settings unchanged.

Four choices are open and must be fixed in the preregistration:

1. **The primary decision rule, one test on one quantity.** The scorer offers (i) E, the median over scored lines of |log(sigma_hat / sigma_pred)|, with a bootstrap percentile interval against a margin (default log sqrt 2), verdict pass / fail / inconclusive; and (ii) the interval of E - E_null on the same resamples. Which one is primary, with which margin, and how the other is reported.
2. **The episode count per cell**, as a formula applied mechanically to the dev map's paired spreads by a script, never by a person who has read the predictions. The design constant on the dev map is: half-width = 0.3198 x paired SD at n = 40 (exact Student-t, df 39).
3. **Exclusions and the stats verifier's open items**, fixed before any gap is inspected: contrast series of zero width (both policies produced identical episodes; 16 of 400 on the dev map); bitwise-identical policy pairs within a cell (43 of 1 200 on dev, reproduced by an independent rerun); lines whose crossing is censored at the grid edge; no multiplicity correction over 40 lines; every cell sharing the same evaluation episodes.
4. **Which report feeds the null**: the 26 November rtx-1 dev map `runs/phase/phase_20261126_184038_27744.json` (reproduced bitwise by `phase_20261126_213058_19724.json`), or run 1's rerun at the preregistration's commit, which must equal it bitwise under `--ignore-arg workers`.

## What the decision affects downstream

- The one sentence Claim A licenses on 29 November: "the blind theory places the held-out boundary within the margin" (pass), "it does not" (fail), or "the test was inconclusive at the preregistered power", and whether "better than the theory-free null" can be added.
- The run time of the only remaining test-tier launch. The 80-cell, 40-episode dev rerun took 65 minutes uncontended at 30 workers on rtx-1; the guard now admits about 20 workers. Derived: 160 cells at n episodes take about 160 / 80 x n / 40 x 65 min x 30 / 20, that is about 3 h 15 min at n = 40 and about 8 h at n = 100. Nothing here is measured at these sizes.
- Whether a null result can be stated as a result, in the preregistration's own terms.

## Hard constraints

- The test launches on rtx-1 after the owner commits the preregistration and creates the approval file, and must have written its report before the morning of 29 November (local time of the host). Run 1, the dev-map rerun at the preregistration's commit (about 1 h 40 min at 20 workers), comes first. Budget the test at no more than 10 hours of wall time at 20 workers.
- The scorer and the null are used as they are in the merged tree. A proposal may ask for a scorer change only if it is a pure addition that leaves E and its interval unchanged, and says so.
- No person who has read the predictions file chooses the episode count or any exclusion threshold. The formula must be computable by `experiments/` code from the dev map alone.
- The rule may not be chosen for its expected outcome. A proposal that reasons from what the theory probably predicts, or from any dev-map value, is disqualified.
- Every number in the preregistration must be derivable from committed files or from the formula; no placeholder.
- One run per preregistration. No second look at the data.

## What would make one option clearly better

1. **Severity.** The rule can fail: a theory that predicts the dev map's own crossings and nothing more must be able to fail it. Credit a rule under which the null's own predictions would be classed "fail" or "inconclusive", not "pass", when E is computed on the null itself.
2. **A single pre-stated verdict**, with pass, fail and inconclusive all defined, and the sentence each licenses written out.
3. **Power stated, not hoped.** The episode formula gives the half-width on each cell's contrast and the smallest log-ratio error the interval on E can resolve, derived from the design constant, with the assumption on the paired SD named.
4. **Exclusions that cannot be gamed**: fixed by a rule on the dev map or on the data's structure (zero width, censoring), never by inspection of the held-out contrast means.
5. **Feasibility** within the time budget above, with the derivation shown.
6. **Honesty of the null comparison**: whether E - E_null is reported as a secondary quantity with its interval, and how the shared-episode dependence and the 40-line multiplicity are described in the paper, not hidden.

## Evidence that would settle it empirically instead

None without contaminating the test: any pilot at the held-out settings is the test itself, and any look at the held-out contrast means is a second look. The episode count is set by a formula on dev spreads, which a script applies; that is part of the decision, not a substitute for it. So this is design judgement, and the arena is the right route.
