# Outcome: the Claim A preregistration's decision rule, episode count and exclusions

## How it was decided

- **Criteria:** `2026-11-27-claim-a-rule-criteria.md`, written before the proposers were spawned and not changed afterwards.
- **Proposals.** Three proposers with identical briefs, each forbidden to open `predictions/` or any phase-report value. Their hand-backs are `-proposal-1.md` to `-proposal-3.md`, verbatim, numbered in shuffled order. Proposal 2 disclosed that a search over `arena/` printed two lines of older-map values (24 November files) it did not ask for; it traces every constant to a committed file and the judge was told to weigh the exposure.
- **The judge** worked blind from the criteria and the three files, and read the scorer and the null to check claims about them. Ranking: proposal 2, then 3, then 1, with small gaps. No proposal broke a hard constraint.
- **Adopted: proposal 2, with the amendments A1 to A9 below.** Decided by the arena's scoring, under the owner's instruction of 2026-11-26 that the arena decides. The session that wrote the criteria and this outcome has read no dev-map value; it has read the stage-2 derivation's text, so it chose nothing from spreads, and every spread-dependent quantity (n) comes from a script that prints n alone.

## Convergence

All three proposals converged on the same primary rule: D = E − E_null with margin 0, the scorer's 95% percentile interval over 10 000 joint episode resamples at seed 0, pass if the interval lies below 0, fail if its lower bound is at or above 0, otherwise inconclusive; no exclusions; a floor of 40 episodes; the same severity argument (the null scored as the theory gives D identically 0 and fails). Identical briefs give correlated answers, and criterion 1 nearly dictates D, so this is weak evidence that the rule is right. The judge said so.

## Where they disagreed

- **Episode-count formula.** Proposal 1: R = 75th percentile of the ratio of the contrast's SD to the pooled outcome SD, ε* = log √2 / 2, cap 110. Proposal 2: SD* = the largest cell SD, tau = 0.005 (half the committed margin m = 0.01 of `arena/2026-11-24-design-outcome.md`), cap 100. Proposal 3: S = 90th percentile of the nonzero cell SDs, h* = 0.01, cap 100. Proposal 2 was adopted: its constant is the only one traced to a committed decision, and its power statement is the only one that corrects for simultaneity over the 40 lines.
- **Which report feeds the null.** Proposal 1: the 26 November map, gated on run 1 matching it. Proposals 2 and 3: run 1, with the null's hash pinned from the 26 November map. Operationally the same when the gate passes; the preregistration pins the map and the null's content hash and makes run 1 the gate.
- **Scorer changes.** Proposal 1 none; proposal 2 a `verdict_vs_null` key and a refusal when the null's contrast or sign differs from the theory's; proposal 3 a separate verdict script. Proposal 2's two additions were made; E and its interval are untouched.
- **E in the claim sentence.** Proposal 2 lets E add nothing to the sentence; proposals 1 and 3 allow a gated "within the margin" clause. Proposal 2's reading stands: E is descriptive.

## Amendments

- **A1 (from proposal 3).** The episode-count script refuses a dev map in which a line's contrast mean is exactly 0 at every sigma, since the null would then carry a line with no sign, which the scorer refuses. The pinned map passed this check. The script is also rerun on run 1 and must print the same n.
- **A2 (from proposal 3).** The null is scored as if it were the theory on the real held-out data, as a secondary quantity with its own E and verdict against log √2. If it "passes" the margin, the margin sentence is stated to carry no predictive content.
- **A3 (judge, missed by all).** Every sentence is limited to these 40 lines at these two settings: the bootstrap resamples episodes, not lines.
- **A4 (judge).** A tie with the null (lower bound of D exactly 0) is a fail, and the preregistration says so.
- **A5 (judge).** The calendar is stated: about 9 h 48 of host time after the commit at 20 workers, one guarded launch at a time; with W below 17 the ten-hour budget is exceeded and the work stops for the owner, with n unchanged.
- **A6 (judge).** No number from the test enters a draft before the run-auditor and the adversarial-reviewer pass.
- **A7 (adversarial reviewer, 27 November).** The derivation's predictions file carries 60 lines, 20 of them at the dev setting, which the test does not measure and which the scorer would refuse. `experiments/claim_a_filter_predictions.py` drops them without touching a value; its output's content hash is pinned. The one direction check it predicts starts at the dev setting and is not scored. Only the file's point values are scored; no other column of the deliverable excuses a line. The derivation assumed 20 evaluation episodes against the map's 40 and the test's 100, which is disclosed.
- **A8 (adversarial reviewer).** The censored counts are reported for the measured, theory and null crossings, with the number of lines whose error is 0 because both sides are censored alike; 21 or more of 40 makes the median 0 by construction (proposal 3 said 20; `np.median` over 40 values averages the 20th and 21st).
- **A9 (protocol).** One test-tier launch per preregistration, by the tier guard and by `sweep_phase_diagram.py`, and one (kappa, delta) per launch. Claim A therefore has two preregistration files, `2026-11-27-claim-a-boundary-kappa.yaml` and `-delta.yaml` (superseded the same day, with the rule unchanged, by `-kappa-2.yaml` and `-delta-2.yaml`; see `decisions/2026-11-27.md`), stating one joint rule; the verdict is computed once, after both reports exist. The alternative, carrying two settings in one report, would change the sweep and the scorer on the critical path and was not taken.

## Not adopted

- Proposal 1's episode formula and its clause conditional on E_null's size; proposal 3's separate verdict script (the scorer's key does the same).
- The adversarial reviewer's alternative primary statistic, a mean over "discriminating lines" where theory and null disagree by more than log √2, with a constant-floor null. It was proposed after the reviewer had read the predictions file, so adopting it would choose the rule from what the theory predicts. It is recorded here for a later preregistration on a fresh held-out family.

## What the proposals could not supply

Criterion 3 asked for the smallest log-ratio error the test resolves, from spreads alone. That needs the slope of the contrast through the crossing, a mean. The preregistration therefore states power as a function of Δ_min, the smallest change of the mean contrast across a crossing bracket, and reports the achieved Δ_min after the test.

## Owner acts

- Commit the merged tree, the stage-2 derivation files, the two preregistration files, the arena files of today and the new scripts; create the two approval files when ready.
- The launches and the scoring run on rtx-1 in the order the preregistration states.
