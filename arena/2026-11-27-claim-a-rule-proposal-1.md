# Proposal 1

I recommend making the paired comparison against the theory-free null the primary test, not E against the margin. The null scored as if it were the theory then fails by construction. E's interval is reported as a secondary quantity. I read only the files listed in the task, nothing under predictions/ and no runs/phase values; nothing below uses a dev-map or contrast value.

**One tension in the criteria.** Turning a per-cell half-width into a "resolvable log-ratio error" needs the contrast's slope through the crossing, and a slope is a quantity of means. An assumption in units of the contrast's own SD makes n independent of the spreads. An absolute assumption needs a scale the spreads cannot supply. I close it with an assumption stated in units of the outcome's own spread (A2, section C), so the spreads do enter.

## A. Primary decision rule

```yaml
primary_quantity: D = E - E_null   # median over the 40 held-out lines of |log(sigma_hat/sigma_pred)|, theory minus null
test: 95% percentile bootstrap interval of D, field E_minus_E_null_ci95 of experiments/score_phase_predictions.py
margin: 0
resamples: 10000
seed: 0
command: .venv/Scripts/python experiments/score_phase_predictions.py <stage-2 predictions file, path+SHA-256> runs/phase/<kappa0.2 report> runs/phase/<delta0.1 report> --null <null file, section E> --margin 0.34657359027997264 --resamples 10000 --seed 0
verdict:
  pass: upper bound < 0
  fail: lower bound >= 0
  inconclusive: otherwise
sentences:
  pass: "At both held-out settings, the blind theory places the crossings of planned-once minus greedy closer than the theory-free null, which carries each dev crossing over unchanged (E - E_null 95% interval [lo, hi])."
  fail: "The blind theory does not place the held-out crossings closer than the theory-free null (E - E_null 95% interval [lo, hi])."
  inconclusive: "The held-out test was inconclusive at the preregistered power: the E - E_null 95% interval [lo, hi] contains 0."
```

- **Why D.** `null_phase_predictions.py` already commits the project to the standard: "A theory earns the word 'predicts' only by placing crossings better than this."
- **Why margin 0.** Any other margin would have to be justified from dev values.
- **No scorer change.** The mapping reads the interval the scorer already prints.
- **Resamples and seed.** 10 000 resamples put about 250 draws in each tail, against about 50 at the default 2 000. Seed 0 is the scorer's default and is not chosen.

## B. Secondary quantities

- **E.** Report E, E_ci95 and the scorer's own verdict at log √2. Only when the primary passes and E's verdict is "pass", the sentence may add ", with median error interval below log √2". Otherwise E is reported as an interval only.
- **E_null.** Report its point value, with the D interval beside it. If the verdict is fail or inconclusive and E_null ≤ log √2, append this pre-written clause: "the held-out settings moved the measured crossings less than log √2 on the median line, so this test separates the theory from carry-over only weakly."
- **Direction checks.** Report every `directions` row with log_shift, ci95, wrong_sign and censored. These are descriptive, not tests.
- **Censored lines.** For the measured, predicted and null crossings, report the count placed below the grid (0.1), above it (0.4) and in the interior. Also count lines where the measured and predicted crossings sit on the same censored side (error 0 by construction). Do the same for the null.

## C. Episode count

Script `experiments/claim_a_episode_count.py <dev map>` prints n only, using `stats.t_quantile`. For each of the 80 dev cells:
- s_c = SD (ddof 1) of the per-episode values of planned-once minus greedy;
- u_c = sqrt((SD_planned² + SD_greedy²)/2), the pooled outcome SD.

Then:
- ρ_c = s_c/u_c over cells with s_c > 0 and u_c > 0;
- R = 75th percentile of ρ_c (numpy linear).

```
d* = 0.5;  eps* = log(sqrt 2)/2 = 0.17329;  n_max = floor(0.9 * 600 / 4.875) = 110
n_req = min{ n >= 2 : 2 R t_{0.975,n-1} / sqrt(n) <= eps* }
n = min(max(n_req, 40), n_max)
```

- **Half-width per cell:** h(n) = t_{0.975,n-1}·s/√n. That is 0.3198·s at n = 40 (the design constant) and 0.1890·s at n = 110.
- **Resolvable log-ratio error:** ε(n) = 2R·t_{0.975,n-1}/√n, which is 0.640·R at n = 40 and 0.378·R at n = 110.
- **Where ε comes from.** Across a factor-2 sigma interval, the crossing fraction f = m0/(m0 − m1) has SE ≤ max(s0, s1)/(√n·|Δm|) for any correlation between the two cells. Also, δ log σ̂ ≤ δf.
- **Assumptions:**
  - A1, spread transfer: held-out ρ ≤ R on the cells that matter.
  - A2: across the interval holding a crossing, the mean contrast moves by at least d* = 0.5 pooled outcome SDs (Cohen's "medium").
  - First-order (delta-method) noise.
- **If the cap binds,** the preregistration states that ε(110) exceeds ε* before any data exist.
- **Floor of 40:** the test is never less precise than the map that feeds the null.

**Feasibility.** Time = 160/80 × n/40 × 65 min × 30/20 = 4.875·n min. That gives 195 min (3 h 15) at n = 40 and 536 min (8 h 56) at n = 110, which leaves 10% headroom inside 600 min. The count is fixed at W = 20. If the guard admits W < 18 (where 110·4.875·20/W > 600), stop and go to the owner; n is not recomputed. Both settings launch as two test-tier reports with the same `--episodes n`.

## D. Exclusions, all structural

- **No line is excluded.** The scorer gives an unpredicted line infinite error, and reports are never edited, so the scored set is fixed at the 40 lines.
- **Zero-width series.** `crossing()` already skips cells whose mean is exactly 0. A line of all-zero cells gets sign 0 and is placed at 0.4 ("not crossed"). Report the count.
- **Bitwise-identical policy pairs.**
  - When planned-once equals greedy, this is the zero-width case above.
  - Pairs among the other policies do not enter the primary contrast and are counted only.
- **Censoring.** Use the scorer's clamping to [0.1, 0.4], unchanged. Theory and null censor identically, so in D the "same side agrees" credit cancels.
- **Incomplete or non-finite reports.** No verdict; the sentence is "the test did not complete".
- **Multiplicity.** The verdict is one test on one scalar, so no correction applies. The 40 per-line errors and the direction checks are labelled "uncorrected, descriptive".
- **Shared episodes.** All 160 cells share test episodes eval_seed(1000) to eval_seed(1000 + n − 1). The bootstrap resamples them jointly, so per-line errors are dependent and are not 40 independent tests. The paper says so in those words.

## E. Which report feeds the null

The null comes from `runs/phase/phase_20261126_184038_27744.json` (SHA-256 `0389151cc481ed8defa42d3c5b8f75fb8c73cf537d8d748c4da47c567f9acf79`). The preregistration can pin that file before run 1 exists. The command is `experiments/null_phase_predictions.py <map> experiments/claim_a_heldout_settings.json --contrast planned_once_minus_greedy --sign-beyond-crossing <the predictions file's field; +1 per decisions/2026-11-27.md>`.

The gate is run 1, and the test launches only if all of these hold:
- `tools/compare_phase_reports.py <18:40 map> <run 1> --ignore-arg workers` prints exactly "PASS: 480 results, 96000 values compared, 0 mismatched, 0 missing or malformed fields", with no setup FAIL;
- the comparator's full SHA-256 is recorded;
- the null script's output on run 1's report is byte-identical to its output on the 18:40 map.

This is equivalent to decisions/2026-11-26.md:381 ("run 1 feeds the null") whenever the gate passes.

## F. Severity check

Score the null file as both the predictions and `--null`. On every resample, `median_error` gets identical inputs for `boot_e` and `boot_null`, so `diff` is identically 0. `_percentiles` then returns [0.0, 0.0]. The upper bound is not below 0, so the rule cannot pass. The lower bound is ≥ 0, so the verdict is fail, with probability 1, whatever the data.

The E-only rule behaves differently. If the doublings shift the median line's crossing by less than a factor of √2, the null passes. That includes the case where every line is censored on the same side, where E = 0 and the upper bound 0 is below the margin. A theory that learned nothing beyond dev would then be licensed "within the margin". The test should never credit carry-over, so this guaranteed fail is the right property.

## G. Risks and what a hostile reviewer would say

1. **Strongest objection.** If the doublings barely move the crossings, even a correct theory cannot pass. This is not fatal: in that case the held-out data cannot separate theory from carry-over, and the clause in B says so.
2. **The null is not resampled.** Its dev noise from 40 episodes is not propagated, and a noisy null is easier to beat. The null is a fixed predictor, as the theory is; the paper states this.
3. **Bootstrap quality.** A percentile bootstrap of a median of discrete, censored errors has uncertain coverage (Efron 1979 on median discreteness). The boundary at "lower ≥ 0" falls on exact ties.
4. **Arbitrary constants.** d* = 0.5 and the 75th percentile are conventions, and the cap at 110 may bind.
5. **Unmeasured time.** The 4.875 min per episode is derived, not measured, and kappa 0.2 or delta 0.1 may change planner cost.
6. **Possible scorer refusal.** If the stage-2 file holds base-setting rows, `load_predictions` refuses ("predicted lines with no measurement"). Any filtering of its rows must be scripted and hashed before the test; I did not open the file.

**What would show I was wrong:**
- the share of held-out ρ_c above R (A1);
- the pre-stated E_null clause (whether the doublings moved the crossings enough);
- run 1's measured wall time at W against 100 min.

**Prior work:** I ran no literature search; these are standard references, cited from memory.
- Diebold & Mariano (1995, JBES), paired test of predictive accuracy on loss differentials;
- Hyndman & Koehler (2006, IJF), MASE, which scales errors against a naive carry-forward forecast;
- Murphy (1988), skill scores against persistence;
- Mayo & Spanos (2006, BJPS), severe testing;
- Cohen (1988), effect-size conventions.

**Files to add:**
- new: `experiments/claim_a_episode_count.py`, which prints n only;
- used unchanged: `experiments/score_phase_predictions.py`, `experiments/null_phase_predictions.py`, `experiments/claim_a_heldout_settings.json`, `tools/compare_phase_reports.py`.
