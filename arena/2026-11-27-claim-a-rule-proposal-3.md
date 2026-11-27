# Proposal 3

I propose making E − E_null the single primary test, with 0 as the margin. E against log √2 drops to a secondary result. The main reason is that a theory which only repeats the dev crossings gets "fail" under this rule whatever the held-out data say. The scorer needs no change. Everything new is in two new scripts under experiments/. I did not open predictions/ or any runs/ file. Nothing in this proposal uses a dev-map value or anything about what the theory predicts.

**A. Primary decision rule**
- **Quantity:** D = E − E_null. E is the theory's error, E_null is the error of the null from `experiments/null_phase_predictions.py`. Both come from `experiments/score_phase_predictions.py` on the same episode resamples. The interval is the scorer's `E_minus_E_null_ci95`: a 95% percentile interval, resamples 10000, seed 0.
- **Margin:** 0, meaning the theory must place held-out crossings closer than the null does. The default log √2 margin is used only in B.
- **Verdict:**
  - pass if hi < 0 (a one-sided test at 2.5%);
  - fail if lo ≥ 0;
  - inconclusive if lo < 0 ≤ hi.
- **Implementation:** a new file, `experiments/claim_a_verdict.py`. It reads the scorer's JSON and applies this mapping. It is a pure addition, and E and its interval are untouched.
- **Scorer command:** `score_phase_predictions.py <stage-2 predictions file> <report kappa 0.2> <report delta 0.1> --null <null file> --resamples 10000 --seed 0`. The two reports come from two test-tier sweeps (`--kappa 0.2` and `--delta 0.1`), each 80 cells with the same `--episodes n`, so their evaluation episodes are shared.
- **Sentences for 29 November:**
  - Pass: "On the 40 held-out lines the blind theory placed the sign change of planned_once_minus_greedy closer than the theory-free null: E − E_null 95% interval [a, b], below 0 (n episodes resampled jointly)."
  - Fail: "The blind theory did not place the held-out sign changes closer than a null that carries each dev crossing unchanged: the 95% interval of E − E_null, [a, b], does not extend below 0."
  - Inconclusive: "The test was inconclusive at the preregistered power: the 95% interval of E − E_null, [a, b], contains 0."

**B. Secondary quantities**
1. **E against log √2:** E, `E_ci95` and the scorer's own verdict, labelled secondary. "Within the margin" may be said only as that secondary clause, and always next to item 2.
2. **The null scored as if it were the theory:** run the scorer with the null file in both positions, same reports, resamples and seed. This gives E_null with an interval and its verdict against log √2. If the null also "passes" the margin, the paper must say that the margin sentence alone carries no predictive content.
3. **Direction checks:** each one with `log_shift`, `ci95`, `wrong_sign` and `censored`. Also the counts wrong_sign / tested and censored / total. All unadjusted and descriptive.
4. **Censoring counts,** from `claim_a_verdict.py` using the scorer's `crossing` and `line_error`:
   - for measured, theory and null: lines placed below, inside and above the grid;
   - the number of lines whose error is 0 only because both sides are censored on the same side, shown for E and for E_null. If that is 20 or more, the median is 0 by construction, and that must be stated.
5. **Cell intervals and achieved precision:** the 160 cell intervals (exact t), labelled unadjusted. Also the achieved half-width at n, using the held-out 90th-percentile paired SD.

**C. Episode count**
- **Script:** `experiments/claim_a_episode_count.py`. It checks the dev map's SHA-256 (`0389151c…acf79`) and prints n and nothing else.
- **Inputs:** SD_c is the sample SD (ddof 1) of the 40 per-episode planned_once_minus_greedy differences in dev cell c (80 cells, read with `measured_lines`). S is the 90th percentile (numpy, linear) of {SD_c : SD_c > 0}. Zero-width series carry no information about spread, so they are left out.
- **Formula:** n = the smallest n in [40, 100] with t_{0.975, n−1}·S/√n ≤ h*, where h* = 0.01 in failure-rate units. If no n up to 100 meets it, n = 100. The t value comes from `experiments/stats.py` `t_quantile`.
  - The design constants are 0.3198 at n = 40 and t_{0.975,99}/10 = 0.1984 at n = 100.
  - So n = 40 if S ≤ 0.0313, and the cap binds if S > 0.0504.
  - The floor of 40 keeps the test at least as precise as the map the null came from.
- **Named assumption A0:** the held-out paired SD's 90th percentile is no larger than the dev map's. The achieved value is reported (B5).
- **What the half-width resolves:**
  - Call the change in the contrast across the doubling interval that brackets a crossing ([0.1, 0.2] or [0.2, 0.4]) the step size Δ.
  - The interpolated crossing is σ̂ = s0(1+f) with f = |m0|/(|m0|+|m1|). Perturbing the two means by at most h moves f by at most h/Δ, so |Δ log σ̂| ≤ h/Δ to first order.
  - The median is 1-Lipschitz in the sup norm. So E moves by at most h/Δ, and D (two medians with fixed predictions) by at most 2h/Δ.
  - At h = 0.01: Δ = 0.02, 0.05, 0.10 gives 0.5, 0.2, 0.1 per line and 1.0, 0.4, 0.2 on D. Per-line noise stays below log √2 = 0.347 when Δ > 0.029.
  - This holds under assumption A1: every cell has |mean| > h, so the sign pattern is stable. Where A1 fails, the crossing can jump by log 2 (one grid interval) or by up to log 4 (from clamp to clamp).
  - Δ is a mean, so it cannot enter the formula. The preregistration states the power as this function of Δ.
- **Feasibility:** T(n) = 160/80 × n/40 × 65 × 30/20 min = 4.875n min. At n = 40 that is 195 min (3 h 15 min). At the cap of 100 it is 487.5 min (8 h 08 min), leaving 1 h 52 min (23%) inside the 10 h budget. Run 1's 1 h 40 min comes before it. The 300 test seeds allow n ≤ 300. This is derived, not measured. It assumes run time scales linearly with cells and episodes and inversely with workers, and that planner cost does not depend on kappa or delta.

**D. Exclusions (none drop data; every rule is on structure)**
- **Zero-width series:** kept. The scorer's `crossing()` skips means of exactly 0, or uses an exact zero as the crossing. A line whose means are all zero gets sign 0 and is placed "above". They are counted and reported.
- **Bitwise-identical policy pairs:** kept. Only a planned_once/greedy pair enters the contrast, and that case is the zero-width case above. Other identical pairs are counted only.
- **Pre-commit structural assertion:** `claim_a_episode_count.py` fails loudly if any dev line has all four contrast means exactly 0. The null would then emit `sign_at_max_sigma = 0`, which `load_predictions` refuses. If this trips, the preregistration goes back before any held-out data exist.
- **Censored crossings:** kept and scored by the scorer's clamp to [0.1, 0.4], with the counts in B4.
- **Multiplicity:** there is one primary scalar, so the verdict needs no correction. All 40 per-line errors, the direction checks and the 160 cell intervals are labelled unadjusted and descriptive, with no per-line inference.
- **Shared episodes:** every cell uses the same n test episodes. The bootstrap resamples episodes jointly, so the interval carries the dependence between cells. The paper says the effective sample is n episodes, not 40 independent lines.

**E. Which report feeds the null**
- The null comes from run 1's report, but it must be byte-identical to the null produced from the 26 November map.
- **Ordering problem:** the preregistration needs n before it is committed, and run 1 runs at that commit. So before the commit:
  - n is computed from `phase_20261126_184038_27744.json`;
  - the SHA-256 of `null_phase_predictions.py <that map> experiments/claim_a_heldout_settings.json --contrast planned_once_minus_greedy --sign-beyond-crossing <the predictions file's sign_beyond_crossing, read by script>` is computed and recorded. Only the hash is printed.
- **Bitwise condition after run 1:** both must hold.
  1. `tools/compare_phase_reports.py runs/phase/phase_20261126_184038_27744.json <run 1> --ignore-arg workers` prints "PASS: 480 results, 96000 values compared, 0 mismatched, 0 missing or malformed fields".
  2. The null built from run 1 and written to `runs/claim_a/` has the recorded SHA-256, and `claim_a_episode_count.py` on run 1 prints the same n.
- Either failing is a stop.

**F. Severity check, from the scorer's code**
- Pass the null file as the predictions as well as the null. In `score()`, every resample then computes `median_error` twice with identical inputs. The results are identical floats, so `diff` is exactly 0.0 and `_percentiles` returns [0.0, 0.0].
- Under my mapping, lo ≥ 0 gives **fail** with certainty, whatever the data, n or seed.
- A theory that omits 20 or more lines gives E = inf. E_null stays finite because the null covers all 40 lines, so the difference is inf and the verdict is fail, with no NaN.
- E against log √2 does not have this property. The null's verdict there depends on the held-out data, and it passes whenever doubling a rate barely moves the crossings or most lines are censored on the same side.
- So the primary test measures what the theory adds beyond the dev map, which is the only thing "predicts" can mean here.

**G. Risks and what a hostile reviewer would say**
1. **"h* = 0.01 is arbitrary."** Agreed: it is a judgement, fixed before any spread was read. There is no mean-free way to turn a half-width into a log-ratio error, and C states the dependence on Δ openly. This is not fatal.
2. **"If the crossings do not move, the null cannot be beaten, so the test is loaded towards fail."** That is intended: the settings were chosen as new lines the null cannot predict, and the fail sentence says only "not better than the null", never "the theory is wrong".
3. **"The cap makes n follow the clock."** The cap is part of the rule and fixed before the spreads are read. If it binds, the achieved resolution is reported.
4. **"The 2h/Δ bound is loose."** It is a worst case. The bootstrap interval is the operative measure.
5. **"The null's dev noise is not in the interval."** True. The null is a frozen competitor, and its noise is disclosed.

What would show this design wrong:
- B5's achieved half-width exceeds h*;
- many cells have |mean| < h, breaking A1;
- the null-as-theory run gives anything other than [0, 0].

Prior work: comparing against a persistence or naive forecast is the forecast skill score (Murphy 1988, Monthly Weather Review 116:2417) and MASE (Hyndman and Koehler 2006, IJF 22:679). The paired bootstrap of a difference on shared resamples follows Efron and Tibshirani (1993). The "can it fail" criterion is Mayo's severity requirement. These citations are from memory; I ran no search.

Files cited: `experiments/score_phase_predictions.py`, `experiments/null_phase_predictions.py`, `experiments/stats.py`, `seeds.yaml`.
