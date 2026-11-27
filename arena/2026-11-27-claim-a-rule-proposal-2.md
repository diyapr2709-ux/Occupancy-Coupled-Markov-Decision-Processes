# Proposal 2

## The primary test is E − E_null, the null is fed by run 1, and the episode count is capped at 100

**Firewall disclosure, read this first.** I opened nothing under predictions/ and no runs/phase file. However, one Grep across arena/ printed two lines I did not ask for:
- `arena/2026-11-24-operations-proposal-2.md:215` holds contrast means ± half-widths at sigma 0.1, 0.2 and 0.4 from an older table.
- `arena/2026-11-24-design-proposal-2.md:101` holds a sigma 0.4 paired SD from an older map.

I did not use either of them. The proposal's constants come only from the committed margin m = 0.01 (`arena/2026-11-24-design-outcome.md:30`), the 10-hour budget and the scorer's code. The judge should still weigh this exposure.

### A. Primary decision rule
```yaml
primary_quantity: D = E - E_null   # scorer key E_minus_E_null_ci95, same resamples as E
scorer: experiments/score_phase_predictions.py <theory.json> <report_kappa0.2> <report_delta0.1> --null <null.json> --resamples 10000 --seed 0
margin_on_D: 0
verdict:
  pass: upper 97.5% bound of D < 0
  fail: lower 2.5% bound of D >= 0
  inconclusive: lower < 0 <= upper
sentences:
  pass: "On 40 held-out lines the blind theory placed the sign change of planned_once_minus_greedy closer than the theory-free null: E - E_null 95% bootstrap interval [a, b], entirely below 0."
  fail: "The blind theory did not place the held-out boundary better than carrying the dev map's crossings over: E - E_null 95% interval [a, b], lower bound at or above 0."
  inconclusive: "The test was inconclusive at the preregistered power: E - E_null 95% interval [a, b] contains 0."
```
Why D is primary: the null's own docstring sets the bar ("A theory earns the word 'predicts' only by placing crossings better than this"), and only D is severe by construction (see F).

Two pure additions to the scorer, neither of which changes E or its interval:
- a key `verdict_vs_null` that applies the mapping above to the existing interval, added after the bootstrap so the random-number stream is untouched;
- a `parser.error` when the null file's `contrast` or `sign_beyond_crossing` differs from the theory file's. Today the scorer reads neither field from the null.

### B. Secondary quantities, which are reported and carry no verdict
- **E against log √2.** Report E, its 95% interval and the scorer's own `verdict` label, worded only as: "median error E = x, 95% interval [lo, hi], against the preregistered margin log √2". It adds no claim sentence.
- **E_null.** Its point value.
- **Per line.** The measured, theory and null crossings and both errors. The intervals are marginal and labelled "not simultaneous".
- **Direction checks.** Each with its interval and `wrong_sign`, unadjusted and descriptive. Entries with `censored: true` are listed, not interpreted.
- **Censored lines.** Counts of lines censored low or high, separately for measured, theory and null crossings.

### C. Episode count
Proposed script `experiments/claim_a_episode_count.py <dev_report>`:
- SD* = the maximum over the 80 dev cells of the sample SD (ddof 1) of the per-episode `planned_once_minus_greedy` contrast. Zero-SD cells cannot raise it.
- tau = m/2 = 0.005 in expected failure rate. A cell contrast of size m then has its sign resolved with probability about 0.975 under a normal approximation.
- n = min(100, max(40, smallest n with `stats.t_quantile(0.975, n-1)` × SD* / √n ≤ tau)). The script prints n and nothing else.

What it implies:
- **Half-width per cell** is c(n) × SD, with c(n) = t₀.₉₇₅,ₙ₋₁/√n. This gives 0.3198 at n = 40 (the design constant) and 0.1984 at n = 100. If n < 100, the half-width is at most tau by construction.
- **Assumptions.** A1: every held-out cell's paired SD is at most SD*. A2: no grid cell other than the two ends of the crossing bracket has a mean within one half-width of 0.
- **Per line.** By the delta method with correlation at most 1 (valid under shared episodes), and (s₁ − s₀)/σ̂ ≤ 1 for the grid's resolvable brackets: the log σ̂ half-width is at most hw/Δ. Δ is the change of the contrast across the bracket.
- **On E.** E is 1-Lipschitz in sup-norm over the lines' log σ̂, so its 95% resolution is at most 1.66 × tau/Δ_min = 0.0083/Δ_min. The factor 1.66 is a Bonferroni t over 40 lines at n = 100.
- **On D.** At most 0.0166/Δ_min. D therefore resolves log √2 when Δ_min ≥ 0.048, and E does when Δ_min ≥ 0.024.

Feasibility, derived and not measured:
- T(n) = 160/80 × n/40 × 65 min × 30/20 = 4.875 n minutes.
- n = 40 takes 3 h 15 min; n = 100 takes 8 h 08 min, leaving 112 min (23%) inside the 10-hour budget.
- Stop rule: if the guard admits W workers with 4.875 n × 20/W > 600 (W < 17 at n = 100), stop and go to the owner; do not change n.
- The derivation assumes time is linear in cells and episodes and scales with 20/W, and that the held-out kappa and delta do not slow the planner.

### D. Exclusions (none data-dependent)
- **Zero-width series and bitwise-identical pairs: kept.** In the scored contrast, identical planned-once and greedy episodes give a mean of exactly 0. `crossing()` already handles that structurally: it skips zero means and places an exact zero inside a bracket. Excluding these would change which lines count, which could be gamed. Report their held-out counts. Identical pairs outside the scored contrast do not enter it.
- **Censored crossings: kept.** The scorer clamps them to [0.1, 0.4], and two crossings censored on the same side agree. A line with no nonzero mean (sign 0) is placed high. Counts are reported.
- **Lines left out of the theory file** have infinite error, as the scorer defines. An infinite D gives fail.
- **Multiplicity.** The primary is one statistic over all 40 lines and makes no per-line claim, so no correction applies to it. Per-line, per-cell and direction intervals are labelled "95% marginal, unadjusted, 40 lines / 160 cells".
- **Shared episodes.** All 160 cells use the same n test episodes. The bootstrap resamples episodes jointly, so cross-line dependence is kept. The paper states that the replication unit is n episodes, not 40 independent lines.
- **Defects.** A missing cell or a non-finite value means the run is not scored and is reported as not completed.

### E. Null source
- The null is fed by run 1, the dev-map rerun at the preregistration's commit.
- **Bitwise condition:** `tools/compare_phase_reports.py runs/phase/phase_20261126_184038_27744.json <run1> --ignore-arg workers` prints PASS with 480 results, 96 000 values compared, 0 mismatched, 0 missing or malformed fields, and no setup FAIL. Anything else is a stop.
- **Before the preregistration is frozen:** a script generates the null from the 18:40 map (SHA-256 `0389151c…`) and prints only its SHA-256, which the preregistration records. `--sign-beyond-crossing` is taken from the theory file's field by script.
- **After run 1:** the null regenerated from run 1 must have the same SHA-256. This pins the null before any held-out data exist and checks it at the test's code state.

### F. Severity
If the null is scored as the theory, `score()` computes `median_error(...)` twice on identical inputs for every resample. So `boot_e[i] == boot_null[i]`, every difference is 0.0 and `E_minus_E_null_ci95` is [0, 0]. E_null is always finite: the null covers all 40 lines and `place()` returns values in [0.1, 0.4].

Under this rule the upper bound 0 is not below 0, and the lower bound 0 is at or above 0, so the null as theory is a fail for any data. A theory that departs from the null only away from the median line also gets a D distribution concentrated at 0 (fail or inconclusive).

By contrast, the E-against-log √2 rule passes the null whenever the held-out crossings barely move. That is why it is secondary: a pass there would not show that the theory predicts anything.

### G. Risks and hostile review
1. **Strongest objection: the null is noisy.** It carries crossings measured on 40 dev episodes. A theory that recovered only the noise-free dev crossings could beat it, so "better than the null" partly rewards removing dev noise. This is not fatal: the theory was blind to the dev map, and the null's own noise contribution is bounded per line by 0.3198 × SD*/Δ. That bound is reported next to D.
2. **A zero margin on D** lets a trivially small improvement pass. The interval is always stated in the sentence.
3. **Ties count as fail.** The median is coarse, so a theory that ties the null on the median line fails. A reviewer may call the rule built to fail. The answer is the null's own docstring and the severity criterion.
4. **The cap may bind.** If SD* > 0.0252, then n = 100 is set by the budget, not by tau, and the half-width is 0.1984 × SD. The report states this.
5. **A1 may fail**, since doubling kappa or delta can raise the spread. The report gives the fraction of held-out cells with SD > SD* (a spread, not a mean).
6. **Percentile bootstrap of a median** of non-smooth functionals can under-cover.
7. **Wrong theory-file shape.** If the theory file holds base-setting lines, `load_predictions` raises an error. Before launch, a script checking only keys should print PASS or FAIL.
8. **tau is borrowed** from the learned-agent margin, not chosen for Claim A.

**What would show this wrong:**
- held-out SD > SD* in many cells;
- measured Δ below 0.048 on most crossing lines, which means D could not resolve log √2;
- D's mass at exactly 0 above 50%, which means the median is too coarse for this design.

**Prior work.** Scoring against a persistence forecast is standard skill-score practice in forecast verification (Murphy 1988, Monthly Weather Review 116:2417). I did not search for prior use of this rule on phase boundaries.

### Files
- `experiments/score_phase_predictions.py`, `experiments/null_phase_predictions.py`, `experiments/stats.py`, `arena/2026-11-24-design-outcome.md`.
