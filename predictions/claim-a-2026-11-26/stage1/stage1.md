# Stage 1: reduced model, mechanism of the sign change, qualitative predictions

Contrast: `planned_once_minus_greedy` = F(planned-once) - F(greedy), expected failure rate per
decision, toy environment, class sizes [2, 2], horizon H = 40 quarters.

Every number below is a noise-free (sigma = 0) deterministic quantity or a closed-form moment
recursion of a reduced model I wrote (`derivation/reduced.py`, `derivation/closure.py`). None
comes from a measurement of the environment, and no stochastic simulation was run.

## 0. Assumptions (stated before anything else)

- A1. `cross_resistance([2, 2])` (defined in `train_agent.py`, which is not in the export) is
  the `ToyConfig.paired` layout with rho = 0: four drugs, rows (1,0),(1,0),(0,1),(0,1). So
  there are two interchangeable classes, and a drug of class j fails with probability r_j. If
  rho > 0 the analysis keeps its form, but the leverage of every policy shrinks.
- A2. `ENV_KWARGS` (also not exported) leave contraindications off and the default
  decisions_per_quarter. They turn on a small start-state jitter (`r0_spread`, `mu0_spread`),
  because the sweep says "starting state jittered as in training". The magnitudes are unknown.
  I treat the start as r0 = 0.05 +/- eps with eps of order 0.01, and m0 near 1/2.
- A3. `expected_failure_rate(log)` is the episode mean of P(fail | drug given), that is the mean
  of `fail_prob[action]`, not of sampled failures. If it is sampled failures instead, the
  expectation is the same and only the variance grows.
- A4. The planner (Adam polishing from the best class rotation, plus cycling and random starts)
  reaches the noise-free optimum of the mean-field model. In my reduced model, an exact-gradient
  projected optimiser started from the best rotation, from uniform and from the greedy schedule
  almost never improves on the best rotation (the largest gain was 5e-4). So "planned-once" is,
  to good approximation, the best bang-bang class rotation for the episode's start state.

## 1. Reduced model

Because of A1, the four-drug system collapses exactly onto two classes. The state is
r = (r1, r2), the class-1 occupancy m (class 2 has 1 - m), and the queue of the last `delay`
class-1 shares. Quarter q runs in this order, copied from `MeanFieldModel.simulate` and
`ToyOCMDPEnv._close_quarter`:

    cost_q   = p_q r1 + (1 - p_q) r2                       (r at the start of quarter q)
    m        <- lam m + (1 - lam) p_{q-delay}               (only once q >= delay; m = m0 before)
    r_j      <- r_j (1 - delta - kappa pi_j) + kappa pi_j,  pi_1 = m, pi_2 = 1 - m
    r_j      <- r_j + fold( sigma sqrt(r_j (1 - r_j)) Z_j,  min(r_j, 1 - r_j) )

Two consequences matter.

1. Prescribing in the last `delay` quarters never reaches r within the horizon. Prescribing in
   quarter q reaches r with weight (1 - lam) lam^k at quarter q + delay + k + 1.
2. For a fixed schedule p (any p that does not look at r), the r-update is affine in r with
   deterministic coefficients. `fold` is an odd map of a symmetric shock, so E[shock | r] = 0.
   Hence E[r_q] follows the noise-free path exactly, and because the cost is linear in r:

       E[F_planned_once](sigma) = F_planned_once(0)   for every sigma.

   This is exact, not approximate, for a bang-bang plan. For quarters with an interior share
   the only correction comes from sampling 25 patients per quarter, and it is O(1e-4).

**Planned-once** is the noise-free optimum computed once from the true start state. By (2), noise
never changes its expected value. All of the sigma-dependence of the contrast therefore sits in
greedy.

**Greedy** gives p_q = 1[r1 < r2]: always the class with the lower current resistance. Ties
happen only at an exactly symmetric start, and the jitter in A2 removes them.

## 2. Noise-free gap G0

At sigma = 0 both policies are deterministic and the planner minimises the same objective over
all schedules, so

    C(0) = F_plan - F_greedy(0) = -G0,  G0 >= 0.

The reduced model gives the following (start r0 = (0.04, 0.06), m0 = 0.5, H = 40). Each cell
shows G0 for kappa/delta = 0.1/0.05 | 0.2/0.05 | 0.1/0.1:

| delay \ lam | 0 | 0.5 | 0.8 | 0.95 |
|---|---|---|---|---|
| 0 | 0 / 0 / 0 | .011/.012/.009 | .028/.031/.022 | .022/.020/.016 |
| 1 | .006/.008/.004 | .015/.019/.011 | .027/.028/.022 | .019/.019/.015 |
| 2 | .008/.015/.004 | .016/.019/.018 | .026/.026/.023 | .018/.018/.014 |
| 4 | .014/.016/.010 | .016/.019/.016 | .025/.033/.017 | .022/.018/.011 |
| 8 | .020/.021/.006 | .022/.029/.016 | .009/.014/.014 | .023/.022/.015 |

- At delay 0, lam 0, greedy alternates every quarter. That is the period-1 rotation, which is
  the optimal plan there, so G0 = 0.
- Whenever pressure lags prescribing (delay > 0 or lam > 0), greedy switches only after the
  failure curves have crossed. The pressure already queued or remembered on the old class keeps
  raising its resistance: an overshoot. The plan anticipates this and also exploits the last
  `delay` "free" quarters, so G0 > 0, about 0.005 to 0.033.

## 3. The mechanism that changes the sign

Greedy acts on the realised r. In the symmetric two-class model, write s = r1 + r2,
x = r1 - r2, q = p1 - p2 and m~ = m1 - m2. Then the per-quarter cost is s/2 + q x / 2. Summing
the two r-updates (pressures sum to 1) gives, in any stationary regime,

    delta E[s] = kappa (1 - E[s]/2 - E[m~ x]/2)

and therefore, for any policy,

    c = F_mix - (kappa / (2 delta + kappa)) E[m~ x] / 2 + E[q x] / 2,   F_mix = kappa / (kappa + 2 delta).

For greedy, q = -sign(x), so E[q x] = -E|x|: it harvests the whole spread between the two
classes' resistance. With no lag (m~ = q) a fraction kappa/(2 delta + kappa) of that harvest is
given back, because the lower-resistance class has the larger (1 - r) and acquires resistance
faster. That leaves

    c_greedy = F_mix - E|x| * delta / (2 delta + kappa).

Noise is mean-zero but widens |x|. A committed plan cannot use the widening: by (2) its expected
cost is flat in sigma. A feedback rule can. So

    C(sigma) = -G0 + g(sigma),   g(sigma) = F_greedy(0) - F_greedy(sigma),

and the sign changes at sigma* with g(sigma*) = G0. Beyond it, greedy fails less than the
committed plan:

    **sign_beyond_crossing = +1** (planned-once minus greedy becomes positive).

The size of g(sigma):

- Where greedy chatters (small lag), the bang-bang drift -kappa(1 - r) sign(x) holds x down.
  A random walk with constant restoring drift c = kappa(1 - r) and shock variance
  v = 2 sigma^2 r(1 - r) has E|x| = v/(2c) = sigma^2 r / kappa. So g grows like sigma^2 at
  small sigma.
- Where the lag stops greedy from controlling x, the spread is that of an Ornstein-Uhlenbeck
  process, and g grows like sigma.
- At sigma ~ 0.4 the spread saturates against the [0, 1] walls, where the reflection fold cuts
  the shock variance.
- With a long delay and little smoothing, small noise can hurt greedy (g < 0). A noise-triggered
  switch is paid for `delay` quarters later, by which time the fluctuation that triggered it has
  decorrelated. g is then non-monotone, and sigma* is pushed up.

A Gaussian moment closure of the closed loop (`closure.py`, described fully in stage 2)
quantifies this. g(0.4) exceeds every G0 in the table above, in all three settings. So the
contrast is predicted to end positive on every line: sign_at_max_sigma = +1 everywhere.

## 4. Qualitative predictions for sigma*

The scorer places every crossing in [0.1, 0.4]. "Crosses below the grid" means sigma* <= 0.1.

1. **Delay 0, lam 0: sigma* <= 0.1 (crossed below the grid).** G0 = 0, and any noise helps
   greedy.
2. **lam = 0.95: sigma* <= 0.1 at every delay.** With heavy smoothing, one quarter of
   prescribing moves the pressure by only 5%. Greedy's harvest of the noise therefore costs
   almost nothing in later pressure (E[m~ x] barely responds to noise-driven switches), so
   g(0.1) is about 0.03 to 0.04 against G0 of about 0.02.
3. **sigma* increases with delay at lam <= 0.8.** G0 grows with delay, and the small-noise
   harvest shrinks and turns negative at delay >= 4. At the base setting I expect delay 0 to 1
   to give <= 0.1, delay 2 to give about 0.1, delay 4 about 0.13, and delay 8 (lam <= 0.5)
   about 0.15 to 0.17.
4. **In lam, sigma* is non-monotone.** At delay 0 it rises from lam 0 to lam 0.8 as G0 grows,
   then falls at 0.95. At delay >= 4 it is lowest at lam 0.95, and lam 0.8 lies below lam 0 to
   0.5.
5. **Doubling kappa (0.2, 0.05) raises sigma*,** by about 10 to 60% on the lines that cross
   inside the grid, most at long delays. Some delay-4 and delay-8 lines move into
   (0.2, 0.4). There are three reasons:
   - stronger control authority kappa(1 - r) shrinks the noise-driven spread, since
     E|x| ~ sigma^2 r / kappa;
   - the give-back fraction kappa/(2 delta + kappa) rises from 1/2 to 2/3;
   - r sits higher (the mixing level goes from 1/2 to 2/3), where sqrt(r(1 - r)) is no larger.

   G0 is roughly unchanged.
6. **Doubling delta (0.1, 0.1) leaves sigma* roughly unchanged,** within about +/-15% with no
   reliable sign. Three effects roughly cancel:
   - the give-back fraction falls from 1/2 to 1/3, which helps greedy;
   - faster reversion shortens the memory of each shock, and r sits lower (1/2 to 1/3), which
     narrows the spread;
   - G0 falls by about 25%.
7. **Large sigma: the sign is +1 on all 60 lines.** No line is predicted to stay negative
   through sigma = 0.4.

## 5. Where theory is weakest

- **G0 depends on the unknown start-state jitter (A2).** Stage 2 averages over several assumed
  jitter levels and reports the sensitivity.
- **The Gaussian closure may misjudge g at sigma = 0.4,** where r is far from Gaussian and the
  fold is strong. It is better at 0.1 to 0.2, which is where most crossings are predicted. If
  the closure overstates the harvest by a factor of 2, the delay >= 4 crossings move up by
  about one grid interval.
- **Lines whose noise-free regret is near zero can be dominated by planner residuals.** This
  covers (0, 0) and the delay-1 to 2, lam-0 lines at the base and delta settings. If the code's
  planner falls short of my optimum, these lines move to the "below grid" side, which is
  already the prediction.
