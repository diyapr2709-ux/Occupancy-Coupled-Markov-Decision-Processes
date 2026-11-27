# Stage 2: numeric predictions of sigma* for planned_once_minus_greedy

Deliverable: `derivation/predictions.json`. It has:

- `"contrast": "planned_once_minus_greedy"` and `"sign_beyond_crossing": 1`;
- 60 lines: 3 settings x delays {0, 1, 2, 4, 8} x lams {0, 0.5, 0.8, 0.95}, horizon 40;
- one optional direction: kappa 0.1 -> 0.2 at delta 0.05, with sign +1.

No numbers were measured. Everything below comes from closed forms, exact noise-free (sigma = 0)
rollouts and optimisations, and a Gaussian **moment closure**. All of these are of my own reduced
model, and none uses a random draw.

Scripts (numpy only, self-contained, run with `python -I`):

- `stage2_compute.py` computes everything, writes `stage2_results.json`, and logs to
  `stage2_log.txt`;
- `stage2_predict.py` applies the scorer's crossing rule and writes `predictions.json`;
- `diag.py` reports closure diagnostics.

## 1. What changed from stage 1

The coordinator supplied the definitions I could not see. These are definitions only, with no
measured values:

- `ENV_KWARGS = dict(expected_reward=True, markov_observation=True, r0_spread=0.04, mu0_spread=0.5)`.
- Rows of `cross_resistance([2, 2])` are one-hot by class: (1,0), (1,0), (0,1), (0,1). This is
  assumption A1 of stage 1, now confirmed.
- The metric is the mean P(fail) over the drugs actually given. This is A3, now confirmed.

The jitter is now applied exactly as `ToyOCMDPEnv.reset` applies it:

- `r0_j = clip(0.05 + U(-0.04, 0.04), 0, 1)`, independently per mechanism. The clip is never
  reached, since r0_j lies in [0.01, 0.09].
- `mu0 = 0.5 * uniform(4) + 0.5 * Dirichlet(1,1,1,1)`. The class-1 occupancy is therefore
  `m0 = 0.25 + 0.5 b`, where b, the sum of two Dirichlet(1,1,1,1) components, is Beta(2, 2).

The expectation over the jitter uses a deterministic equal-weight midpoint rule over 100 start
states:

- u1 at the 4 bin midpoints {+/-0.01, +/-0.03};
- u2 at the 5 bin midpoints {0, +/-0.016, +/-0.032};
- b at the 5 Beta(2, 2) quantile midpoints (probabilities 0.1, 0.3, 0.5, 0.7, 0.9).

The u1 and u2 nodes are disjoint, so no node is an exact tie r1 = r2. The real environment would
break such a tie by sampling patients; a midpoint rule cannot.

`expected_reward` and `markov_observation` change neither policy nor the metric. Greedy reads
`info["fail_prob"]`, and the oracle reads `env.state()`.

## 2. How each number was obtained (per setting and line)

1. **F_plan(start).** This is the noise-free optimum of the reduced model, which reproduces
   `MeanFieldModel.simulate` step for step. It is the minimum of:
   - the best class rotation (all orders, periods 1 to 40, all phases, the same bank as
     `rotation_schedules`);
   - projected Adam on the exact adjoint gradient, 400 steps, started from that rotation;
   - the same Adam polish started from the greedy schedule.

   Over all 18,000 (line, start) pairs, polishing improved on the best rotation by at most
   7.7e-4. The code's planner is at least as good as the same rotation bank, so its noise-free
   regret differs from mine by under 1e-3.
2. **E[F_planned_once](sigma) = E[F_plan] at every sigma.** This is exact, not an
   approximation. Under a committed schedule, r follows an affine recursion with deterministic
   coefficients. The reflected shock is odd, so its conditional mean is zero. The cost is linear
   in r.
3. **F_greedy(0).** An exact noise-free greedy rollout per start, with the same tie rule
   (`np.isclose`).
4. **F_greedy(sigma) for sigma > 0: Gaussian moment closure** (`greedy_closure`). The state is
   z = (r1, r2, m, pending shares), carried as a Gaussian mean and covariance. Each quarter:
   - **Greedy share.** The share p = 1[x < 0], with x = r1 - r2, is replaced by its statistically
     linearised Gaussian. Its mean Phi(-mu_x/s_x) and variance p(1-p) are exact. Its
     cross-covariances use Stein's lemma, Cov(p, z) = -phi(mu_x/s_x)/s_x * Cov(x, z).
   - **Cost.** The quarter's cost E[r2 + x 1[x<0]] = mu_r2 + mu_x Phi(-mu_x/s_x) - s_x phi(mu_x/s_x)
     is exact for Gaussian x.
   - **Occupancy update.** The update, with its delay queue, is linear and so exact.
   - **Selection step.** The step is bilinear in (m, r). Its mean and covariance are exact for a
     Gaussian input (Isserlis). `check_closure_quadratic` checks the hand-coded corrections
     against 2 tr(Q_i S Q_j S) and agrees to 7e-18.
   - **Shock.** The shock adds variance sigma^2 E[r(1-r)] at the post-step moments, as
     `_close_quarter` does. That variance is multiplied by the variance ratio of the reflection
     fold, evaluated at the mean r. The ratio is the exact folded-normal variance, tabulated by
     quadrature of the normal density, not by sampling.

   Each start begins with zero covariance, and the results are averaged over the 100 starts.
5. **Contrast.** C(sigma) = E F_plan - E F_greedy(sigma) on the scorer's grid {0, 0.1, 0.2, 0.4}.
   C(0) = -G0 is exact.
6. **sigma_star.** This is the scorer's own `crossing` rule (code copied) applied to the predicted
   grid means with beyond = +1. It is the crossing the scorer would compute if the measured means
   equalled my predicted expectations: the linear interpolation between grid points, not the
   continuous root. The continuous root, computed from a finer grid {0, .05, .1, .15, .2, .3, .4},
   is within 0.01 of the grid interpolation on every line that crosses above 0.1. On lines that
   cross below 0.1 the two differ by up to 0.015, but the scorer clamps both to 0.1, so the choice
   does not matter. Values
   below 0.1 are kept as numbers; the scorer clamps them to 0.1. A line whose every nonzero grid
   mean is positive gets `null` with sign +1, which the scorer also places at 0.1.
   `sign_at_max_sigma` is the sign of C(0.4).

## 3. Results

Column key:

- **G0** = E F_greedy(0) - E F_plan, which is exact.
- **C(.)** comes from the closure.
- **pred** is the scorer-rule crossing; **placed** is that crossing clamped to [0.1, 0.4].
- **range** is the placed crossing when the closure's noise gain g(sigma) = F_greedy(0) - F_greedy(sigma)
  is scaled by 0.6, 0.75, 1.25 and 1.5, in that order.
- **conf**:
  - H: placed value stays within a factor of 1.2 across the 0.6 to 1.5 scaling.
  - M: it stays within a factor of sqrt 2.
  - L: it moves further than that, or the crossing is above 0.2, where it rests on the closure
    at sigma = 0.4.

`sign_at_max_sigma = +1` on every line. C(0.4) is at least +0.029 everywhere, and it stays positive
even if g is overstated by 40%.

### kappa 0.1, delta 0.05 (base)

| delay | lam | G0 | C(0.1) | C(0.2) | C(0.4) | pred | placed | range x0.6/0.75/1.25/1.5 | conf |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | .0000 | +.0107 | +.0340 | +.0742 | null | 0.100 | .100/.100/.100/.100 | H |
| 0 | .5 | .0120 | +.0082 | +.0357 | +.0787 | 0.060 | 0.100 | .100/.100/.100/.100 | H |
| 0 | .8 | .0273 | +.0037 | +.0377 | +.0864 | 0.088 | 0.100 | .143/.116/.100/.100 | L |
| 0 | .95 | .0254 | +.0163 | +.0585 | +.1150 | 0.061 | 0.100 | .101/.100/.100/.100 | H |
| 1 | 0 | .0068 | +.0065 | +.0318 | +.0739 | 0.051 | 0.100 | .100/.100/.100/.100 | H |
| 1 | .5 | .0159 | +.0019 | +.0306 | +.0754 | 0.089 | 0.100 | .130/.112/.100/.100 | M |
| 1 | .8 | .0270 | -.0005 | +.0344 | +.0842 | 0.101 | 0.101 | .153/.127/.100/.100 | L |
| 1 | .95 | .0227 | +.0153 | +.0578 | +.1147 | 0.060 | 0.100 | .100/.100/.100/.100 | H |
| 2 | 0 | .0109 | +.0010 | +.0274 | +.0711 | 0.092 | 0.100 | .124/.110/.100/.100 | M |
| 2 | .5 | .0163 | -.0025 | +.0268 | +.0727 | 0.109 | 0.109 | .146/.127/.100/.100 | M |
| 2 | .8 | .0253 | -.0034 | +.0318 | +.0824 | 0.110 | 0.110 | .158/.134/.100/.100 | L |
| 2 | .95 | .0209 | +.0143 | +.0570 | +.1141 | 0.059 | 0.100 | .100/.100/.100/.100 | H |
| 4 | 0 | .0152 | -.0071 | +.0205 | +.0664 | 0.126 | 0.126 | .162/.144/.115/.107 | M |
| 4 | .5 | .0176 | -.0087 | +.0208 | +.0684 | 0.129 | 0.129 | .169/.149/.117/.110 | M |
| 4 | .8 | .0241 | -.0101 | +.0249 | +.0764 | 0.129 | 0.129 | .175/.152/.115/.106 | M |
| 4 | .95 | .0185 | +.0119 | +.0546 | +.1121 | 0.061 | 0.100 | .101/.100/.100/.100 | H |
| 8 | 0 | .0221 | -.0170 | +.0113 | +.0593 | 0.160 | 0.160 | .214/.186/.144/.134 | M |
| 8 | .5 | .0195 | -.0154 | +.0137 | +.0626 | 0.153 | 0.153 | .197/.175/.139/.131 | M |
| 8 | .8 | .0148 | -.0080 | +.0262 | +.0782 | 0.123 | 0.123 | .152/.138/.115/.109 | M |
| 8 | .95 | .0150 | +.0080 | +.0503 | +.1080 | 0.065 | 0.100 | .105/.100/.100/.100 | H |

### kappa 0.2, delta 0.05 (held out)

| delay | lam | G0 | C(0.1) | C(0.2) | C(0.4) | pred | placed | range x0.6/0.75/1.25/1.5 | conf |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | .0000 | +.0054 | +.0184 | +.0379 | null | 0.100 | .100/.100/.100/.100 | H |
| 0 | .5 | .0135 | +.0008 | +.0200 | +.0455 | 0.095 | 0.100 | .143/.119/.100/.100 | L |
| 0 | .8 | .0314 | -.0052 | +.0227 | +.0590 | 0.119 | 0.119 | .194/.156/.100/.100 | L |
| 0 | .95 | .0254 | +.0109 | +.0481 | +.0978 | 0.070 | 0.100 | .116/.100/.100/.100 | H |
| 1 | 0 | .0098 | -.0018 | +.0145 | +.0386 | 0.111 | 0.111 | .151/.131/.100/.100 | M |
| 1 | .5 | .0206 | -.0091 | +.0123 | +.0421 | 0.142 | 0.142 | .210/.175/.123/.110 | L |
| 1 | .8 | .0334 | -.0120 | +.0175 | +.0569 | 0.141 | 0.141 | .225/.179/.118/.103 | L |
| 1 | .95 | .0230 | +.0090 | +.0469 | +.0978 | 0.072 | 0.100 | .117/.100/.100/.100 | H |
| 2 | 0 | .0168 | -.0097 | +.0083 | +.0363 | 0.154 | 0.154 | .221/.185/.135/.123 | L |
| 2 | .5 | .0217 | -.0149 | +.0071 | +.0401 | 0.168 | 0.168 | .245/.201/.148/.135 | L |
| 2 | .8 | .0300 | -.0152 | +.0147 | +.0566 | 0.151 | 0.151 | .225/.184/.131/.117 | L |
| 2 | .95 | .0216 | +.0079 | +.0460 | +.0980 | 0.073 | 0.100 | .117/.100/.100/.100 | H |
| 4 | 0 | .0205 | -.0183 | +.0006 | +.0344 | 0.197 | 0.197 | .277/.237/.175/.161 | M |
| 4 | .5 | .0211 | -.0216 | -.0002 | +.0374 | 0.201 | 0.201 | .276/.238/.181/.168 | L |
| 4 | .8 | .0285 | -.0206 | +.0083 | +.0533 | 0.171 | 0.171 | .247/.205/.152/.139 | L |
| 4 | .95 | .0175 | +.0061 | +.0442 | +.0975 | 0.074 | 0.100 | .115/.100/.100/.100 | H |
| 8 | 0 | .0257 | -.0293 | -.0110 | +.0292 | 0.255 | 0.255 | .340/.297/.229/.212 | L |
| 8 | .5 | .0278 | -.0277 | -.0085 | +.0335 | 0.240 | 0.240 | .329/.285/.214/.196 | L |
| 8 | .8 | .0148 | -.0175 | +.0079 | +.0554 | 0.169 | 0.169 | .208/.188/.157/.149 | M |
| 8 | .95 | .0157 | +.0018 | +.0383 | +.0926 | 0.090 | 0.100 | .124/.109/.100/.100 | M |

### kappa 0.1, delta 0.1 (held out)

| delay | lam | G0 | C(0.1) | C(0.2) | C(0.4) | pred | placed | range x0.6/0.75/1.25/1.5 | conf |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | .0000 | +.0086 | +.0294 | +.0656 | null | 0.100 | .100/.100/.100/.100 | H |
| 0 | .5 | .0110 | +.0073 | +.0329 | +.0711 | 0.060 | 0.100 | .100/.100/.100/.100 | H |
| 0 | .8 | .0212 | +.0066 | +.0379 | +.0793 | 0.076 | 0.100 | .124/.101/.100/.100 | M |
| 0 | .95 | .0178 | +.0182 | +.0550 | +.1002 | 0.049 | 0.100 | .100/.100/.100/.100 | H |
| 1 | 0 | .0055 | +.0052 | +.0276 | +.0644 | 0.052 | 0.100 | .100/.100/.100/.100 | H |
| 1 | .5 | .0129 | +.0016 | +.0278 | +.0664 | 0.089 | 0.100 | .127/.110/.100/.100 | M |
| 1 | .8 | .0204 | +.0022 | +.0339 | +.0755 | 0.090 | 0.100 | .136/.114/.100/.100 | M |
| 1 | .95 | .0168 | +.0160 | +.0530 | +.0981 | 0.051 | 0.100 | .100/.100/.100/.100 | H |
| 2 | 0 | .0092 | +.0010 | +.0240 | +.0610 | 0.091 | 0.100 | .122/.109/.100/.100 | M |
| 2 | .5 | .0142 | -.0024 | +.0235 | +.0620 | 0.109 | 0.109 | .146/.127/.100/.100 | M |
| 2 | .8 | .0196 | -.0021 | +.0293 | +.0708 | 0.107 | 0.107 | .148/.128/.100/.100 | M |
| 2 | .95 | .0153 | +.0144 | +.0514 | +.0964 | 0.052 | 0.100 | .100/.100/.100/.100 | H |
| 4 | 0 | .0114 | -.0065 | +.0164 | +.0532 | 0.129 | 0.129 | .162/.145/.119/.112 | M |
| 4 | .5 | .0155 | -.0077 | +.0170 | +.0547 | 0.131 | 0.131 | .173/.152/.119/.110 | M |
| 4 | .8 | .0142 | -.0050 | +.0251 | +.0660 | 0.116 | 0.116 | .148/.132/.107/.101 | M |
| 4 | .95 | .0122 | +.0123 | +.0491 | +.0940 | 0.050 | 0.100 | .100/.100/.100/.100 | H |
| 8 | 0 | .0108 | -.0127 | +.0090 | +.0448 | 0.159 | 0.159 | .192/.175/.149/.142 | M |
| 8 | .5 | .0159 | -.0157 | +.0067 | +.0431 | 0.170 | 0.170 | .221/.194/.156/.147 | M |
| 8 | .8 | .0087 | -.0086 | +.0185 | +.0581 | 0.132 | 0.132 | .153/.142/.125/.121 | H |
| 8 | .95 | .0124 | +.0065 | +.0420 | +.0864 | 0.066 | 0.100 | .105/.100/.100/.100 | H |

Totals:

- **Placement at 0.1:** 11 lines at the base setting, 7 at kappa 0.2 and 12 at delta 0.1 sit
  at 0.1 (crossed below or at the bottom of the grid). Base line (1, 0.8) sits at 0.101.
- **Inside the grid:** 9, 13 and 8 lines respectively cross inside it, counting (1, 0.8).
- **Above 0.2:** three lines are predicted to cross above 0.2, all at kappa 0.2: (4, 0.5),
  (8, 0) and (8, 0.5).
- **No line is predicted to stay negative** at sigma = 0.4.

## 4. Moment-closure error

**Where it is exact or checked:**

- It is exact for any fixed schedule, since the dynamics are then affine and C(0) is exact.
- It is exact at the level of the quadratic step.
- In the chattering limit (delay 0, lam 0) it reproduces the exact stationary spread of a
  bang-bang random walk to leading order, E|x| = v / (2c) with c = kappa(1 - r) and
  v = 2 sigma^2 r(1 - r).

**Where it approximates:**

- **E1. Shape of x.** The real distribution of x is not Gaussian. The harvest
  E[x 1[x<0]] = -E|x|/2 + E[x]/2 differs between symmetric shapes of equal variance by up to
  about 10% (Laplace 0.71 sd, Gaussian 0.80 sd, uniform 0.87 sd).
- **E2. Linearising a sharp switch.** In the long-lag lines, each episode switches sharply at a
  random time. The closure instead represents a ramp of mixed shares. Its error is second order
  in the switch-time jitter, but it is unchecked.
- **E3. Boundaries.**

  | sigma | Gaussian mass of r outside [0, 1], mean over the horizon | worst quarter |
  |---|---|---|
  | <= 0.1 | < 0.5% | — |
  | 0.2 | 0.1 to 2.3% | 3.8% |
  | 0.4 | 3.6 to 9.1% | 11.5% |

  (`diag.py`, representative lines.) The fold correction is applied at the mean r, whereas
  the true r distribution at sigma = 0.4 is skewed against the walls.

**Expected error in the noise gain g:** about +/-20% at sigma = 0.1, +/-30% at 0.2 and +/-40%
at 0.4. I do not know the sign of the bias.

**Effect on the scored crossings** (the range column):

- With g scaled by 0.75 to 1.25, every placed crossing below 0.2 moves by at most a factor of
  1.3.
- The kappa-0.2, delay-8, lam-0 or 0.5 lines move between 0.21 and 0.30 under that scaling.
  They depend on C(0.4), where the closure is weakest.

**Measurement noise** is not part of the prediction. Lines with |C(0.1)| < 0.003 can be measured
on either side of 0.1 from 20 episodes. Both sides sit near 0.1, so the log error stays small.

## 5. Directions

- **kappa 0.1 -> 0.2 (sign +1, included).** All 19 lines where both predicted crossings are
  numeric show an increase, median log shift +0.33. The reasons are those given in stage 1:
  stronger control authority shrinks the spread, the give-back fraction kappa/(2 delta + kappa)
  rises from 1/2 to 2/3, and G0 grows slightly.
- **delta 0.05 -> 0.1 (not included).** Only 8 of 19 shifts are positive, with median -0.008.
  Theory does not decide the sign, as stage 1 said.

The stage-1 qualitative predictions all survive with the exact jitter:

- (0, 0) and every lam-0.95 line sit at 0.1;
- sigma* rises with delay at lam <= 0.8;
- sigma* is lowest at lam 0.95 and non-monotone in lam;
- sigma* rises with kappa and is roughly unchanged with delta.

## 6. Remaining assumptions

- The planner's noise-free optimum equals my reduced model's optimum to within 1e-3, as argued
  in section 2.
- The 100-point midpoint rule is the jitter expectation. The 20 evaluation episodes are a
  random sample of it: per-start G0 spans up to 0.007 to 0.024 on one line, so the sample-mean
  G0 is uncertain by about +/-0.002.
- Patient sampling within a quarter has no effect. Both policies are pure in class shares, apart
  from a few Adam-polished interior quarters, where the effect is O(1e-4).
- Reflection keeps the shock mean exactly zero. This is used only for planned-once, where it is
  exact.
