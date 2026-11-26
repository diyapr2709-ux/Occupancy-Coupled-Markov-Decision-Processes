# Information-relaxation (perfect-information) dual bounds: method and code

Access date for every item: 2026-11-26. Gatherer: baseline-hunter.
Purpose: the gate in arena/2026-11-23-scope-outcome.md item 5. If the certainty-equivalent replanner sits within the phase test's margin of the perfect-information bound, no new noise-optimal reference is needed.

## Primary reference

- https://econpapers.repec.org/RePEc:inm:oropre:v:58:y:2010:i:4-part-1:p:785-801
  Brown, Smith, Sun, "Information Relaxations and Duality in Stochastic Dynamic Programs", Operations Research 58(4, Part 1) 785-801, 2010, DOI 10.1287/opre.1090.0796.
  The publisher page https://pubsonline.informs.org/doi/10.1287/opre.1090.0796 returned HTTP 403. The Duke repository page served a bot check. Only the abstract was read.
  Exact claim (abstract, paraphrased): the paper gives upper bounds on maximal values (lower bounds on minimal costs) by relaxing the nonanticipativity constraints and adding a penalty for violating them. It proves weak duality, strong duality and complementary slackness, and gives tight bounds in an adaptive inventory problem and in option valuation.
  Tag: BASELINE (the bound used by the gate).

## Penalty construction, as restated in an opened secondary source

- https://ar5iv.labs.arxiv.org/html/2007.14295
  Chen, Ma, Liu, Yu, "Information Relaxation and A Duality-Driven Algorithm for Stochastic Dynamic Programs", arXiv 2007.14295 (v. 24 Jul 2020), Section 2, reviewing Brown, Smith and Sun (2010). Cost-minimisation convention.
  - Perfect-information relaxation (before Eq. 7): the decision maker chooses actions after observing the whole noise path. By Jensen's inequality, E[ inf_a sum_t r_t | x_0 ] <= V_0(x_0) (Eq. 7). With zero penalty this is the perfect-information bound.
  - Penalty (Eq. 8): z(a, xi) = sum_t { E[r_t(x_t,a_t,xi_t) + W_{t+1}(f_t(x_t,a_t,xi_t))] - (r_t(x_t,a_t,xi_t) + W_{t+1}(f_t(x_t,a_t,xi_t))) }, for any sequence of functions W_t from state to reals. Each term has zero conditional mean, so the penalty is dual feasible and E[ inf_a (sum r + z) ] stays a lower bound.
  - Strong duality (Eq. 9): V_0(x) = sup_W E[ inf_a (sum r + z(a, xi)) | x_0 = x ]. Equality holds at the optimal choice of W. The standard reading is that W = the optimal value function gives the ideal penalty, and the source's Theorem 3.3(iii) is consistent with this. The source does not state it explicitly in Section 2.
  - Policy gap (after Eq. 9): 0 <= Vbar_0(x) - V_0(x) <= Vbar_0(x) - Vunder_0(x), where Vbar is a feasible policy's cost and Vunder the dual bound.
  Tag: RELATED (secondary restatement; equation numbers are the secondary source's, not those of Brown, Smith and Sun 2010).

Estimation procedure (inference from the above, not quoted from a source): draw noise paths. For each path, solve the inner deterministic problem over the K^T action sequences, or by deterministic DP on the noise-fixed path, with the penalty added. Average, and report a Monte Carlo interval. In this setting the natural W is the certainty-equivalent planner's value function. The perfect-information bound (W = 0) is the cheapest first gate.

## Reference code

- https://github.com/amspector100/dual_bounds_paper. Replication code for Ji, Lei and Spector (2023). No licence is shown. The README does not define "dual bounds". By the authors' names this is very likely a causal-inference partial-identification paper, not information relaxation, but that was not confirmed. Tag: RELATED, and not a reference implementation.
- No public reference code for Brown, Smith and Sun (2010) information-relaxation bounds was found.

## Further reading listed by search, not opened

- Brown and Smith, review and tutorial on information relaxations: https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxation_Review_Paper.pdf. The fetched PDF could not be decoded and its text was not read.

## Unsourced

- The penalty construction and theorem numbering from the 2010 paper itself (full text not opened).
- Any public implementation, with a licence.
