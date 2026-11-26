# Sample-average approximation for open-loop schedules under noise: references and statistical bounds

Access date for every item: 2026-11-26. Gatherer: baseline-hunter.
Purpose: Claim A component (i), the best open-loop schedule under noise, by sample-average approximation (arena/2026-11-23-scope-outcome.md items 2 and 5).
Fit to this setting: an open-loop schedule over T quarters with K drugs is a single-stage stochastic program over a finite feasible set of size K^T, so log|S| = T log K. That makes the finite-feasible-set theory below the directly relevant one. This reading is the gatherer's and is not stated in the sources.

## Primary reference

- https://portal.mardi4nfdi.de/wiki/Publication:2784421
  Kleywegt, Shapiro, Homem-de-Mello, "The sample average approximation method for stochastic discrete optimization", SIAM Journal on Optimization 12(2) 479-502, DOI 10.1137/S1052623499363220 (publication date on the page: 23 April 2002).
  Exact claim (zbMATH review on the page, plus the abstract text returned by search): for problems with a finite feasible set, the SAA optimal solution is an exact optimum of the true problem with probability approaching one exponentially fast in the sample size. The paper discusses convergence rates, stopping rules and computational complexity, with a stochastic knapsack example.
  Tag: BASELINE (the method and its convergence guarantee).
  The full text at https://core.ac.uk/download/127606343.pdf returned HTTP 403. The explicit sample-size estimate (of the form N >= (C / eps^2) log(|S| / alpha)) and its theorem number were therefore not verified.

## Statistical bounds on a candidate schedule

- Mak, Morton, Wood, "Monte Carlo bounding techniques for determining solution quality in stochastic programs", Operations Research Letters 24(1-2) 47-56, 1999, DOI 10.1016/S0167-6377(98)00054-6.
  Opened https://calhoun.nps.edu/bitstream/10945/38421/1/MakMortonWood.pdf. The page returned no readable text. The bohrium and researchr pages failed (WAF challenge; connection refused).
  Claim (from search summaries only): confidence intervals on the optimality gap of a candidate solution, built from the expected SAA optimal value, which is a lower bound on the true minimum. Common random numbers keep gap estimates non-negative and reduce variance.
  Tag: BASELINE. Status: citation not verified from an opened page.

- Kim, Pasupathy, Henderson, "A Guide to Sample Average Approximation", Ch. 8 of Handbook of Simulation Optimization (ed. Fu), Springer ISOR vol. 216, pp. 207-243, DOI 10.1007/978-1-4939-1384-8_8. The Springer page redirected to a login and was not read. The bibliographic data come from search results only.
  Tag: RELATED (survey).

- https://www.lancaster.ac.uk/pg/rhodesle/SAA_blog.php (Rhodes-Leader, 16 Feb 2016; a blog, so not a citable source for a paper)
  Claim: SAA consistency via the uniform law of large numbers, and an n^(-1/2) rate for the solution under a unique optimum with Lipschitz and quadratic-growth conditions. Cites Kim, Pasupathy and Henderson.
  Tag: RELATED.

## Reference code

None found. The search found no public SAA implementation for finite-set schedules, and no code is linked from the pages above.

## Unsourced

- The verified theorem number and sample-size formula from Kleywegt, Shapiro and Homem-de-Mello (full text 403).
- The verified multiple-replication gap interval from Mak, Morton and Wood (full text unreadable).
- Any result on SAA when the feasible set is searched heuristically rather than enumerated. K^T is too large to enumerate at K = 20, so the planner's schedule search is not an exact SAA optimum, and the exponential-rate guarantee would not apply as stated.
