# Comparator: hierarchical or two-level SAC for grouped actions

Tag: BASELINE
Access date: 2026-11-26
Status: gathered, not promoted. Supplements inbox/2026-11-16-mechanism-level-sac.md
(Hybrid SAC, SOAC, Soft Options Critic already filed). Nothing cloned or run.

## HSD-3 factorised SAC: discrete first level with its own temperature, reference code

- https://arxiv.org/abs/2110.10809 : Gehring, Synnaeve, Krause, Usunier, "Hierarchical
  Skills for Efficient Exploration", NeurIPS 2021. Code page linked:
  https://facebookresearch.github.io/hsd3 (not opened).
- https://ar5iv.labs.arxiv.org/html/2110.10809 , as extracted: the high-level policy
  picks a discrete feature subset F, then a continuous goal conditioned on it.
  Discrete policy evaluated over all discrete actions (Appendix D.1); its entropy
  computed "in closed form", offset by log|F| (Section 3.3); "separate, automatically
  adjusted temperature coefficients" for discrete and continuous parts. Table 5:
  discrete target entropy 0.5*log|F|, continuous target -1, initial temperatures 1,
  lrs Q 1e-3, discrete policy 3e-3 (Walker) / 1e-3 (Humanoid), goal policy 3e-3,
  temperatures 1e-3, tau 0.005, gamma 0.99, buffer 1e6 / 2e6, batch 256 / 512, 50
  gradient steps per 50 (Walker) or 500 (Humanoid) env steps, warmup 1e3 / 1e4.
  Appendix E reports searching lrs over {1e-4, 3e-4} and initial temperatures over
  {0.1, 1}, which conflicts with Table 5 as extracted.
- Code: https://github.com/facebookresearch/hsd3 , HEAD
  2fdacb3accf3056637d6420358a9349898f7de3f (2022-06-03). Licence: MIT (README: "hsd3 is
  MIT licensed"). Hydra configs in config/. Whether the factorised-SAC file can be read
  in isolation: not checked.
- Relevance: closest released code for a two-level SAC whose upper level is discrete
  with its own learned temperature; the lower level is continuous (MuJoCo), so the
  mechanism-then-drug form needs a discrete lower level. The tuned values are for
  locomotion, not grouped discrete actions.

## Other leads, not comparators as found

- https://github.com/MLD3/OfflineRL_FactoredActions (Tang et al., NeurIPS 2022, offline
  RL with factored action spaces in healthcare; search result only, repository not
  opened). Product-factored actions with linear Q decomposition, not a group-then-member
  hierarchy. RELATED rather than BASELINE.
- arXiv 2206.05108 (multi-agent hybrid-action maximum entropy) and ACM 10.1145/3643862
  (Bayesian strategy networks SAC): search results only, not opened.

Published tuned configuration for a group-then-member discrete SAC: NOT FOUND.
Reference code for that exact form: NOT FOUND.
