# Prior-art pass, areas 1-4 (hereditary and delay control; drug cycling; structured entropy; self-induced kernels)

Access date for every entry: 2026-11-26. One file for all four areas, because the coordinator capped tool calls at delivery; sections follow the four areas.
An entry marked "opened" had its abstract page, arXiv API record or full text fetched. Items seen only in search snippets are listed at the end with no tag.

## PRE-EMPTS (listed first)

- **Baram, Tennenholtz, Mannor, "Action Redundancy in Reinforcement Learning", UAI 2021 (MinRed)**
  - URL: https://ar5iv.labs.arxiv.org/html/2102.11329 (full text opened); https://proceedings.mlr.press/v161/baram21a.html (opened)
  - Claim: action entropy "does not necessarily coincide with state entropy". They maximise transition entropy, decomposed as g_pi(s,a) = H(s'|s,a) + KL(P(.|s,a) || P(.|s;pi)), which in the deterministic case is -log(pi(a|s) + eta(s,a)). Read as "the action entropy of a modified action space in which all redundant actions are unified". MinRed SAC and MinRed Q-learning/DQN, with a learned backward model q_phi(a|s,s'). No code link found.
  - Tag: **PRE-EMPTS** (method part of the toy entropy claim).
  - Reason: SAC with entropy over classes of effect-equivalent actions, against plain action entropy, is MinRed's contribution. Pushforward entropy over mechanism classes with exactly interchangeable drugs is its known-structure special case.
  - What still stands if it holds: only the substantive question of whether class-level entropy preserves efficacy (expected failure rate) at parity in an occupancy-coupled MDP. There entropy acts on long-run selection pressure, not on exploration. B2's fixed, measured, partially overlapping kernel is not MinRed's learned soft unification, but B2 is cut.

## Area 1: hereditary, delay and memory control; dual control; feedback against open loop; performative RL

- **Maynard-Zhang, Choi, Sattar, Jedra, Dean, Fazel, arXiv 2610.33412 (Oct 2026)**, https://arxiv.org/abs/2610.33412 (opened)
  - Claim: in an LQ system with bilinear observations, they compare optimal, CE and optimal open-loop control. Small information strength: nothing beats open loop by more than O(1). Large: CE beats open loop by Omega(T). Intermediate: a probing policy beats open loop by Omega(T) and CE gains only O(1).
  - Tag: RELATED. Reason: a regime characterisation of OL/CE/optimal gaps, but the dual effect runs through observations. There is no action-history kernel, delay or memory, and no blind boundary test. Closest control paper for Claim A. Verified; the earlier tag stands.
- **Mohamed et al., "On the Feedback Law in Stochastic Optimal Nonlinear Control", arXiv 2004.01041**, https://arxiv.org/abs/2004.01041 (API record opened)
  - Claim: repeatedly re-solving deterministic open-loop problems gives a feedback law within O(eps^4) of the stochastic optimum, for small noise eps.
  - Tag: RELATED. Reason: a theoretical prediction for Claim A's component (iii), the suboptimality of CE replanning, at small noise. It must be cited, and it bears on any "derived boundary".
- **Derman, Dalal, Mannor, "Acting in Delayed Environments with Non-Stationary Markov Policies", ICLR 2021**, https://arxiv.org/abs/2101.11992 (opened)
  - Claim: under execution delay, deterministic non-stationary Markov policies on the original state suffice, and stationary ones are suboptimal. Delayed-Q without augmentation. Code: github.com/galdl/rl_delay_basic, github.com/galdl/rl_delay_atari.
  - Tag: BASELINE. Reason: the strongest published delay-aware RL with code, for the completeness item "delay-aware RL".
- **Basu, Legault, Romero-Soriano, Precup, "Delay and Prolongedness of Action Effects" (PAE-POMDP), AAAI 2023**, https://arxiv.org/abs/2301.00512 (opened)
  - Claim: delayed and prolonged drug effects break the Markov property. They convert the PAE-POMDP to an MDP by state augmentation, are competitive with recurrent policies, and are cheaper.
  - Tag: BASELINE. Reason: the published source for augmenting with an exponentially decaying action effect, against a recurrent policy. It supplies the missing source for a recurrent comparator without the pending feature. No code mentioned.
- **Gu and Wang, arXiv 2403.07566**, https://arxiv.org/abs/2403.07566 (API record opened)
  - Claim: they model drug concentration with exponential decay to turn glucose control from a PAE-POMDP into an MDP.
  - Tag: RELATED. Reason: the same exponential action memory, in a single-drug dosing setting.
- **Rantzer, "Linear Quadratic Dual Control", arXiv 2312.06014**, https://arxiv.org/abs/2312.06014 (opened)
  - Claim: analyses CE adaptive control through a data-driven Riccati equation, with stability and robustness conditions.
  - Tag: RELATED. Reason: CE in dual control under parameter uncertainty, not a self-induced kernel.
- **Mandal, Triantafyllou, Radanovic, "Performative Reinforcement Learning", arXiv 2207.00046**, https://arxiv.org/abs/2207.00046 (API record opened)
  - Claim: the deployed policy changes rewards and transitions across rounds. Repeated regularised optimisation converges to a performatively stable policy.
  - Tag: RELATED. Reason: the kernel depends on the deployed policy across retraining rounds, not on within-episode action history.
- **Rank, Triantafyllou, Mandal, Radanovic, "Performative RL in Gradually Shifting Environments", UAI 2024**, https://arxiv.org/abs/2402.09838 (opened)
  - Claim: "the current environment depends on the deployed policy as well as its previous dynamics". The algorithm MDRR pools deployments.
  - Tag: RELATED. Reason: memory across deployments, not inside an episode. No feedback/open-loop analysis.
- **Basu et al., "Performative Policy Gradient", arXiv 2512.20576**, https://arxiv.org/abs/2512.20576 (opened)
  - Claim: performative performance-difference lemma and policy-gradient theorem. PePG converges to performatively optimal policies.
  - Tag: RELATED. Reason: same distinction as above.
- **Mandal et al., "Local and Global Stability in Performative RL", arXiv 2609.06467**, https://arxiv.org/abs/2609.06467 (opened)
  - Claim: local against global mixed stability, and weighted Hedge.
  - Tag: RELATED.

## Area 2: drug cycling, evolutionary steering, RL and optimal control for drug scheduling

- **Weaver, King, Maltas, Scott, PNAS 2024, 121(16) e2303165121**, https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439 (opened)
  - Claim: RL agents on 15 beta-lactam landscapes beat naive paradigms and approach the MDP optimum, robust to measurement noise. RL-genotype holds up at observation delay of one step or less, and falls to about random above one. The SSWM dynamics have no memory. Code: evo_dm and rl_cycling on GitHub.
  - Tag: BASELINE (rung 1, already targeted). Reason: the published reference implementation exists. Its delay result touches Claim A, but it concerns observation delay in a memoryless kernel.
- **Maltas and Wood, PLoS Biology 2019, 17(10) e3000515**, https://pmc.ncbi.nlm.nih.gov/articles/PMC6834293 (opened, first 100k characters)
  - Claim: an MDP policy on the resistance profile beats random cycling and the best collateral-sensitivity cycles in simulation. Experiments use MDP-derived fixed sequences. Simulations suggest a predetermined cycle is near-optimal over the experimental timescale. No code seen.
  - Tag: RELATED (closest drug-domain feedback-against-open-loop comparison). Reason: it measures a feedback-versus-predetermined gap, but without a delay/memory/noise boundary or a decomposition.
- **Fischer, Vazquez-Garcia, Mustonen, "The value of monitoring to control evolving populations", PNAS 2015 (arXiv 1406.6957)**, https://arxiv.org/abs/1406.6957 (opened)
  - Claim: stochastic optimal control of evolving populations. "To control one must monitor": high-resolution monitoring is needed, and finite monitoring loses control.
  - Tag: RELATED. Reason: the earliest value-of-feedback statement in evolutionary control. No action-history kernel, delay boundary or CE decomposition.
- **Nichol et al., PLoS Comput Biol 2015, 11(9) e1004493**, https://pmc.ncbi.nlm.nih.gov/articles/PMC4567305 (opened)
  - Claim: a Markov-chain landscape model. About 70% of 2-4 drug sequences promote resistance to the final drug, and steering sequences are fixed (open loop). Code: github.com/Daniel-Nichol/EvolutionarySteering.
  - Tag: BASELINE (open-loop sequence reference for rungs 1-3). Reason: the published open-loop optimiser with code.
- **Lee and Blumberg, abx_amr_simulator, arXiv 2603.11369 / SoftwareX 2026**, https://arxiv.org/html/2603.11369 (opened)
  - Claim: a Gymnasium environment with "leaky-balloon" latent resistance pressure that rises with prescribing and decays without it. Configurable cross-resistance, noisy, biased and delayed observations. PPO and hierarchical RL supported. No benchmark results. Code: github.com/jl56923/abx_amr_simulator.
  - Tag: RELATED (already filed). Reason: an occupancy-coupled environment in all but name, but no open-loop/feedback analysis and no entropy study. It weakens any claim that rung-6 release is the first such benchmark.

## Area 3: entropy over structured or embedded action spaces

- MinRed: see PRE-EMPTS above.
- **Zhong, Yang, Zhao, "No Prior Mask", arXiv 2312.06258**, https://arxiv.org/abs/2312.06258 (opened)
  - Claim: filters redundant actions with a similarity factor from distances between state distributions and a modified inverse model. Code: https://github.com/zhongdy15/npm
  - Tag: BASELINE. Reason: the scope outcome kept it "only if code exists". Code exists.
- **Liu et al., "Reducing Action Space via Causal Effect Estimation", arXiv 2501.14543**, https://arxiv.org/abs/2501.14543 (opened)
  - Claim: a pretrained inverse-dynamics model estimates each action's causal effect and suppresses redundant actions during exploration. Code: https://github.com/agi-brain/cee
  - Tag: BASELINE. Reason: the most recent redundant-action method with code. A reviewer will name it beside MinRed.
- **Leibfried and Grau-Moya, "Mutual-Information Regularization in MDPs and Actor-Critic Learning" (MIRACLE), arXiv 1909.05950**, https://arxiv.org/abs/1909.05950 (opened)
  - Claim: the reference prior is optimised to the marginal action distribution (MI regularisation), beating SAC on MuJoCo. No code mentioned.
  - Tag: BASELINE. Reason: OC-SAC's entropy equals KL to a fixed class-balanced prior (decisions/2026-11-23.md). A reviewer will ask for the learned-prior version.
- **Zhou et al., "Revisiting Discrete Soft Actor-Critic" (SD-SAC), arXiv 2209.10081**, https://arxiv.org/abs/2209.10081 (opened)
  - Claim: entropy penalty, double average Q and Q-clip stabilise discrete SAC. Code: https://github.com/coldsummerday/SD-SAC.git
  - Tag: BASELINE (already adopted as the stabiliser set).
- **Dulac-Arnold et al., "Deep RL in Large Discrete Action Spaces", arXiv 1512.07679**, https://arxiv.org/abs/1512.07679 (opened)
  - Claim: embeds actions in continuous space and uses nearest-neighbour lookup, up to 1e6 actions.
  - Tag: RELATED. Reason: K <= 22 needs no large-action machinery.
- **Chandak et al., "Learning Action Representations for RL", ICML 2019, arXiv 1902.00183**, https://arxiv.org/abs/1902.00183 (opened)
  - Claim: the policy factors into a representation-space policy and a learned map to actions.
  - Tag: RELATED. Reason: learned drug embeddings without an entropy over them.
- **Lintunen, VendiRL, arXiv 2509.02930**, https://arxiv.org/abs/2509.02930 (API record opened)
  - Claim: the Vendi Score is used to evaluate and learn sets of diverse skills.
  - Tag: RELATED. Reason: diversity over skill sets, not a regulariser on one policy's action distribution.
- **DRAGON, arXiv 2504.15217**, https://arxiv.org/abs/2504.15217 (API record opened)
  - Claim: Vendi diversity is a distributional reward for diffusion fine-tuning.
  - Tag: RELATED.
- **arXiv 2410.15184 (action abstractions for amortized sampling)** and **arXiv 2206.03467 (state-action abstraction via successor representation)**, API records opened.
  - Tag: RELATED. Reason: action abstraction without entropy over the abstract actions as a policy regulariser.
- Null searches (arXiv API, 0 hits):
  - abs:"action similarity" AND entropy AND reinforcement;
  - abs:entropy AND "action embeddings" AND policy;
  - entropy over "action groups/classes/clusters" AND policy;
  - "similarity-sensitive" AND policy/reinforcement (one unrelated hit, 2510.12405);
  - redundant actions AND entropy (only MinRed flagged).
  - No Leinster-Cobbold or Vendi entropy used as a single-policy regulariser was found.

## Area 4: environments driven by the agent's own past actions; occupancy-dependent objectives

- **Tennenholtz, Merlis, Shani, Mladenov, Boutilier, "RL with History-Dependent Dynamic Contexts" (DCMDP), ICML 2023, arXiv 2302.02061**, https://arxiv.org/abs/2302.02061 (opened)
  - Claim: context evolves with history; logistic DCMDPs use aggregation functions of history; UCB regret; latent-space planning; MovieLens users shifting with recommendations.
  - Tag: RELATED (already filed), **closest formal framework** to an occupancy-coupled MDP. Whether its aggregation covers exponentially weighted action indicators with a delay was not checked in the full text.
- **Leqi et al., "Rebounding Bandits for Modeling Satiation Effects", arXiv 2011.06741 (NeurIPS 2021)**, https://arxiv.org/abs/2011.06741 (opened)
  - Claim: per-arm satiation follows a linear dynamical system driven by pulls. Greedy is optimal when arms share identical deterministic dynamics. Explore-Estimate-Plan for unknown dynamics.
  - Tag: RELATED (already filed). Reason: the nearest single-state special case. Its greedy-optimality result bears on Claim A's greedy contrast at sigma = 0.
- **Clerici, Laforgue, Cesa-Bianchi, "Linear Bandits with Memory", arXiv 2302.08345**, https://arxiv.org/abs/2302.08345 (opened)
  - Claim: reward depends on actions in a fixed window m with exponent gamma (rotting/rising). OFUL variant competing with cycling policies.
  - Tag: RELATED (already filed).
- **Laforgue et al., "Last Switch Dependent bandits", arXiv 2110.11819**, https://arxiv.org/abs/2110.11819 (opened)
  - Claim: reward is set by the time since the arm's last switch.
  - Tag: RELATED.
- **Zahavy et al., "Reward is enough for convex MDPs", arXiv 2106.00661**, https://arxiv.org/abs/2106.00661 (opened)
  - Claim: objectives convex in the stationary distribution; Fenchel min-max.
  - Tag: RELATED. Reason: there occupancy enters the objective; here it enters the kernel.
- **Zhang et al., "Variational Policy Gradient for RL with General Utilities", arXiv 2007.02151**, https://arxiv.org/abs/2007.02151 (opened)
  - Claim: policy gradient for concave utilities of the occupancy measure; global convergence.
  - Tag: RELATED. Same distinction.
- **Mutti et al., "Challenging Common Assumptions in Convex RL", NeurIPS 2022, arXiv 2202.01511**, https://arxiv.org/abs/2202.01511 (opened)
  - Claim: finite-trial objectives on the empirical occupancy differ from infinite-trial ones.
  - Tag: RELATED. Reason: the empirical own-occupancy within a trial is the closest objective-side analogue.

## Seen in search snippets only (not opened; no tag assigned; to chase)

- APS March Meeting 2026 abstract "Information and Optimization in Markov Decision Process Models of Drug Protocols", https://meetings-archive.aps.org/smt/2026/mar-p63/11/ (HTTP 403). The title suggests value of information in drug-protocol MDPs. **Must be opened before Claim A is frozen.** (Followed up below.)
- PubMed 41394675, "Reinforcement learning for adaptive control of phenotypically heterogeneous bacterial populations" (cookie wall).
- Kouyos, Abel zur Wiesch, Bonhoeffer 2011, https://pmc.ncbi.nlm.nih.gov/articles/PMC3048378 (earlier BASELINE; not reopened).
- Bergstrom, Lo, Lipsitch PNAS 2004, https://pmc.ncbi.nlm.nih.gov/articles/PMC516561
- Beardmore et al. MBE 2017, https://pmc.ncbi.nlm.nih.gov/articles/PMC5400377
- Lassig and Mustonen PNAS 2020, https://pmc.ncbi.nlm.nih.gov/articles/PMC7443876 (snippet: a phase diagram with an error threshold between successful and failed control; relevant to Claim A). (Followed up below.)
- Iram et al. Nat Phys 2021, https://www.biorxiv.org/content/10.1101/867143.full.pdf
- Gallagher et al. Cancer Res 2024, https://pubmed.ncbi.nlm.nih.gov/38569183/
- Brown, Smith, Sun Oper Res 2010, https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf
- Escudero et al. TOP 2007 (multistage VSS), https://econpapers.repec.org/RePEc:spr:topjnl:v:15:y:2007:i:1:p:48-64
- Ramstedt/Bouteiller et al., RL with random delays (DCAC), https://arxiv.org/pdf/2010.02966v3.pdf
- Grau-Moya et al. ICLR 2019 MIRL, https://iclr.cc/virtual/2019/poster/822
- SHIRO, https://arxiv.org/pdf/2212.12786
- Gallego-Posada et al. GAIT entropy, https://arxiv.org/html/1906.08325
- Personalized HeartSteps, https://arxiv.org/abs/1909.03539 (opened; the dosage variable is not defined in the abstract).

## Follow-up for Claim A (appended 2026-11-26)

The question for each source: does it derive or measure a boundary over (delay, memory, noise), or an equivalent, on where feedback control beats open-loop or predetermined schedules in drug or evolutionary control?

- **Lässig and Mustonen, "Eco-evolutionary control of pathogens", PNAS 2020, 117(33):19694-19704, doi 10.1073/pnas.1920263117**
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7443876 (full text opened)
  - Accessed: 2026-11-26.
  - Claim:
    - The phase diagram (Fig. 3) plots control efficiency over the scaled pathogen and host cost parameters (c_p, c_h).
    - "An error threshold separates parameter regions of efficient control from regions of compromised control" (Eqs. 13-14). Strong control sits on one side and weak control on the other.
    - The only closed-loop protocol is a "metastable" two-state rule: a baseline dose, plus a rescue boost once an escape mutant is detected above a frequency threshold. The paper calls this "closed loop control". It beats the stationary optimum when theta = 2 N_e mu < theta_c, that is, in small populations or at low mutation rates.
    - Computational (forecasting, HJB) control can beat evolutionary and regulatory control.
    - Delay is not a parameter; regulation is assumed faster than the pathogen dynamics. There is no explicit memory kernel. Observation noise is not modelled; the nearest analogue is the detection threshold x_esc.
    - No formal value-of-information quantity is computed.
  - Tag: **RELATED**.
  - Reason: its threshold is over costs, and its closed-loop-versus-stationary condition is over theta. Neither is over delay, memory or observation noise, and there is no CE/feedback decomposition. It is the nearest precedent for an analytic "where closed loop wins" condition in evolutionary control and must be cited beside Claim A.

- **APS Global Physics Summit / March Meeting 2026 abstract, "Information and Optimization in Markov Decision Process Models of Drug Protocols"**
  - URL: https://meetings-archive.aps.org/smt/2026/mar-p63/11/ (HTTP 403 on two direct fetches)
  - Accessed: 2026-11-26.
  - Source of the content: search-engine extracts of the abstract only. Authors not found. No arXiv or preprint version found: an arXiv API search on Maxwell's demon with drug or resistance returned only unrelated physics, and title and author searches failed.
  - Claim, as extracted:
    - The MDP model of drug protocols constrains "the mutual information between the genetic state of the disease and the drug administered for that genetic state, inducing a tradeoff between information and optimality".
    - The setup is called "feedback control, leading to a non-equilibrium steady-state driven by a kind of Maxwell Demon".
    - It is suggested as a guide to how often a tumour should be biopsied.
  - Tag: **RELATED** (provisional: the page itself was never opened).
  - Reason: it trades information against optimality through a mutual-information cap on the state-to-drug map. That is an information-constrained policy, not a boundary over delay, memory or observation noise, and no CE/feedback decomposition or open-loop comparison is mentioned in the extracts. Its MI cap is a cousin of our noise axis and of the MIRACLE-style regulariser.
  - Before Claim A freezes: the full abstract and author list should be obtained by a human with browser access.

- **Kratz and Adamczyk, "Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics", ICLR 2025, arXiv 2410.08439**
  - URL: https://arxiv.org/abs/2410.08439 (opened)
  - Accessed: 2026-11-26.
  - Claim:
    - Cells "leverage a memory of past environments" (non-Markovian dynamics).
    - Exact solutions are known for specific switching models, and "model-free deep RL is able to recover exact solutions".
    - They show "robust RL-based control strategies in environments with measurement noise and dynamic memory strength".
    - The abstract mentions no open-loop or fixed-schedule baseline and no delay.
  - Tag: **RELATED**.
  - Reason: memory strength and measurement noise both appear as axes in a drug-dosing control problem. They are used for RL robustness, not for a boundary on where feedback beats open loop. The full text should be checked for a comparison with fixed pulsing.
