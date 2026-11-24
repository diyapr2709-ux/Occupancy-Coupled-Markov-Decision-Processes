# One hop from rebounding bandits and bandits with memory, filed 2026-11-23

Access date for every entry: 2026-11-23. No PRE-EMPTS or BASELINE found in this hop.

Queries (arXiv API, export.arxiv.org):
- abs:"rebounding bandits" OR abs:"bandits with memory" OR abs:"satiation" AND cat:cs.LG (14 results)
- abs:"recovering bandits" OR abs:"recharging bandits" OR abs:"rotting bandits" OR abs:"rested rising bandits" (9 results)
- abs:"past actions" AND abs:"reinforcement learning" AND (abs:"exponentially weighted" OR abs:"exponential decay" OR abs:"discounted history"): returned nothing (totalResults 0)
- abs:"action history" AND (abs:"transition" OR abs:"dynamics") AND (abs:"reinforcement learning" OR abs:"Markov decision"): 3 relevant, 8 omitted (action history used only as policy input)

## Opened (abstract page fetched)

- https://arxiv.org/abs/2504.08200 (Sato, Ito, "Influential Bandits: Pulling an Arm May Change the Environment", TMLR). Claim: selecting an arm changes other arms' future losses through an unknown symmetric PSD interaction matrix. Tag: RELATED. Reason: cross-arm coupling of own actions, like a cross-resistance matrix, but bandit, no decay or delay stated, no entropy.
- https://arxiv.org/abs/2610.05659 (Liu, Jia, "Bellman-Centric Learning: Near-Optimal Regret for Linear Bandits with Memory", Oct 2026). Claim: RSM-LinUCB, "learns as in linear bandits and plans as in reinforcement learning", for a known bounded matrix-valued memory map; matching minimax bounds. Tag: RELATED. Reason: planning over own-action memory, but known memory map, bandit setting, no delay or entropy.
- https://arxiv.org/abs/2403.11520 (Tanimoto, Fukumizu, "State-Separated SARSA", 2024). Claim: an RL algorithm for recovering bandits whose reward depends on rounds since the last pull, with rounds treated as states. Tag: RELATED. Reason: own-action-history state solved by RL; time since last pull, no coupling, no entropy.

## Seen only in the arXiv API feed (abstract text in the feed, page not opened)

- https://arxiv.org/abs/2110.11819 (Laforgue et al., last switch dependent bandits, satiation and seasonality). RELATED: payoff depends on time since last switch.
- https://arxiv.org/abs/2306.00338 (Foussoul et al., last switch dependent bandits with monotone payoffs). RELATED: same class.
- https://arxiv.org/abs/1910.14354 (Pike-Burke, Grunewalder, Recovering Bandits). RELATED: reward a function of time since last play.
- https://arxiv.org/abs/1702.07274 (Levine et al., Rotting Bandits) and https://arxiv.org/abs/1811.11043 (Seznec et al.). RELATED: reward decays with pull count.
- https://arxiv.org/abs/2610.19179 (Palma Foster et al., strategic queuing with geometric recharge). RELATED: geometric recharge after use, multi-agent queuing.
- https://arxiv.org/abs/2201.12408 (Ou et al., networked restless bandits). RELATED: pulls affect neighbouring arms.
- https://arxiv.org/abs/2604.03641 (Lee et al., delayed homomorphic RL). RELATED: delayed feedback handled by action-history state augmentation; delay of feedback, not own-history-driven dynamics.
- https://arxiv.org/abs/2605.04345 (Sintes et al., delayed MARL equivalence). RELATED: action and observation delay equivalence.

Correction (validator check of the source, 2026-11-24): 2604.03641 treats action-history state augmentation as the baseline it improves on, not as its own method.
