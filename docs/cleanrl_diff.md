# Diff against CleanRL

Base: `cleanrl/sac_atari.py` (discrete-action SAC) from github.com/vwxyzjn/cleanrl. Pin the commit hash here when the fork is taken.

The base is the discrete SAC, not `sac_continuous_action.py`, because the action set is a finite drug panel.

Planned changes, each to be documented with the exact lines once made:

1. **Environment.** Drop the Atari wrappers and CNN encoder. Use `gym.make` on the `ocmdp/*` ids and MLP critics.
2. **Action distribution.** The actor scores C2 embeddings instead of emitting raw logits per index, and the C5 mask is applied to logits before the softmax (`ocmdp.components.constraint_filter.mask_logits`).
3. **Entropy term.** `alpha * log pi` is replaced by `alpha(s) * H(psi#pi)` (`ocmdp.agents.losses.actor_loss`).
4. **Temperature.** The scalar `log_alpha` and its fixed target entropy become a state-conditioned head trained against `entropy_target(s)` (`ocmdp.agents.losses.temperature_loss`).
5. **Reserve critic and dual variable.** Add a third critic (C4) with its own target, and `xi` updated on a slow clock (`ocmdp.agents.lagrangian.DualVariable`).
6. **Clocks.** C1 and the C4 target update once per simulated quarter, and `xi` once every `dual_update_every` steps.

Everything else, including the replay buffer, the twin-critic min, Polyak target updates and the logging, stays as in CleanRL. The standard-SAC comparator must share all of it.
