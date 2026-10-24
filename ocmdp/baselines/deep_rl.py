"""PPO, DQN and standard SAC comparators. Owner: Soumitra. Not implemented yet.

Run CleanRL's ppo.py, dqn.py and sac_atari.py unmodified against the Gymnasium ids in
ocmdp.envs; the only change is the env id. Standard discrete SAC is the comparison that
isolates pushforward entropy from action entropy, so it must share every
hyperparameter with ocmdp.agents.oc_sac except the entropy term and alpha(s).
"""
