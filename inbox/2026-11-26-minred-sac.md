# Comparator: MinRed (redundancy-aware) SAC, known-class and learned-inverse-model forms

Tag: BASELINE
Access date: 2026-11-26
Status: gathered, not promoted. Supplements section 3 of
inbox/2026-11-16-mechanism-level-sac.md. Nothing cloned, downloaded or run.

## Paper

- https://proceedings.mlr.press/v161/baram21a.html : Baram, Tennenholtz, Mannor, "Action
  redundancy in reinforcement learning", UAI 2021, PMLR 161:376-385. Links on the page:
  paper PDF, supplementary PDF, supplementary ZIP
  (https://proceedings.mlr.press/v161/baram21a/baram21a-supp.zip). No code link. The ZIP
  was NOT opened (downloading would breach the no-download rule); whether it contains
  code is unknown. The supplementary PDF could not be text-extracted.
- https://ar5iv.labs.arxiv.org/html/2102.11329 , as extracted:
  - ARS, Definition 2, Eq. 4: eta(s,a) = E_{a'~pi} 1_{R(s,a)}(a'), with
    R(s,a) = {a' != a : f(s,a') = f(s,a)} (deterministic transition f).
  - ARR, Definition 3: zeta(s,a,s') = log( q(a|s,s') / pi(a|s) ).
  - Known-class form: Section 5.1, "Alternatively, the ARR can be replaced by the ARS for
    deterministic domains." No experiment with that form is reported. With equivalence
    classes given (e.g. by a mechanism map), the ARS is computable without learning;
    this is the known-class comparator.
  - Learned form: Algorithm 1 "MinRed SAC"; critic target r + alpha*(pi_theta/pi_i)*zeta,
    inverse model q_phi(a|s,s') trained by cross-entropy (Eq. 7).
  - Hyperparameters, all for MuJoCo (continuous; Tables 5-7): 2 x 256 ReLU, batch 256,
    lr 3e-4, buffer 1e5, 2M steps, learning starts 1e4, train frequency 1000 with 1000
    gradient steps, gamma 0.99, tau 0.05; alpha and lambda per task (e.g. HalfCheetah
    alpha 0.05, lambda 0.05). Inverse-model architecture and target entropy: not given.
  - Discrete experiment: four-room with "Right" repeated n times, 100-step episodes;
    reported by the authors (Fig. 3), "As n increases, the SAC agent fails to reach the
    goal." How discrete SAC was set up and whether q was learned there: not stated; no
    hyperparameters.
  - Atari (Fig. 4) uses MinRed DQN, not SAC (Tables 2-4).

## Third-party implementation of MinRed (discrete)

- https://arxiv.org/abs/2312.06258 : Zhong, Yang, Zhao, "No Prior Mask: Eliminate
  Redundant Action for Deep Reinforcement Learning" (2023-12-11; AAAI-24 per a search
  snippet, the AAAI page was blocked). Code: https://github.com/zhongdy15/npm , HEAD
  07f38dbc45d0f2899927d84bb053441d2bc993bd (2023-12-11). Licence: NONE (API null, no
  LICENSE file). Contains min_red/ with min_red_ppo.py and action_model_trainer.py; no
  SAC file visible in the partial listing.
- https://ar5iv.labs.arxiv.org/html/2312.06258 , as extracted: MinRed-DQN and
  MinRed-PPO baselines on discrete tasks (four-room with 4/11/19/35 actions, Maze with
  16/64/256 actions, MiniGrid 7, Atari 18). MinRed-PPO configuration: lr 3e-4, N-value
  update interval 11, Adam, gamma 0.99, entropy coefficient 0.15, update rate 2048,
  hidden [64, 64] Tanh, batch 64, buffer 5e4, lambda searched over {0.0005, 0.001,
  0.005, 0.01, 0.05, 0.1, 0.5}. Whether lambda was tuned per task: not extracted.
  No MinRed SAC configuration.

Tuned discrete MinRed SAC configuration: NOT FOUND. Official MinRed code: NOT FOUND
(supplementary ZIP unchecked).
