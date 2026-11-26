# Context-Conditioned Masked DQN for GridGame 9×9

> A Reinforcement Learning project based on Deep Q-Networks (DQN), Action Masking, Partial Observability, Bayesian Belief Tracking, and QMDP approximations in a competitive GridGame environment.

**Course:** Reinforcement Learning  
**Academic Year:** 2025–2026  
**Author:** []  

---

## Overview

This project develops a context-conditioned Deep Q-Network agent capable of playing a competitive turn-based game on a 9×9 grid.

The agent learns a shared policy across multiple maps and opponent strategies using a standard Deep Q-Network (DQN), with action masking to prevent illegal actions.

The framework investigates different observability settings and approaches to uncertainty management:

- Fully Observable Markov Decision Process (MDP).
- Fog of War and Partial Observability (POMDP).
- Belief-DQN with Bayesian opponent-position tracking.
- QMDP approximation.
- Opponent-Unknown POMDP with Bayesian opponent-policy identification.

The same neural network architecture operates across different game contexts, allowing systematic comparisons between fully observable and partially observable reinforcement learning approaches.

### Main Features

- Standard DQN (not Double DQN).
- Context-conditioned learning.
- Fully Observable MDP.
- Fog of War POMDP.
- Belief-DQN.
- QMDP approximation.
- Bayesian opponent identification.
- Six opponent policies.
- Four different maps.
- Parallel environment collection.
- CPU-based experience replay.
- Target network.
- Action masking.
- GPU-batched inference.
- Stratified evaluation.
- Wilson confidence intervals.
- Stratified bootstrap confidence intervals.
- Checkpoint restoration and reproducible training.

---

## Table of Contents

1. [Environment](#1-environment)
2. [Reinforcement Learning Formulation](#2-reinforcement-learning-formulation)
3. [Observability Settings](#3-observability-settings)
4. [Belief-DQN](#4-belief-dqn)
5. [QMDP Approximation](#5-qmdp-approximation)
6. [Opponent-Unknown POMDP](#6-opponent-unknown-pomdp)
7. [Network Architecture](#7-network-architecture)
8. [Training Algorithm](#8-training-algorithm)
9. [Project Structure](#9-project-structure)
10. [Installation](#10-installation)
11. [Training](#11-training)
12. [Evaluation](#12-evaluation)
13. [Output Files](#13-output-files)
14. [Reproducibility](#14-reproducibility)
15. [Technical Design Choices](#15-technical-design-choices)
16. [Troubleshooting](#16-troubleshooting)
17. [License](#17-license)

---

# 1. Environment

## 1.1 Game Description

The environment is a competitive turn-based game played on a 9×9 grid containing fixed obstacles.

Two players compete:

- **Agent:** controlled by the learned DQN policy.
- **Opponent:** controlled by one of six predefined behavioral policies.

The agent must collect objectives, engage in combat, and maximize its Victory Points before the episode terminates.

### Game Specifications

| Property | Description |
|---|---|
| Grid size | 9×9 |
| Maps | 4 predefined maps |
| Players | Agent and Opponent |
| Episode horizon | 100 individual activations |
| Initial positions | (5,1) and (5,9) |
| Objectives | 3 active objectives at a time |
| Opponent policies | 6 |
| Total evaluation contexts | 24 |
| Discount factor | γ = 1 |
| Action space | 7 actions |

One activation corresponds to one player's turn.

### Actions

| Action | Description |
|---|---|
| North | Move one cell north |
| South | Move one cell south |
| East | Move one cell east |
| West | Move one cell west |
| Shoot | Perform ranged combat |
| Pass | Skip the current action |
| DUMMY | Forced activation when no decision is available |

### Movement

The player can move one cell in one of the four cardinal directions.

Movement is illegal when:

- The destination is outside the grid.
- The destination contains an obstacle.
- The destination is occupied by the opponent.

### Combat

**Ranged combat**

A shot is possible when:

\[
2 \leq d_{\text{Manhattan}} \leq 3.
\]

The probability of killing the opponent is:

\[
P(\text{kill}) = 0.20.
\]

Obstacles do not block ranged attacks.

**Melee combat**

Melee combat occurs automatically when the players occupy adjacent cells.

The combat probabilities are:

- Attacker kills defender with probability 0.75.
- Otherwise, the defender can counterattack with probability 0.50.

### Objectives

Three objectives are active at a time.

Capturing an objective provides:

\[
+1 \text{ Victory Point}.
\]

When all three objectives have been collected, a new set of three objectives is generated.

### Death and Respawn

When a player kills the opponent:

- The killer receives +2 Victory Points.
- The defeated player receives -5 Victory Points.
- Both players respawn at their initial positions.

Death does not terminate the episode.

### Episode Termination

An episode ends after 100 individual activations.

The winner is determined by the accumulated Victory Points.

If both players have the same score, the episode ends in a draw.

---

# 2. Reinforcement Learning Formulation

## 2.1 Fully Observable MDP

In the standard setting, the agent observes the complete game state.

The state can be represented as:

\[
s_t =
(
AgentPos,
OpponentPos,
Objectives,
Score,
Time,
Context
).
\]

The environment is modeled as a Markov Decision Process:

\[
\mathcal M =
(
\mathcal S,
\mathcal A,
P,
R,
\gamma
).
\]

Where:

- \(\mathcal S\): state space.
- \(\mathcal A\): action space.
- \(P(s'|s,a)\): transition function.
- \(R(s,a,s')\): reward function.
- \(\gamma\): discount factor.

The discount factor is:

\[
\gamma = 1.
\]

The objective is to learn an optimal action-value function:

\[
Q^*(s,a).
\]

The agent selects actions according to:

\[
a^* = \arg\max_a Q(s,a).
\]

No belief state is required because the complete state is observable.

## 2.2 State Representation

The neural network receives two inputs:

**Spatial observation**

\[
X_t \in \mathbb R^{4\times9\times9}.
\]

**Feature vector**

\[
f_t \in \mathbb R^{13}.
\]

The spatial representation contains the grid information, while the feature vector contains additional information such as scores, remaining time, turn information, map identification, and opponent-related features.

## 2.3 Action Masking

The environment provides a Boolean action mask:

\[
m_t \in \{0,1\}^{7}.
\]

Where:

\[
m_t(a)=
\begin{cases}
1 & \text{if action }a\text{ is legal}\\
0 & \text{otherwise}.
\end{cases}
\]

Illegal actions are excluded during action selection.

The same masking mechanism is applied during the Bellman target computation.

This prevents the agent from selecting invalid actions and avoids illegal actions influencing the estimated future value.

---

# 3. Observability Settings

The framework investigates different ways of handling incomplete information.

## 3.1 Fully Observable MDP

The agent observes:

- Its own position.
- Opponent position.
- Obstacles.
- Objectives.
- Scores.
- Remaining time.
- Context information.

The learned function is:

\[
Q(s,a).
\]

This is the baseline reinforcement learning configuration.

## 3.2 Fog of War POMDP

Fog of War introduces partial observability by restricting the visibility of the opponent.

The opponent is visible only when:

\[
d_{\text{Manhattan}} \leq fog\_radius.
\]

When the opponent is outside the visibility radius:

- Its position is hidden.
- The game rules remain unchanged.
- The reward function remains unchanged.
- The transition dynamics remain unchanged.

Only the observation changes.

The problem becomes a Partially Observable Markov Decision Process:

\[
\mathcal P =
(
\mathcal S,
\mathcal A,
P,
R,
\Omega,
O,
\gamma
).
\]

Where:

- \(\Omega\): observation space.
- \(O(o|s)\): observation model.

The agent can maintain a Bayesian belief over hidden opponent positions:

\[
b_t(s)
=
P(s_t=s\mid o_{1:t},a_{1:t-1}).
\]

The Bayesian filtering equations are:

**Prediction:**

\[
b^-_{t+1}(s')
=
\sum_s P(s'|s,a_t)b_t(s).
\]

**Observation update:**

\[
b_{t+1}(s')
=
\eta O(o_{t+1}|s')b^-_{t+1}(s').
\]

Here, \(\eta\) is a normalization constant.

When the opponent becomes visible, the belief is updated using the observed position.

---

# 4. Belief-DQN

Belief-DQN trains directly under partial observability.

Instead of receiving the complete state, the network receives a representation containing the agent's available information and a Bayesian belief distribution over possible opponent positions.

The network input includes:

- Map layout.
- Obstacles.
- Objectives.
- Agent position.
- Belief distribution over opponent positions.
- Additional game features.

The learned function becomes:

\[
Q(b,a)
\]

rather than:

\[
Q(s,a).
\]

The selected action is:

\[
a^* =
\arg\max_a Q(b,a).
\]

### Main Properties

- Training is performed under partial observability.
- Bayesian filtering maintains the opponent-position belief.
- The belief is directly provided to the neural network.
- The game dynamics and reward function remain unchanged.
- The network learns to make decisions under uncertainty.

Belief-DQN allows the agent to learn a policy directly from its information state rather than reconstructing the complete environment state.

---

# 5. QMDP Approximation

QMDP provides an alternative approach to decision-making under partial observability.

Unlike Belief-DQN, QMDP does not require a dedicated POMDP training procedure.

Instead, it uses a Q-network trained in the fully observable MDP setting.

During evaluation, the agent maintains a belief distribution over possible hidden states.

For each possible state, the corresponding Q-values are calculated and averaged according to the belief distribution.

The QMDP approximation is:

\[
Q_{QMDP}(b,a)
=
\sum_s b(s)Q(s,a).
\]

The selected action is:

\[
a^* =
\arg\max_a Q_{QMDP}(b,a).
\]

### Interpretation

QMDP assumes that uncertainty disappears after the current action.

Therefore, it approximates future decision-making as if the complete state were observable from the next step onward.

### Belief-DQN vs QMDP

| Property | Belief-DQN | QMDP |
|---|---|---|
| Training under POMDP | Yes | No |
| Training under MDP | No | Yes |
| Uses belief information | Yes | Yes |
| Weighted Q-value aggregation | No | Yes |
| Assumes future full observability | No | Yes |
| Learned function | \(Q(b,a)\) | \(Q(s,a)\) |
| Dedicated training required | Yes | No |

**Important:** QMDP requires a checkpoint trained in the fully observable MDP setting.

---

# 6. Opponent-Unknown POMDP

This setting introduces a different type of partial observability.

The opponent's position remains fully observable, but its behavioral policy is unknown.

The uncertainty concerns the identity of the opponent policy rather than its physical position.

## 6.1 Hidden Opponent Policy

At the beginning of each episode, an opponent policy is sampled from:

\[
K \in
\{
Random,
Aggressive,
Hunter,
Hybrid,
Shooter,
ASH
\}.
\]

The selected policy remains fixed throughout the episode.

The agent does not directly observe the opponent's identity.

Instead, it observes the actions performed by the opponent.

The augmented state can be represented as:

\[
s_t=(x_t,K)
\]

where:

- \(x_t\): observable game state.
- \(K\): hidden opponent policy.

## 6.2 Bayesian Belief

The agent maintains a probability distribution over possible opponent policies:

\[
b_t(k)
=
P(K=k\mid h_t).
\]

Here, \(h_t\) represents the available observation and action history.

The initial belief is:

\[
b_1(k)=q_O(k).
\]

By default, the prior is uniform:

\[
b_1(k)=\frac{1}{6}.
\]

A custom prior can also be provided.

## 6.3 Bayesian Update

After observing an opponent action \(u_t\), the belief is updated using Bayes' rule:

\[
b_{t+1}(k)
=
\frac{
b_t(k)\pi_k(u_t|x_t)
}{
\sum_j b_t(j)\pi_j(u_t|x_t)
}.
\]

Where:

\[
\pi_k(u_t|x_t)
\]

represents the probability that opponent policy \(k\) selects the observed action.

The posterior is updated exclusively using observable opponent behavior.

The implementation does not use:

- The true opponent identity.
- Internal simulator RNG states.
- Privileged environment information.
- Hidden policy identifiers.

## 6.4 Network Representation

The six belief probabilities are concatenated with the observable game features:

\[
b_t =
[b_t^1,b_t^2,\ldots,b_t^6].
\]

The network learns:

\[
Q(x_t,b_t,a).
\]

The selected action is:

\[
a^* =
\arg\max_a Q(x_t,b_t,a).
\]

### Main Properties

- Opponent position remains observable.
- Opponent behavioral policy is hidden.
- Bayesian inference identifies the most likely opponent strategy.
- The belief is reset at the beginning of each episode.
- Game rules, rewards, and transition dynamics remain unchanged.

This setting allows the agent to adapt its behavior according to the inferred opponent strategy.

---

# 7. Network Architecture

## 7.1 QNetwork

The Q-network combines a CNN spatial encoder with an MLP feature encoder.

```text
              Spatial Input
                (B, 4, 9, 9)
                     |
                     v
                Conv 3x3
                32 channels
                     |
                     v
                Conv 3x3
                32 channels
                     |
                     v
                Conv 3x3
                32 channels
                     |
                     v
                  Flatten
                     |
                     v
              Linear 2592 -> 128
                     |
                     +------------------+
                                        |
                                        v
                                    Concatenate
                                        ^
                                        |
              Feature Input             |
                 (B, 13)                 |
                     |                   |
                     v                   |
               Linear 13 -> 64          |
                     |                   |
                     v                   |
               Linear 64 -> 64           |
                     |                   |
                     +-------------------+
                                        |
                                        v
                                  Linear 192 -> 128
                                        |
                                        v
                                  Linear 128 -> 128
                                        |
                                        v
                                  Linear 128 -> 7
                                        |
                                        v
                                    Q-values


# 8. Training Algorithm

## 8.1 Deep Q-Learning

The agent is trained using a standard Deep Q-Network (DQN).

The learned action-value function is approximated by a neural network parameterized by:

\[
Q(s,a;\theta).
\]

For partially observable settings, the input state is replaced by the available information state or belief representation.

The Bellman target is:

\[
y_t
=
r_t
+
\gamma
\max_{a'}
Q_{target}(s_{t+1},a').
\]

The loss function is the Huber loss between predicted and target Q-values.

## 8.2 Experience Replay

Transitions are stored inside a replay buffer:

\[
(s_t,a_t,r_t,s_{t+1},done_t).
\]

Mini-batches are sampled uniformly.

Experience replay:

- removes temporal correlations;
- improves stability;
- increases sample efficiency.

## 8.3 Target Network

A separate target network is maintained:

\[
Q_{target}(s,a;\theta^-).
\]

Parameters are periodically synchronized from the online network.

This improves the stability of Bellman updates.

## 8.4 Action Masking

The environment provides a legal-action mask.

Only legal actions are considered when:

- selecting actions;
- computing Bellman targets.

This prevents invalid actions from influencing learning.

## 8.5 Exploration

The policy uses ε-greedy exploration:

\[
a=
\begin{cases}
\text{random legal action} & \text{with probability }\epsilon \\
\arg\max_a Q(s,a) & \text{otherwise}.
\end{cases}
\]

The value of ε decreases linearly during training.

# 9. Project Structure

```text
.
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
│
├── agent/
│   ├── network.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── collector.py
│   ├── action_selection.py
│   └── parallel_env.py
│
├── environment/
│   ├── game.py
│   ├── maps.py
│   ├── context.py
│   ├── observations.py
│   ├── fog_of_war.py
│   ├── belief.py
│   ├── action_mask.py
│   └── opponents.py
│
├── analysis/
│   └── performance_metrics.py
│
├── utils/
│   ├── checkpoint.py
│   ├── metrics.py
│   └── seeds.py
│
├── runs/
└── evaluations/
```

### Main Components

#### agent

Contains all reinforcement learning components:

- neural networks;
- replay buffer;
- training algorithm;
- action selection;
- experience collection.

#### environment

Contains the GridGame simulator and all game mechanics.

#### analysis

Contains evaluation and statistical analysis utilities.

#### utils

Contains helper functions, checkpoint management, and reproducibility utilities.

# 10. Installation

## Requirements

- Python 3.10+
- PyTorch 2.x
- NumPy
- PyYAML

GPU acceleration is recommended but not required.

## Clone Repository

```bash
git clone https://github.com/<username>/<repository>.git

cd <repository>
```

## Create Virtual Environment

Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Minimal requirements.txt

```text
numpy
torch
pyyaml
```

## Verify Installation

```bash
python train.py \
    --device cpu \
    --num-envs 2 \
    --num-workers 2 \
    --total-activations 2000 \
    --output-dir runs/smoke
```


# 11. Training

## Standard MDP

```bash
python train.py \
    --device cuda:0 \
    --observation-mode mdp \
    --num-envs 16 \
    --num-workers 8 \
    --total-activations 1600000 \
    --output-dir runs/mdp
```

## Fog of War

```bash
python train.py \
    --device cuda:0 \
    --fog-of-war \
    --fog-radius 4 \
    --num-envs 16 \
    --num-workers 8 \
    --total-activations 1600000 \
    --output-dir runs/fow
```

## Belief-DQN

```bash
python train.py \
    --device cuda:0 \
    --fog-of-war \
    --fog-radius 4 \
    --belief-filter \
    --num-envs 16 \
    --num-workers 8 \
    --total-activations 1600000 \
    --output-dir runs/belief_dqn
```

## Opponent-Unknown POMDP

```bash
python train.py \
    --device cuda:0 \
    --observation-mode pomdp \
    --num-envs 16 \
    --num-workers 8 \
    --total-activations 1600000 \
    --output-dir runs/pomdp
```

## Resume Training

```bash
python train.py \
    --resume runs/mdp/checkpoint_800000.pt \
    --total-activations 1600000 \
    --output-dir runs/mdp
```

# 12. Evaluation

Evaluation is performed using frozen policies.

No replay buffer updates or gradient computations are performed.

## MDP Evaluation

```bash
python evaluate.py \
    --checkpoint runs/mdp/checkpoint_1600000.pt \
    --observation full \
    --output-dir evaluations/mdp
```

## Fog of War Evaluation

```bash
python evaluate.py \
    --checkpoint runs/fow/checkpoint_1600000.pt \
    --observation mask \
    --fog-radius 4 \
    --output-dir evaluations/fow
```

## Belief-DQN Evaluation

```bash
python evaluate.py \
    --checkpoint runs/belief_dqn/checkpoint_1600000.pt \
    --observation belief \
    --fog-radius 4 \
    --output-dir evaluations/belief_dqn
```

## QMDP Evaluation

```bash
python evaluate.py \
    --checkpoint runs/mdp/checkpoint_1600000.pt \
    --observation qmdp \
    --fog-radius 4 \
    --output-dir evaluations/qmdp
```

## Opponent-Unknown POMDP Evaluation

```bash
python evaluate.py \
    --checkpoint runs/pomdp/checkpoint_1600000.pt \
    --output-dir evaluations/pomdp
```

## Metrics

The main evaluation metric is:

\[
P(win)
=
\frac{\text{wins}}
{\text{episodes}}.
\]

Additional reported statistics include:

- average return;
- behavioral metrics;
- Wilson confidence intervals;
- stratified bootstrap confidence intervals;
- performance by map;
- performance by opponent.

# 13. Output Files

## Training Output

```text
runs/
├── checkpoint_*.pt
├── metrics.jsonl
└── logs
```*
## Evaluation Output

```text
eva*uations/
├── summary.json
└── epis*des.jsonl
```

### summary.json

C*ntains:

- win rate;
- confidence *ntervals;
- bootstrap statistics;
* map-level performance;
- opponent*level*performance.

### episodes.jsonl

*ontains per-episode records:

- se*d;
- context;
- return;
- outcome;*- behavioral statistics.

# 16. Troubleshooting

### *UDA not detected

Verify PyTorch i*stallation:

```bash
python -c "im*ort torch; print(torch.cuda.is_ava*lable())"
```

### Resume fails

E*sure that:

- training configurati*n matches the original run*
- observation mode is unchanged;
* replay buffer capacity is unchang*d.

### QMDP evaluation fails

QMD* requires a checkpoint trained und*r the fully observable MDP setting*

### Out-of-memory errors

Reduce*

- batch size;
- number of parall*l environments;
- replay buffer*size.
# RL_Context-Conditioned-Reinforcement-Learning-for-a-Competitive-Grid-Game
