# 🏓 dPickleBall — Deep Reinforcement Learning Agent

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unity](https://img.shields.io/badge/Unity_ML--Agents-222222?style=flat-square&logo=unity&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement_Learning-PPO-2563EB?style=flat-square)
![Competition](https://img.shields.io/badge/Competition-2nd_of_13-F59E0B?style=flat-square)

A **PPO-based multi-agent reinforcement learning system** developed to train autonomous agents in the Unity dPickleBall environment using curriculum learning, reward shaping, and adaptive league self-play.

**Course:** AIT306 — Deep Reinforcement Learning  
**Institution:** Xiamen University Malaysia  
**Project:** Group 11 | April 2026 Semester  
**Achievement:** 🥈 2nd Place among 13 competing teams

[Competition Recording](https://www.youtube.com/live/CitFn9FK12s) · [Original Environment](https://github.com/dPickleball/dPickleBallEnv)

---

## Project Overview

This project explores reinforcement learning strategies for improving autonomous gameplay in a simulated pickleball environment. The agent learns three-dimensional discrete action control involving vertical movement, horizontal movement, and paddle rotation.

A major challenge in multi-agent reinforcement learning is training agents that initially lack fundamental gameplay skills. Direct self-play between untrained policies can produce ineffective interactions and limited learning signals.

To address this, the project introduces a **two-stage training framework**, combining foundational skill development with competitive policy optimization.

## Training Pipeline

```text
              dPickleBall Environment
                         │
                         ▼
                Foundation Training
               Six-Level Wall Curriculum
                         │
                         ▼
                 PPO + Reward Shaping
                         │
                         ▼
                  Trained Policies
                         │
                         ▼
                 League Self-Play
                Dynamic Opponent Pool
                         │
                         ▼
             PSRO / SFL-Inspired Selection
                         │
                         ▼
                Round-Robin Evaluation
                         │
                         ▼
                 Final Agent: RealFT
```

### 1. Foundation Wall Training

A six-level curriculum progressively develops essential gameplay behaviors in a simplified wall-based environment.

| Level | Ball Speed Range | Rebound Angle | Focus |
|---|---|---|---|
| 1 | 0.85–1.35 | ±45° | Basic tracking and contact |
| 2 | 2.35–3.05 | ±45° | Faster reactions |
| 3 | 0.35–0.80 | ±75° | Wide-angle interception |
| 4 | 1.85–2.35 | ±75° | Diverse medium-fast trajectories |
| 5 | 0.35–3.70 | ±75° | Mixed-speed adaptation |
| 6 | 0.80–3.70 | ±75° | Stronger controlled returns |

**Reward shaping** encourages effective movement, paddle alignment, successful returns, and recovery positioning. Penalties discourage passive behavior, poor contact, and ineffective positioning.

Different reward configurations were explored to produce agents with distinct playing behaviors and strengths.

### 2. League Self-Play Training

Following foundation training, the agents transition to competitive self-play within the Unity environment.

**Proximal Policy Optimization (PPO)** is used to improve action selection through repeated interactions and clipped policy updates.

The agent operates with a multidiscrete action space:

`MultiDiscrete([3, 3, 3])`

| Action | Available Controls |
|---|---|
| Vertical movement | None, Up, Down |
| Horizontal movement | None, Right, Left |
| Paddle rotation | None, Counter-clockwise, Clockwise |

An opponent pool containing independently trained agents, historical checkpoints, and previous learner policies supports diverse training experiences.

To avoid relying on a single opponent, the system incorporates:

- **PSRO-inspired selection:** Uses a population of policies for adaptive opponent sampling.
- **SFL-inspired frontier sampling:** Prioritizes opponents near the learner's current skill level.
- **Softmax-based sampling:** Balances opponent difficulty and selection diversity.

This design aims to improve policy robustness and adaptability against different gameplay strategies.

## Model Evaluation

Candidate agents were compared through internal round-robin tournaments, followed by a Top-4 confirmation stage.

| Rank | Agent | Series W–L | Game W–L | Point Difference |
|---|---|---|---|---|
| 1 | **RealFT** | **2–1** | **5–2** | **+20** |
| 2 | Haoren Original Lifted | 2–1 | 3–3 | −17 |
| 3 | Wilbert Blocker Converted | 1–2 | 4–4 | +1 |
| 4 | V67 Best 3 | 1–2 | 2–5 | −4 |

**RealFT** achieved the strongest overall internal tournament result and was selected as the final competition agent.

RealFT uses a different checkpoint on each side, picked by real-game matches against a reference opponent:

| Side | Checkpoint | Origin |
|---|---|---|
| Left (`TeamX`, `model_left.pt`) | `realft_01600004` | PPO fine-tuned in the real Unity game, then 1.6M steps against a PFSP opponent pool |
| Right (`TeamY`, `model_right.pt`) | `realft_00300000` mirrored | 300k-step real-game fine-tune, mirrored through the court's centre line |

Both are small TorchScript policies (13→128→128→64 tanh, separate vertical/horizontal heads, rotation fixed at 0) that act on 16 features extracted from the 168×84 camera frame with OpenCV.

The final agent was then evaluated against agents developed by other participating groups.

**[Watch the competition on YouTube](https://www.youtube.com/live/CitFn9FK12s)**

## My Contributions

My responsibilities focused on independent reinforcement learning experimentation and collaborative agent evaluation:

- Trained and optimized an autonomous PPO-based agent.
- Experimented with reward and penalty configurations to improve gameplay behavior.
- Evaluated trained policies through iterative testing and performance comparison.
- Contributed candidate policies for team-wide evaluation.
- Participated in selecting the strongest agent for the final competition.

The training and final selection process was conducted collaboratively as part of the group project.

## Technology Stack

| Category | Technologies / Methods |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Simulation | Unity, Unity ML-Agents |
| RL Algorithm | Proximal Policy Optimization (PPO) |
| Training | Curriculum Learning, Reward Shaping, League Self-Play |
| Opponent Selection | PSRO-Style, SFL-Inspired Sampling |
| Evaluation | Round-Robin Tournament |

## Repository Structure

```text
dpickleball-rl-agent/
│
├── Competition Script/
│   ├── Competition.py        Runs a match in the Unity competition build
│   ├── teamX.py / teamY.py   Left / right RealFT players (OpenCV feature extraction + policy)
│   └── model_left.pt / model_right.pt   RealFT TorchScript checkpoints
│
├── Foundation training/
│   └── Curriculum-based wall training
│
├── League Self-Play Training/
│   └── PPO training and opponent selection
│
└── README.md
```

## Installation and Execution

Tested on Windows 10/11 with Python 3.10.12. A GPU is not required to run matches.

### 1. Create the Python environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) and [Git](https://git-scm.com/), then open **Anaconda Prompt** or PowerShell:

```powershell
conda create -n dpickleball pip python=3.10.12
conda activate dpickleball
```

### 2. Install the dPickleBall version of Unity ML-Agents

The game talks to Python through dPickleBall's own fork of ML-Agents (it adds the side channel used to start the match). Install the fork, **not** `mlagents-envs` from PyPI:

```powershell
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
cd dpickleball-ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
cd ..
```

Keep this `dpickleball-ml-agents` folder: the `-e` install loads ML-Agents from it.

### 3. Install the other packages

```powershell
pip install numpy==1.23.5 opencv-python==4.7.0.72 matplotlib pynput==1.8.1
pip install torch==2.5.1
```

For the training code you also need:

```powershell
pip install stable-baselines3 gymnasium tensorboard
```

### 4. Download the Unity game

Download the dPickleBall build from the [official dPickleBall Google Drive](https://drive.google.com/drive/u/0/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi) (linked from the [dPickleBallEnv repository](https://github.com/dPickleball/dPickleBallEnv)) and extract it. You need the Windows builds:

```text
dPickleball BuildFiles/
├── Competition/Windows/dp.exe   ← used to run matches
└── Training/Windows/dp.exe      ← used by League Self-Play Training
```

### 5. Clone this repository

```powershell
git config --global core.longpaths true
git clone https://github.com/WAYTECHG/dpickleball-rl-agent.git
cd dpickleball-rl-agent
```

The first line lets Windows check out the long checkpoint file names in `League Self-Play Training/league_pool_filtered/`. Without it, the clone can fail with `Filename too long`.

### 6. Run a match: RealFT vs RealFT

```powershell
conda activate dpickleball
cd "Competition Script"
python Competition.py --env-path "C:\path\to\dPickleball BuildFiles\Competition\Windows\dp.exe"
```

1. The game opens on its title screen. Optionally type the player names, then click **START**.
2. Keep the game window in front while it plays. The game pauses when it loses focus.
3. The score is printed in the terminal. Press `Ctrl+C` to stop.

| Argument | Description |
|---|---|
| `--env-path` | Path to the Competition build `dp.exe`. Setting the `DPICKLEBALL_EXE` environment variable also works. |
| `--serve` | Serve code sent to the game (default `212`, as in the competition). |
| `--wait` | Seconds to wait for START to be clicked (default `300`). |

`teamX.py` (left) and `teamY.py` (right) load `model_left.pt` and `model_right.pt` from the same folder. To try another agent, replace those files with its TorchScript policies.

### 7. (Optional) Train

**Foundation training** runs in a lightweight Python wall simulator, so the Unity game is not needed:

```powershell
cd "Foundation training"
python run_train.py
```

**League self-play training** runs 6 learners in parallel in the Unity Training build. Point it at your `Training/Windows/dp.exe`:

```powershell
cd "League Self-Play Training"
python run_fast_training.py --build-path "C:\path\to\dPickleball BuildFiles\Training\Windows\dp.exe"
python run_normal_finetune.py --build-path "C:\path\to\dPickleball BuildFiles\Training\Windows\dp.exe"
```

See the README in each folder for details and options.

### Troubleshooting

| Problem | Fix |
|---|---|
| `UnityTimeOutException: The Unity environment took too long to respond` | Click **START** on the game's title screen, and check that `--env-path` points to `Competition/Windows/dp.exe`. |
| `ModuleNotFoundError: No module named 'mlagents_envs.envs.custom_side_channel'` | `mlagents-envs` from PyPI is installed instead of the dPickleBall fork. Run `pip uninstall mlagents-envs`, then repeat step 2. |
| `ModuleNotFoundError: No module named 'PIL'` or `'typing_extensions'` | `pip install Pillow typing_extensions` |
| `error: Filename too long` while cloning | Run `git config --global core.longpaths true` and clone again. |
| The game freezes mid-match | The game window lost focus. Click on it to continue. |

## Acknowledgements

Developed collaboratively for **AIT306 — Deep Reinforcement Learning** at Xiamen University Malaysia.

The project extends the original dPickleBall simulation through reinforcement learning experimentation and agent training.

**Source Repositories**
- [Original dPickleBall Environment](https://github.com/dPickleball/dPickleBallEnv)
