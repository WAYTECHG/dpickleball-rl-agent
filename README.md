<div align="center">

# dPickleBall: Deep Reinforcement Learning Agent

<p>PPO agents trained with curriculum learning, reward shaping and league self-play · 2nd place of 13 teams</p>

<a href="https://www.youtube.com/live/CitFn9FK12s"><img src="https://img.shields.io/badge/Competition_recording-YouTube-FF0000?style=for-the-badge&amp;logo=youtube&amp;logoColor=white" alt="Competition recording on YouTube" /></a>
<a href="https://github.com/dPickleball/dPickleBallEnv"><img src="https://img.shields.io/badge/Original_environment-GitHub-171717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Original dPickleBall environment on GitHub" /></a>

<br />

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/Unity_ML--Agents-222222?style=flat-square&amp;logo=unity&amp;logoColor=white" alt="Unity ML-Agents" />
<img src="https://img.shields.io/badge/Reinforcement_Learning-PPO-2563EB?style=flat-square" alt="Proximal Policy Optimization" />
<img src="https://img.shields.io/badge/Competition-2nd_of_13-F59E0B?style=flat-square" alt="2nd place of 13 teams" />

</div>

## Overview

This project trains autonomous agents for the Unity dPickleBall environment with Proximal Policy Optimization (PPO). The agent learns three-dimensional discrete control: vertical movement, horizontal movement and paddle rotation.

Self-play between untrained policies produces weak interactions and little learning signal, because neither side can return the ball. The project addresses this with a **two-stage training framework**: foundation skills are learned first in a simplified wall environment, and the resulting policies are then improved through competitive league self-play.

| Method | Evaluation | Result |
| --- | --- | --- |
| PPO with a six-level curriculum, reward shaping and league self-play | Internal round-robin tournament, then a Top-4 confirmation stage | 2nd place among 13 competing teams |

| | |
| --- | --- |
| **Course** | AIT306, Deep Reinforcement Learning |
| **Institution** | Xiamen University Malaysia |
| **Team** | Group 11, April 2026 semester |

## Training pipeline

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

### Stage 1: foundation wall training

A six-level curriculum develops basic gameplay behaviour in a simplified wall-based environment.

| Level | Ball speed range | Rebound angle | Focus |
| --- | --- | --- | --- |
| 1 | 0.85–1.35 | ±45° | Basic tracking and contact |
| 2 | 2.35–3.05 | ±45° | Faster reactions |
| 3 | 0.35–0.80 | ±75° | Wide-angle interception |
| 4 | 1.85–2.35 | ±75° | Diverse medium-fast trajectories |
| 5 | 0.35–3.70 | ±75° | Mixed-speed adaptation |
| 6 | 0.80–3.70 | ±75° | Stronger controlled returns |

Reward shaping encourages effective movement, paddle alignment, successful returns and recovery positioning. Penalties discourage passive behaviour, poor contact and ineffective positioning. Different reward configurations were tried to produce agents with distinct playing styles.

### Stage 2: league self-play

After foundation training, the agents move to competitive self-play in the Unity environment, with PPO improving action selection through clipped policy updates.

The action space is `MultiDiscrete([3, 3, 3])`:

| Action | Available controls |
| --- | --- |
| Vertical movement | None, Up, Down |
| Horizontal movement | None, Right, Left |
| Paddle rotation | None, Counter-clockwise, Clockwise |

The opponent pool holds independently trained agents, historical checkpoints and earlier learner policies. To avoid overfitting to a single opponent, selection combines:

- **PSRO-inspired selection:** a population of policies for adaptive opponent sampling.
- **SFL-inspired frontier sampling:** priority for opponents near the learner's current skill level.
- **Softmax-based sampling:** a balance between opponent difficulty and diversity.

## Evaluation

Candidate agents were compared in internal round-robin tournaments, followed by a Top-4 confirmation stage.

| Rank | Agent | Series W–L | Game W–L | Point difference |
| --- | --- | --- | --- | --- |
| 1 | **RealFT** | **2–1** | **5–2** | **+20** |
| 2 | Haoren Original Lifted | 2–1 | 3–3 | −17 |
| 3 | Wilbert Blocker Converted | 1–2 | 4–4 | +1 |
| 4 | V67 Best 3 | 1–2 | 2–5 | −4 |

**RealFT** had the strongest internal result and was selected as the competition agent. It was then played against the agents of the other groups, and the team finished 2nd of 13.

RealFT uses a different checkpoint on each side, chosen through real-game matches against a reference opponent:

| Side | Checkpoint | Origin |
| --- | --- | --- |
| Left (`TeamX`, `model_left.pt`) | `realft_01600004` | PPO fine-tuned in the real Unity game, then 1.6M steps against a PFSP opponent pool |
| Right (`TeamY`, `model_right.pt`) | `realft_00300000`, mirrored | 300k-step real-game fine-tune, mirrored through the court's centre line |

Both are small TorchScript policies (13→128→128→64 tanh, separate vertical and horizontal heads, rotation fixed at 0). They act on 16 features extracted with OpenCV from the 168×84 camera frame.

## My contributions

This was a group project, and training and final selection were carried out collaboratively. My own work was:

- Trained and tuned an independent PPO agent.
- Tested reward and penalty configurations to improve gameplay behaviour.
- Evaluated trained policies through repeated testing and comparison.
- Contributed candidate policies for team-wide evaluation; my candidate reached the Top-4 stage.
- Took part in selecting the final competition agent.

## Technology stack

| Category | Technologies and methods |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch |
| Simulation | Unity, Unity ML-Agents |
| RL algorithm | Proximal Policy Optimization (PPO) |
| Training | Curriculum learning, reward shaping, league self-play |
| Opponent selection | PSRO-style and SFL-inspired sampling |
| Evaluation | Round-robin tournament |

## Repository structure

```text
dpickleball-rl-agent/
├── Competition Script/
│   ├── Competition.py                   # Runs a match in the Unity competition build
│   ├── teamX.py / teamY.py              # Left and right RealFT players (OpenCV features + policy)
│   └── model_left.pt / model_right.pt   # RealFT TorchScript checkpoints
├── Foundation training/                 # Curriculum-based wall training
├── League Self-Play Training/           # PPO training and opponent selection
└── README.md
```

## Getting started

Tested on Windows 10/11 with Python 3.10.12. A GPU is not required to run matches.

### 1. Create the Python environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) and [Git](https://git-scm.com/), then open Anaconda Prompt or PowerShell:

```powershell
conda create -n dpickleball pip python=3.10.12
conda activate dpickleball
```

### 2. Install the dPickleBall version of Unity ML-Agents

The game talks to Python through dPickleBall's own fork of ML-Agents, which adds the side channel used to start the match. Install the fork, **not** `mlagents-envs` from PyPI:

```powershell
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
cd dpickleball-ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
cd ..
```

Keep the `dpickleball-ml-agents` folder, because the `-e` install loads ML-Agents from it.

### 3. Install the other packages

```powershell
pip install numpy==1.23.5 opencv-python==4.7.0.72 matplotlib pynput==1.8.1
pip install torch==2.5.1
```

The training code also needs:

```powershell
pip install stable-baselines3 gymnasium tensorboard
```

### 4. Download the Unity game

Download the dPickleBall build from the [official dPickleBall Google Drive](https://drive.google.com/drive/u/0/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi), linked from the [dPickleBallEnv repository](https://github.com/dPickleball/dPickleBallEnv), and extract it. You need the Windows builds:

```text
dPickleball BuildFiles/
├── Competition/Windows/dp.exe   # Used to run matches
└── Training/Windows/dp.exe      # Used by League Self-Play Training
```

### 5. Clone this repository

```powershell
git config --global core.longpaths true
git clone https://github.com/WAYTECHG/dpickleball-rl-agent.git
cd dpickleball-rl-agent
```

The first line lets Windows check out the long checkpoint file names in `League Self-Play Training/league_pool_filtered/`. Without it, the clone can fail with `Filename too long`.

### 6. Run a match: RealFT against RealFT

```powershell
conda activate dpickleball
cd "Competition Script"
python Competition.py --env-path "C:\path\to\dPickleball BuildFiles\Competition\Windows\dp.exe"
```

1. The game opens on its title screen. Optionally type the player names, then click **START**.
2. Keep the game window in front while it plays. The game pauses when it loses focus.
3. The score is printed in the terminal. Press `Ctrl+C` to stop.

| Argument | Description |
| --- | --- |
| `--env-path` | Path to the Competition build `dp.exe`. Setting the `DPICKLEBALL_EXE` environment variable also works. |
| `--serve` | Serve code sent to the game (default `212`, as in the competition). |
| `--wait` | Seconds to wait for START to be clicked (default `300`). |

`teamX.py` (left) and `teamY.py` (right) load `model_left.pt` and `model_right.pt` from the same folder. To try another agent, replace those files with its TorchScript policies.

### 7. Train (optional)

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
| --- | --- |
| `UnityTimeOutException: The Unity environment took too long to respond` | Click **START** on the game's title screen, and check that `--env-path` points to `Competition/Windows/dp.exe`. |
| `ModuleNotFoundError: No module named 'mlagents_envs.envs.custom_side_channel'` | `mlagents-envs` from PyPI is installed in place of the dPickleBall fork. Run `pip uninstall mlagents-envs`, then repeat step 2. |
| `ModuleNotFoundError: No module named 'PIL'` or `'typing_extensions'` | `pip install Pillow typing_extensions` |
| `error: Filename too long` while cloning | Run `git config --global core.longpaths true` and clone again. |
| The game freezes mid-match | The game window lost focus. Click on it to continue. |

## Acknowledgements

Developed collaboratively for **AIT306, Deep Reinforcement Learning** at Xiamen University Malaysia. The project builds on the [original dPickleBall environment](https://github.com/dPickleball/dPickleBallEnv) and the [dPickleBall ML-Agents fork](https://github.com/dPickleball/dpickleball-ml-agents); credit for the simulation remains with their authors.
