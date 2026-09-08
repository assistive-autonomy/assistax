---
license: apache-2.0
pretty_name: Assistax Zoo
task_categories:
  - reinforcement-learning
tags:
  - robotics
  - assistive-robotics
  - multi-agent-reinforcement-learning
  - zero-shot-coordination
  - ad-hoc-teamwork
  - jax
  - brax
  - mujoco
---

# 🦓 Assistax Zoo

<div style="display: flex; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/assistive-autonomy/assistax/main/docs/imgs/task_scratch.png" alt="Scratch Itch" style="width: 19%;">
  <img src="https://raw.githubusercontent.com/assistive-autonomy/assistax/main/docs/imgs/task_bed_bathing.png" alt="Bed Bathing" style="width: 19%;">
  <img src="https://raw.githubusercontent.com/assistive-autonomy/assistax/main/docs/imgs/task_feeding.png" alt="Feeding" style="width: 19%;">
  <img src="https://raw.githubusercontent.com/assistive-autonomy/assistax/main/docs/imgs/task_tooth_brushing.png" alt="Teeth Brushing" style="width: 19%;">
  <img src="https://raw.githubusercontent.com/assistive-autonomy/assistax/main/docs/imgs/task_arm_assist.png" alt="Arm Assist" style="width: 19%;">
</div>

A population of pre-trained partner policies for [**Assistax**](https://github.com/assistive-autonomy/assistax), a multi-agent hardware-accelerated reinforcement learning benchmark for assistive robotics. Each "human" in the zoo is a policy trained with its own sampled preferences over how it likes to be assisted — how fast the robot should move, how much contact force it should apply, how much unexpected touching it tolerates. Together they form the partner pool for zero-shot coordination (ZSC) and ad-hoc teamwork (AHT) experiments, where a robot has to work with partners it has never trained against.

| | |
|---|---|
| 📄 **Paper** | [Assistax: A Multi-Agent Hardware-Accelerated Reinforcement Learning Benchmark for Assistive Robotics](https://arxiv.org/abs/2507.21638) (Reinforcement Learning Journal, 2026) |
| 💻 **Code** | [assistive-autonomy/assistax](https://github.com/assistive-autonomy/assistax) |
| ⚖️ **License** | Apache 2.0 |
| 📦 **Contents** | `zoo.tar.gz` (~728 MB) |

## 👥 The population

- **630 unique humans per task**, split evenly across three training algorithms: 210 **IPPO**, 210 **MAPPO**, 210 **MASAC**.
- Agents are saved in **teams**. A team is one robot and one human trained together in the same run; both rows share a `team_uuid`. Across all five tasks that is **3,150 human+robot teams, 6,300 agents in total**.
- Five tasks are covered: `scratchitch`, `bedbathing`, `armmanipulation`, `feeding`, `teethbrushing`.
- The `pushcoop` and `handover` environments are robot–robot and have no preference rewards, so they have no zoo entries.

Humans within a task are distinguished by their preference weights `w_speed`, `w_force` and `w_touch`, which are sampled per agent — the population is deduplicated on those weights (plus disability settings), so no two humans in a task share an identity.

## ⬇️ Download

The archive is around 728 MB, so give it a moment.

```bash
hf download leohink/assistax-zoo zoo.tar.gz --repo-type dataset --local-dir .
tar -xzf zoo.tar.gz
```

Or without the Hugging Face CLI:

```bash
wget https://huggingface.co/datasets/leohink/assistax-zoo/resolve/main/zoo.tar.gz
tar -xzf zoo.tar.gz
```

Then point Assistax at the extracted `zoo/` directory by setting `ZOO_PATH` in `assistax/baselines/ZSC/config/ppo_aht.yaml` (or `sac_aht.yaml`, or `crossplay_zoo.yaml`), which defaults to `./zoo`:

```yaml
ZOO_PATH: /path/to/zoo
```

## 🗂️ Layout

```
zoo/
  index.csv                        # one row per agent
  config/<agent_uuid>.yaml         # the full training config for that agent
  params/<agent_uuid>.safetensors  # flax parameters, flattened with sep='/'
```

Every agent is identified by a UUID. Its parameters and the config it was trained under live under that same UUID, which is how the library rebuilds the right network for a given checkpoint.

### `index.csv` columns

| Column | Meaning |
|---|---|
| `agent_uuid` | Unique id; keys into `config/` and `params/` |
| `scenario` | Environment name, e.g. `scratchitch` |
| `scenario_agent_id` | `robot` or `human` |
| `algorithm` | `IPPO`, `MAPPO` or `MASAC` |
| `is_rnn` | Whether the policy is recurrent |
| `rnn_dim` | GRU hidden size when recurrent, else `0` |
| `team_uuid` | Shared by the robot and human trained together |
| `w_speed` | Weight on the human's speed preference |
| `w_force` | Weight on the human's contact-force preference |
| `w_touch` | Weight on the touch penalty (always negative) |

## 🙋 What makes each human different

Humans are differentiated by a preference reward that sits on top of the task reward:

- **Speed preference** — the human has a preferred band for the robot's end-effector speed and is happiest inside it, with Gaussian falloff outside. Band bounds are sampled from roughly 0.03–0.20 m/s.
- **Force preference** — the same idea for contact force on the body, with bounds sampled from roughly 1.0–5.0 N.
- **Action efficiency** — smaller, smoother robot actions score higher.
- **Touch penalty** — a negative weight applied whenever the robot makes a new, unexpected contact.

The positive components are budget-normalised (`reward_budget = 1.5` by default), so the maximum preference reward per step is the same for every human regardless of how their individual weights came out — what changes is *which behaviour* earns it. Full formulas, meta-ranges and sampling mechanics are in [`assistax/envs/README.md`](https://github.com/assistive-autonomy/assistax/blob/main/assistax/envs/README.md).

One thing to watch when loading these policies: agents trained with preference rewards observe **7 extra preference dimensions** on top of the base observation, so their first layer is wider than a vanilla policy for the same environment.

## 🧑‍💻 Loading a policy

The library handles the index, the architecture reconstruction and the parameter loading. It is pip-installable straight from GitHub:

```bash
# NVIDIA GPU (CUDA 12); use [cuda13] or [cpu] as appropriate
pip install "assistax[cuda12] @ git+https://github.com/assistive-autonomy/assistax.git"
```


```python
from assistax.wrappers.aht import ZooManager

zoo = ZooManager("/path/to/zoo")

# Pick the humans for one task
humans = zoo.index[
    (zoo.index.scenario == "scratchitch")
    & (zoo.index.scenario_agent_id == "human")
]
print(len(humans), "humans available")

agent = zoo.load_agent(humans.iloc[0].agent_uuid)
# agent.apply_fn         -> flax apply function
# agent.params           -> parameter pytree
# agent.hstate_reset_fn  -> initial hidden state (None for feed-forward policies)
```

`ZooManager` reads `config/<uuid>.yaml` to decide which network to rebuild — `IPPOActorCritic`, `IPPOActorCriticRNN`, `MAPPOActor`, `MAPPOActorRNN` or `SACActor`. The default `ff_nps` network is a feed-forward actor and critic with 128-unit hidden layers, ReLU activations and no parameter sharing between agents.

If you would rather not depend on Assistax, the parameters are plain safetensors:

```python
import safetensors.flax
from flax.traverse_util import unflatten_dict

flat = safetensors.flax.load_file("zoo/params/<agent_uuid>.safetensors")
params = unflatten_dict(flat, sep="/")
```

## 👯 Using the zoo for ZSC / AHT

With `ZOO_PATH` set, train a PPO robot against the pre-trained partner population:

```bash
uv run python assistax/baselines/ZSC/ppo_aht.py ENV_NAME=scratchitch
```

There is a SAC variant too (`assistax/baselines/ZSC/sac_aht.py`).

The population is split into train and test partners. By default this is a random split controlled by `SPLIT_RATIO` in the AHT config (`0.25` in `ppo_aht.yaml`, `0.2` in `sac_aht.yaml` — `0.25` means 25% train, 75% test). Setting `EXTREME_SPLIT` instead selects partners by how extreme their preferences are, in one of three modes — `single` (rank on one preference dimension), `multi` (rank on several independently) or `composite` (distance from the population centroid) — which is useful for testing generalisation to partners well outside the training distribution.

You can also run crossplay across the whole population to inspect behavioural diversity:

```bash
uv run python assistax/baselines/ZSC/crossplay_zoo.py ENV_NAME=scratchitch
```

## 🏝️ Tasks

| Task | Description |
|---|---|
| **Scratch Itch** (`scratchitch`) | An itch target is sampled on the human's right arm. The robot must reach it and apply a specified force; the human can move their arm to help. |
| **Bed Bathing** (`bedbathing`) | 52 bathing points are distributed along the human's arm. The robot must wipe each one with sufficient contact force before the episode ends. |
| **Arm Assist** (`armmanipulation`) | The human is too weak to lift their arm alone. The robot must align with a section of the arm and move it into a comfortable resting position. |
| **Feeding** (`feeding`) | The robot navigates a spoon to the mouth of a human seated in a wheelchair, keeping the spoon correctly oriented, moving at the right speed and making gentle contact. |
| **Teeth Brushing** (`teethbrushing`) | The robot approaches the mouth with a toothbrush, aligns the bristles and brushes with an appropriate tangential speed and contact force. |

All tasks are two-agent, built on Brax/MJX with continuous `Box(-1, 1)` actions and 1000-step episodes.

## 📜 License

Apache 2.0, matching the Assistax library.

## 📚 Citation

If you use the Assistax zoo in your work, please cite:

```bibtex
@article{hinckeldey2026assistax,
    title={Assistax: A Multi-Agent Hardware-Accelerated Reinforcement Learning Benchmark for Assistive Robotics},
    author={Leonard Hinckeldey and Elliot Fosong and Rimvydas Rubavicius and Elle Miller and Trevor McInroe and Fan Zhang and Patricia Wollstadt and Stefano V. Albrecht and Subramanian Ramamoorthy},
    journal={Reinforcement Learning Journal},
    volume={7},
    pages={},
    year={2026}
}
```
