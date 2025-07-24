# 🦾 Assistax 

<div style="display: flex; justify-content: space-between;">
  <img src="docs/imgs/scratch.jpeg" alt="Scratching" style="width: 32%;">
  <img src="docs/imgs/bedbath.jpeg" alt="Scratching" style="width: 32%;">
  <img src="docs/imgs/armassist.jpeg" alt="Bedbathing" style="width: 32%;">
</div>

Assistax is a Python library that provides hardware-accelerated environments in the domain of assistive robotics together with accompanying baseline algorithm implementations. We utilize JAX and Brax for quick RL and MARL training pipelines.

## 🏄 Installation

We use `uv` for environment and package management. We highly recommend using `uv` when working with this project. For installing `uv`, see [uv installation](https://docs.astral.sh/uv/getting-started/installation/), and for more information and further documentation, see [uv docs](https://docs.astral.sh/uv/).

1. Clone the repository
```bash
git clone https://github.com/LeoHink/assistax.git
```

2. Install all packages with uv
```bash
cd assistax
uv sync && uv pip install -e .
```

## 🚀 Quick Start

### 🏋️‍♀️ Run one of the baselines

```bash
uv run python assistax/baselines/IPPO/ippo_run.py ENV_NAME=scratchitch
```

Outputs are saved to a new directory which is created inside the algorithm-specific directory (e.g., `assistax/baselines/IPPO/outputs`). This will contain: 1. **results** as a `.npy` file, 2. **renders** as `.html` files, 3. **parameters** of trained models as `.safetensors`.

### 🎓 Generating multiple partner policies

```bash
uv run python assistax/baselines/IPPO/ippo_zoo_gen.py ENV_NAME=scratchitch
```

This will create a `zoo` directory where configs and parameters used during training are stored. You can add pre-trained partners trained by multiple different algorithms to this `zoo` directory.

### 👯 Training for ZSC

```bash
uv run python assistax/baselines/IPPO/ippo_aht.py ENV_NAME=scratchitch
```

This will run a ZSC experiment for a single PPO robot agent against the pre-trained partner policies in the zoo. Check the config `{alg}_aht.yaml`. By default, this will do a 50-50 train-test split of the pre-trained partner agent population.

### ⚔️ Crossplay of agent population

```bash
uv run python assistax/baselines/crossplay_zoo.py ENV_NAME=scratchitch
```

This will generate crossplay returns which can be used to create crossplay matrices to check for diversity among the population's learned policies. Note that this currently will not generate any renders.

### 🧹 Sweeps

```bash
uv run python assistax/baselines/IPPO/ippo_sweep.py ENV_NAME=scratchitch
```

This will generate a sweep for the specified IPPO variant for the scratchitch task. You can run larger sweeps by utilizing Hydra's multirun feature (see the Hydra documentation for more details).

### 🥱 Other information

- We use Hydra for managing configuration and training runs. For more information, see the [Hydra docs](https://hydra.cc/docs/intro/).

## 🏝️ Environments

- **Scratch**: A scratching target is randomly sampled on the surface of the human's right arm. The robot must move its end-effector to this position and apply a specified force. The human can move their arm to make the target more accessible to the robot. [implementation](assistax/envs/scratchitch.py)

- **Bed Bath**: We provide target bathing points distributed along the surface of the human's arm. The robot must reach each point and apply a certain force to activate the next point. The aim is to reach (wipe) all points before the end of an episode. [implementation](assistax/envs/bedbathing.py)

- **Arm Assist**: The robot must help the human lift their right arm back into a comfortable position on the bed. In this task, the human is too weak to complete the task on their own and thus requires the robot. The robot has to learn to align its end-effector with a target section of the arm (shown in green in Figure 1(c)), and then move the human's arm until the green and blue targets overlap. [implementation](assistax/envs/armmanipulation.py)

## 📈 Baselines 

| Algorithm | FF | PS | NPS | RNN | ZSC |
|-----------|----|----|----|----|----|
| IPPO    | ✅ | ✅ | ✅ | ✅ | ✅  |
| MAPPO  | ✅ | ✅ | ✅ | ✅ | ✅  |
| ISAC   | ✅  | ❌ | ✅ | ❌ | ❌|
| MASAC  | ✅  | ❌ | ✅ | ❌ | ✅ |

## Related 

Some relavant repo's you should check out include:

TODO add relevant other repos 

## Citations 

TODO add citation for the paper here once uploaded on Arxiv

## TODOS 

- [ ] Homogonize the configs e.g. have `rl.gamma` instead of `GAMMA` 


