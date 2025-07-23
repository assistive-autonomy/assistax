# 🦾 Assistax 

<div style="display: flex; justify-content: space-between;">
  <img src="docs/imgs/scratch.jpeg" alt="Scratching" style="width: 32%;">
  <img src="docs/imgs/bedbath.jpeg" alt="Scratching" style="width: 32%;">
  <img src="docs/imgs/armassist.jpeg" alt="Bedbathing" style="width: 32%;">
</div>

Assistax is a Python library that provides hardware-accelerated environments in the domain of
assistive robotics together with accompanying baseline algorithm implementation. We utilize jax and brax for quick RL and MARL training pipelines. 

## 🏄 Installation 

We use `uv` for environment and package management. We highly recommend to use `uv` when working with this project for installing `uv` see [`uv` installation](docs.astral.sh/uv/getting-started/installation/) and for more infromation and further documentation see [`uv` docs](https://docs.astral.sh/uv/)

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

Outputs are saved a new directory which is created inside the algorithm specific directory e.g. `assistax/baselines/IPPO/outputs`. This will contain 1. **results** as a `.npy` file, 2. **renders** as `.html` files 3. **parameters** of trained models as `.safetensors`. 

### 🎓 Generating multiple partner policies 

```bash
uv run python assistax/baselines/IPPO/ippo_zoo_gen.py ENV_NAME=scratchitch
```

Will create a `zoo` directory where configs and parameters used during training are stored. You can add pre-trained partners trained by multiple different algorithms to this `zoo` directory. 

### 👯 Training for ZSC


```bash
uv run python assistax/baselines/IPPO/ippo_aht.py ENV_NAME=scratchitch
```

This will run a ZSC experiment for a single ppo robot agent against the pre-trained partner policies in zoo. Check config `{alg}_aht.yaml`. By default will do 50-50 trin test split of the pre-trained partner agent population. 

### ⚔️ Crossplay of agent population 

```bash
uv run python assistax/baselines/crossplay_zoo.py ENV_NAME=scratchitch
```

This will generate crossplay returns which can be used to create crossplay-matrices to check for the diversity amongst the populations learnt policies. Note that this currently will not generate any renders. 

### 🧹 Sweeps

```bash
uv run python assistax/baselines/IPPO/ippo_sweep.py ENV_NAME=scratchitch
```

This will generate a sweep for the specified IPPO varient for the scratchitch task. You can run larger sweeps by utilizing hydras multirun feature see more below. 

### 🥱 Other information 

- We use hydra for managing configuration and training runs for more information see the [hydra docs](https://hydra.cc/docs/intro/)

## 🏝️ Environments 

- Scratch: A scratching target is randomly sampled on the surface of the human’s right
arm. The robot must move its end-effector to this position and apply a specified force. The human
can move its arm to make the target more accessible to the robot. [implementation](assistax/envs/scratchitch.py)
- Bed Bath: We provide target bath points distributed along the surface of the human’s
arm. The robot must reach each point and apply a certain force to activate the next point. The aim
is to reach (’wipe’) all points before the end of an episode. [implementation](assistax/envs/bedbathing.py)
- Arm Assist: The robot must help the human lift its right arm back into a comfortable
position on the bed. In this task the human is too weak to complete the task on their own and thus
requires the robot. The robot has to learn to align its end-effector with a target section of the arm
(shown in green on Figure 1(c)), and then move the human arm until the green and blue targets
overlap. [implementation](assistax/envs/armmanipulation.py)

## 📈 Baselines 

TODO add a discription or a table of the baselines we use here 

## Related 

TODO add related repositories here 

## Citations 

TODO add citation for the paper here 

## Other TODOS 

-[] Add the license Apache 2.0?
-[] Add link to Arxiv once uploaded. 


