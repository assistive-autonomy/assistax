# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Assistax is a hardware-accelerated reinforcement learning benchmark for assistive robotics using JAX and Brax. It provides multi-agent environments where robots assist humans with tasks like scratching, bed bathing, and arm manipulation.

## Imporant Rules

Do not make changes to environments unless given explicit permission by the user. This is espeically relevant to the reward function of the environments, these are completely off limits unless explicit premission by the user is given to make changes. 

## Common Commands

### Training and Execution
```bash
# Run IPPO baseline training
uv run python assistax/baselines/IPPO/ippo_run.py ENV_NAME=scratchitch

# Generate partner policy zoo
uv run python assistax/baselines/IPPO/ippo_zoo_gen.py ENV_NAME=scratchitch

# Train for Zero-Shot Coordination (ZSC)
uv run python assistax/baselines/ZSC/ppo_aht.py ENV_NAME=scratchitch

# Run crossplay evaluation
uv run python assistax/baselines/crossplay_zoo.py ENV_NAME=scratchitch

# Run hyperparameter sweeps
uv run python assistax/baselines/IPPO/ippo_sweep.py ENV_NAME=scratchitch

# Test preference rewards wrapper
uv run python assistax/baselines/IPPO/preference_rewards_test.py
```

### Installation and Environment
```bash
# Install with CUDA support
uv sync && uv pip install -e ".[cuda]"

# Install CPU-only version
uv sync && uv pip install -e ".[cpu]"
```

## Architecture Overview

### Environment Structure
- **assistax/envs/**: Core environment implementations
  - `scratchitch.py`: Robot scratches target on human's arm
  - `bedbathing.py`: Robot wipes bathing points on human's body
  - `armmanipulation.py`: Robot helps lift human's weak arm
  - `pushcoop.py`: Cooperative pushing task
  - `base_env.py`: Base environment class with common functionality

### Algorithm Implementations
- **assistax/baselines/**: Multi-agent RL algorithm implementations
  - **IPPO/**: Independent PPO with feedforward/recurrent networks, parameter sharing variants
  - **MAPPO/**: Multi-agent PPO with centralized critic
  - **ISAC/MASAC/**: Independent and Multi-agent Soft Actor-Critic
  - **ZSC/**: Zero-Shot Coordination algorithms for training with partner populations

### Environment Creation Pipeline
Environments are created through `assistax.envs.create()` which applies wrappers in sequence:
1. Base environment (e.g., ScratchItch)
2. EpisodeWrapper (episode length, action repeat)  
3. VmapWrapper (batching)
4. AutoResetWrapper (automatic episode reset)
5. DisabilityWrapper (human impairment simulation)
6. PreferenceRewardWrapper (human preference modeling)

### Configuration System
- Uses Hydra for configuration management
- Config files in `config/` directories with network-specific variants
- Main configs: `ippo.yaml`, `mappo.yaml`, `isac.yaml`, `masac.yaml`
- Sweep configs for hyperparameter optimization
- ZSC configs for population-based training

## Key Implementation Details

### Multi-Agent Setup
- **Agent 0**: Robot (action space varies by environment)
- **Agent 1**: Human (can be passive or active participant)
- Heterogeneous rewards supported via `het_reward` flag
- Parameter sharing vs non-parameter sharing variants available

### JAX/Brax Integration
- All environments built on Brax physics engine with MJX backend
- Hardware acceleration via JAX compilation and vectorization
- Disable JIT compilation with `DISABLE_JIT: True` for debugging

### Rendering and Evaluation
- HTML renders generated in `outputs/` directory
- Evaluation results saved as `.npy` files
- Model parameters saved as `.safetensors` files
- Hydra manages output directory structure with timestamps

### Performance Optimization
- GPU environment capacity configured via `GPU_ENV_CAPACITY`
- Advantage computation unrolling depth via `ADVANTAGE_UNROLL_DEPTH`
- Set `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` for GPU performance
- Use JAX memory allocation flags for memory issues

### Testing
- `preference_rewards_test.py`: Test preference reward wrapper functionality
- No comprehensive test suite - individual algorithm test files only
