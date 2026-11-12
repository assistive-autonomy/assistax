"""Training configs for the v2 population. Each is an algorithm's zoo-gen config
(assistax/baselines/<ALG>/config/<alg>_zoo_gen.yaml), composed by hydra as the zoo-gen scripts compose it, plus:
- the paper's per-(algorithm, task) settings (paper_baselines/settings.json, the paper's own launch overrides) minus
  NUM_SEEDS and TOTAL_TIMESTEPS. This fixes the old population scripts' errors: IPPO arm assist minibatches, MAPPO
  feeding entropy, MAPPO bed bathing blocks with arm assist's settings, MASAC batch sizes, and MASAC's no-op
  ALPHA_LEARNING_RATE (the key is ALPHA_LR);
- the population budgets (Leo, 2026-11-11): IPPO 4e7 steps x 1024 envs, MAPPO 3e7 x 1024, MASAC 1e7 x 256;
- the task's preference reward budget B (MAPPO's and MASAC's configs have none and would fall back to 1), and the
  block's disability on the right-shoulder actuator (joint_idx 13, no tremor).
"""
import json
from pathlib import Path
from typing import Sequence

from hydra import compose as hydra_compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "paper_baselines/settings.json").read_text())["overrides"]
# Reward budget B per task. Feeding 4.91 = 2/3 of v2's max reward per step 7.368 (the tight term), the rule behind the
# old scripts' arm assist, teeth and feeding values (Leo, 2026-11-12; was 3.11 = 2/3 of the paper's 4.7).
TASKS = {"scratchitch": 2.0, "bedbathing": 1.0, "armmanipulation": 7.52, "feeding": 4.91, "teethbrushing": 2.09}
BUDGETS = {"ippo": {"TOTAL_TIMESTEPS": "4e7", "NUM_ENVS": 1024}, "mappo": {"TOTAL_TIMESTEPS": "3e7", "NUM_ENVS": 1024},
           "masac": {"TOTAL_TIMESTEPS": "1e7", "NUM_ENVS": 256}}
DISABILITY_JOINT = 13


def overrides(task: str, alg: str, restriction: float, strength: float) -> list[str]:
    """Hydra overrides of one (task, algorithm, disability) cell, applied to <alg>_zoo_gen.yaml."""
    paper = [o for o in SETTINGS[alg][task] if o.split("=")[0] not in ("++NUM_SEEDS", "++TOTAL_TIMESTEPS")]
    pref, dis = "++ENV_KWARGS.preference_rewards", "++ENV_KWARGS.disability"
    return [f"++ENV_NAME={task}", "++NUM_SEEDS=1", *paper, *(f"++{k}={v}" for k, v in BUDGETS[alg].items()),
            f"{pref}.reward_budget={TASKS[task]}", f"{pref}.overall_weight=1",
            f"{dis}.joint_idx={DISABILITY_JOINT}", f"{dis}.joint_restriction_factor={restriction}",
            f"{dis}.joint_strength={strength}", f"{dis}.tremor_magnitude=0.0"]


def compose(task: str, alg: str, restriction: float, strength: float, extra: Sequence[str] = ()) -> dict:
    """The resolved training config of one cell, with `extra` overrides applied last."""
    with initialize_config_dir(str(ROOT / "assistax/baselines" / alg.upper() / "config"), version_base=None):
        cfg = hydra_compose(config_name=f"{alg}_zoo_gen",
                            overrides=[*overrides(task, alg, restriction, strength), *extra])
    return OmegaConf.to_container(cfg, resolve=True)
