"""The v2 population's training configs (population/config.py): each algorithm's zoo-gen config plus the paper's
per-task settings (paper_baselines/settings.json), the population budgets, the task's reward budget and the block's
disability, with the old population scripts' errors fixed."""
import json
import unittest
from pathlib import Path

from population import config as pc

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "paper_baselines/settings.json").read_text())["overrides"]
B = {"scratchitch": 2.0, "bedbathing": 1.0, "armmanipulation": 7.52, "feeding": 3.11, "teethbrushing": 2.09}


class TestCompose(unittest.TestCase):
    def test_ippo_scratching(self) -> None:
        c = pc.compose("scratchitch", "ippo", 0.5, 1.0)
        self.assertEqual((c["ENV_NAME"], c["ALGORITHM"]), ("scratchitch", "IPPO"))
        for k, v in {"LR": 0.000334, "UPDATE_EPOCHS": 8, "NUM_MINIBATCHES": 16, "CLIP_EPS": 0.16875672,
                     "ENT_COEF": 0.0016069901, "NUM_STEPS": 64, "TOTAL_TIMESTEPS": 4e7, "NUM_ENVS": 1024,
                     "NUM_SEEDS": 1}.items():
            self.assertEqual(c[k], v, k)
        pref = c["ENV_KWARGS"]["preference_rewards"]
        self.assertEqual((pref["reward_budget"], pref["overall_weight"], pref["touch_threshold"]), (2.0, 1, 0.3))
        self.assertEqual(c["ENV_KWARGS"]["disability"], {"joint_idx": 13, "joint_restriction_factor": 0.5,
                                                        "joint_strength": 1.0, "tremor_magnitude": 0.0})
        self.assertEqual((c["network"]["recurrent"], c["network"]["agent_param_sharing"]), (False, False))

    def test_old_script_bugs_fixed(self) -> None:
        self.assertEqual(pc.compose("armmanipulation", "ippo", 1.0, 1.0)["NUM_MINIBATCHES"], 8)
        self.assertEqual(pc.compose("feeding", "mappo", 1.0, 1.0)["ENT_COEF"], 0.000794748)
        bb = pc.compose("bedbathing", "mappo", 1.0, 1.0)
        self.assertEqual((bb["LR"], bb["NUM_MINIBATCHES"], bb["ENT_COEF"]), (0.00112, 8, 0.0029010084))
        self.assertEqual(pc.compose("feeding", "masac", 1.0, 1.0)["BATCH_SIZE"], 128)
        self.assertEqual(pc.compose("scratchitch", "masac", 1.0, 1.0)["BATCH_SIZE"], 512)
        aa = pc.compose("armmanipulation", "masac", 1.0, 1.0)
        self.assertEqual(aa["ALPHA_LR"], 0.000198)
        self.assertNotIn("ALPHA_LEARNING_RATE", aa)

    def test_budgets(self) -> None:
        m, s = pc.compose("feeding", "mappo", 1.0, 1.0), pc.compose("feeding", "masac", 1.0, 1.0)
        self.assertEqual((m["TOTAL_TIMESTEPS"], m["NUM_ENVS"]), (3e7, 1024))
        self.assertEqual((s["TOTAL_TIMESTEPS"], s["NUM_ENVS"]), (1e7, 256))

    def test_every_settings_override_applied(self) -> None:
        for alg, tasks in SETTINGS.items():
            for task, overrides in tasks.items():
                c = pc.compose(task, alg, 1.0, 1.0)
                for o in overrides:
                    key, val = o.removeprefix("++").split("=")
                    if key not in ("NUM_SEEDS", "TOTAL_TIMESTEPS"):
                        self.assertEqual(c[key], float(val) if "." in val else int(val), (alg, task, key))

    def test_reward_budget_for_every_algorithm(self) -> None:
        for task, b in B.items():
            for alg in ("ippo", "mappo", "masac"):
                self.assertEqual(pc.compose(task, alg, 1.0, 1.0)["ENV_KWARGS"]["preference_rewards"]["reward_budget"],
                                 b, (task, alg))

    def test_extra_overrides_apply_last(self) -> None:
        c = pc.compose("scratchitch", "ippo", 1.0, 1.0, ["++TOTAL_TIMESTEPS=1024", "++NUM_ENVS=8"])
        self.assertEqual((c["TOTAL_TIMESTEPS"], c["NUM_ENVS"]), (1024, 8))


if __name__ == "__main__":
    unittest.main()
