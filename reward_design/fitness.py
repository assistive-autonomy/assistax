"""Episode fitness and binary task success for the five human-robot tasks.

Computed from the per-step diagnostics that ``tasks.py`` writes to ``metrics`` (``x_*``), over each evaluation
episode's steps before its first done. Fitness is a number in [0, 1]; success needs enough fitness and a safe
episode. The thresholds are reward-design choices, collected in ``SPEC`` so they are easy to see and change.
"""
import numpy as np

# ok:     what x_ok means per step (documentation only)
# fit:    "frac" = fraction of steps with x_ok, "gentle_wiped" = gently wiped targets at the end / 52
# need:   fitness needed for success
# f_safe: robot-human normal force (sum over contacts, N) that 99% of steps must stay under
SPEC = {
    "teethbrushing": dict(fit="frac", need=0.25, f_safe=10.0,
                          ok="bristles on the face within 2.5 cm of the mouth, 0.3-5 N, moving 0.5-10 cm/s"),
    "feeding": dict(fit="frac", need=0.10, f_safe=10.0,
                    ok="spoon tip within 2.5 cm of the mouth, aimed at it, < 5 cm/s, < 5 N on the head"),
    "scratchitch": dict(fit="frac", need=0.10, f_safe=15.0,
                        ok="scratcher on the arm within 2.5 cm of the itch, 0.1-10 N, moving 1-25 cm/s"),
    "bedbathing": dict(fit="gentle_wiped", need=0.80, f_safe=30.0,
                       ok="wiper pad on the arm with 0.5-20 N; a target counts when wiped at <= 20 N"),
    "armmanipulation": dict(fit="frac", need=0.25, f_safe=60.0, ok="arm point within 5 cm of its target"),
}
MAX_ROOT_DISP, MAX_ROOT_TILT = 0.10, np.radians(20)  # the human stays seated / lying: < 10 cm and < 20 deg


def first_episode_mask(done: np.ndarray) -> np.ndarray:
    """(T, E) mask of each episode's steps before its first done (the step that ends it is excluded)."""
    return np.cumsum(done, axis=0) == 0


def episode_table(env: str, metrics: dict, done: np.ndarray) -> dict:
    """Per-episode numbers, each of shape (E,), from (T, E) per-step metrics of one seed."""
    spec, mask = SPEC[env], first_episode_mask(done)
    mask[0] |= ~mask.any(0)  # an episode that ends at step 0 keeps one step
    n = mask.sum(0)

    def mean(k):
        return (metrics[k] * mask).sum(0) / n

    def pct(k, q):
        return np.array([np.percentile(metrics[k][mask[:, e], e], q) for e in range(mask.shape[1])])

    def last(k):
        return metrics[k][n - 1, np.arange(mask.shape[1])]

    fitness = mean("x_ok") if spec["fit"] == "frac" else last("x_n_gentle") / 52.0
    f99 = pct("x_f_rh", 99)
    disp = np.array([metrics["x_root_disp"][mask[:, e], e].max() for e in range(mask.shape[1])])
    tilt = np.array([metrics["x_root_tilt"][mask[:, e], e].max() for e in range(mask.shape[1])])
    safe = (f99 <= spec["f_safe"]) & (disp <= MAX_ROOT_DISP) & (tilt <= MAX_ROOT_TILT)
    out = {
        "fitness": fitness, "success": (fitness >= spec["need"]) & safe, "safe": safe, "length": n,
        "f_rh_p99": f99, "f_rh_max": pct("x_f_rh_max", 100), "contact_frac": ((metrics["x_f_rh"] > 0) * mask).sum(0) / n,
        "f_tool_p50_in_contact": np.array([np.median(metrics["x_f_tool"][mask[:, e] & (metrics["x_f_tool"][:, e] > 0), e])
                                           if ((metrics["x_f_tool"][:, e] > 0) & mask[:, e]).any() else 0.0
                                           for e in range(mask.shape[1])]),
        "f_other_mean": mean("x_f_other"), "task_d_p50": pct("x_task_d", 50), "task_d_min": pct("x_task_d", 0),
        "human_act": mean("x_human_act"), "robot_act": mean("x_robot_act"), "speed_p50": pct("x_speed", 50),
        "root_disp_max": disp, "root_tilt_max_deg": np.degrees(tilt),
        "torso_lean_p50_deg": np.degrees(pct("x_torso_lean", 50)),
    }
    if "x_neck_w" in metrics:
        out["neck_w_p99"] = pct("x_neck_w", 99)
    if "x_n_wiped" in metrics:
        out["wiped"], out["gentle_wiped"] = last("x_n_wiped"), last("x_n_gentle")
    return out


def summarise(table: dict) -> dict:
    """Mean over episodes of every per-episode number (success and safe as rates)."""
    return {k: float(np.mean(v)) for k, v in table.items()}
