"""How much does the human act, and does the shared reward give it a reason to? Reads audit_run.py recordings.

For one run (audit_sampled/mean/human_zero.npz, seed axis first) it reports:
  shared reward   largest |robot reward - human reward| over all steps (0 = one shared reward)
  command         the human's applied controls (actions clipped to the actuator range): effort = mean |ctrl| as a
                  share of the range, pinned = share of steps whose raw action is past the range (so the control
                  sits at its limit), change = mean over episodes of the control's std over time, as a share
  motion          per human-driven joint: median over episodes of its range (deg) and mean speed (deg/s)
  gap share       share of the tool-target approach made by the target's side (the human's body) rather than the
                  tool, summed over each episode; the same with the human zeroed shows what is passive or pushed
  counterfactual  return and median tool-target distance, sampled vs human zeroed
  itch (scratch)  the human's mean elbow angle for upper- vs lower-arm itches
and a capability test with plain MuJoCo and no policy: each human actuator at full effort, either way, for 2 s.

Usage: python -m reward_design.human_audit <run dir> <scene xml> [--json out.json]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

TOOL_TARGET = {  # scene -> (tool site, target: site name, or "itch", or ("geom", name))
    "teethbrushing_scene.xml": ("toothbrush_center", "mouth"),
    "feeding_scene.xml": ("spoon_center", "mouth"),
    "wheelchair_scene.xml": ("scratcher_point", "itch"),
    "bed_scene.xml": ("wiper_centre", ("geom", "right_larm")),
    "bed_scene_armmanip.xml": ("platform_center", "hook_target"),
}
HUMAN_ACT = {"teethbrushing_scene.xml": [0, 1], "feeding_scene.xml": [0, 1]}  # others: [11, 12, 13]


def first_mask(done: np.ndarray) -> np.ndarray:
    """(T, E) steps before each episode's first done."""
    m = np.cumsum(done, axis=0) == 0
    m[0] |= ~m.any(0)
    return m


def positions(m: mujoco.MjModel, qpos: np.ndarray, rec: dict, scene: str, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Tool and target world positions (T, E, 3) for seed s, by forward kinematics."""
    tool_name, tgt = TOOL_TARGET[scene]
    tool_id = m.site(tool_name).id
    T, E = qpos.shape[:2]
    tool, target = np.zeros((T, E, 3)), np.zeros((T, E, 3))
    d = mujoco.MjData(m)
    for e in range(E):
        for t in range(T):
            d.qpos[:] = qpos[t, e]
            mujoco.mj_kinematics(m, d)
            tool[t, e] = d.site_xpos[tool_id]
            if tgt == "itch":
                g = m.geom("right_larm" if rec["itch_arm"][s, t, e] else "right_uarm1").id
                target[t, e] = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ rec["itch_pos"][s, t, e]
            elif isinstance(tgt, tuple):
                target[t, e] = d.geom_xpos[m.geom(tgt[1]).id]
            else:
                target[t, e] = d.site_xpos[m.site(tgt).id]
    return tool, target


def gap_share(tool: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Share of the summed tool-target approach made by the target's motion (positive = toward the tool)."""
    u = tool[:-1] - target[:-1]
    u /= np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9
    robot = -np.sum((tool[1:] - tool[:-1]) * u, -1)
    human = np.sum((target[1:] - target[:-1]) * u, -1)
    mk = mask[1:]
    r, h = (robot * mk).sum(), (human * mk).sum()
    return float(h / (r + h)) if (r + h) > 1e-6 else float("nan")


def itch_side(m: mujoco.MjModel, rec: dict) -> dict:
    """Scratch itch: per episode, which way the itch faces at step 0 (world z of the arm's surface normal there),
    the median and smallest tip-itch distance after the first 2 s (every 10th step), and the human's mean
    shoulder and elbow angles over that time; grouped into itches facing up (z > 0.3), sideways, and down."""
    d = mujoco.MjData(m)
    tip = m.site("scratcher_point").id
    adr = {n: m.jnt_qposadr[m.joint(n).id] for n in ("right_shoulder1", "right_shoulder2", "right_elbow")}
    rows = []
    for s in range(rec["qpos"].shape[0]):
        for e in range(rec["qpos"].shape[2]):
            dists, angles = [], []
            for t in range(0, rec["qpos"].shape[1], 10):
                d.qpos[:] = rec["qpos"][s, t, e]
                mujoco.mj_kinematics(m, d)
                g = m.geom("right_larm" if rec["itch_arm"][s, t, e] else "right_uarm1").id
                rot, p = d.geom_xmat[g].reshape(3, 3), rec["itch_pos"][s, t, e]
                if t == 0:
                    z = (rot @ np.array([p[0], p[1], 0.0]) / (np.linalg.norm(p[:2]) + 1e-9))[2]
                elif t >= 250:
                    dists.append(np.linalg.norm(d.site_xpos[tip] - (d.geom_xpos[g] + rot @ p)))
                    angles.append([np.degrees(d.qpos[a]) for a in adr.values()])
            rows.append((z, np.median(dists), np.min(dists), *np.mean(angles, 0)))
    rows = np.array(rows)
    out = {}
    for name, sel in (("up", rows[:, 0] > 0.3), ("sideways", np.abs(rows[:, 0]) <= 0.3), ("down", rows[:, 0] < -0.3)):
        r = rows[sel]
        out[name] = {"n": int(sel.sum()), "tip_itch_cm": float(100 * np.median(r[:, 1])),
                     "closest_cm": float(100 * np.median(r[:, 2])),
                     **{f"{k.replace('right_', '')}_deg": float(np.mean(r[:, 3 + i])) for i, k in enumerate(adr)}}
    return out


def capability(m: mujoco.MjModel, acts: list[int], seconds: float = 2.0) -> dict:
    """Each human actuator at full effort, + then -, from the keyframe, the robot holding its pose: joint excursion."""
    out = {}
    robot = [i for i in range(m.nu) if m.actuator(i).name.startswith("actuator")]
    for a in acts:
        j = m.actuator_trnid[a, 0]
        adr = m.jnt_qposadr[j]
        ext = []
        for sign in (1.0, -1.0):
            d = mujoco.MjData(m)
            d.qpos[:] = m.keyframe("init").qpos
            mujoco.mj_forward(m, d)
            q0 = d.qpos[adr]
            for r in robot:  # hold the robot where it is: zero affine force at the current angle
                jr = m.jnt_qposadr[m.actuator_trnid[r, 0]]
                d.ctrl[r] = -m.actuator_biasprm[r, 1] * d.qpos[jr] / m.actuator_gainprm[r, 0]
            d.ctrl[a] = sign * m.actuator_ctrlrange[a, 1]
            qs = []
            for _ in range(int(seconds / m.opt.timestep)):
                mujoco.mj_step(m, d)
                qs.append(d.qpos[adr])
            ext.append(float(np.degrees((max(qs) - q0) if sign > 0 else (min(qs) - q0))))
        lo, hi = np.degrees(m.jnt_range[j])
        out[m.joint(j).name] = {"gear": float(m.actuator_gear[a, 0]), "max_torque_Nm": float(m.actuator_gear[a, 0] * m.actuator_ctrlrange[a, 1]),
                                "moved_plus_deg": ext[0], "moved_minus_deg": ext[1], "range_deg": [float(lo), float(hi)]}
    return out


def audit(run: Path, scene: str) -> dict:
    """All audit numbers for one run directory."""
    m = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "assistax/envs/assets" / scene))
    acts = HUMAN_ACT.get(scene, [11, 12, 13])
    lim = m.actuator_ctrlrange[acts, 1]
    recs = {mode: dict(np.load(run / f"audit_{mode}.npz")) for mode in ("sampled", "mean", "human_zero")}
    agents = list(recs["sampled"]["agents"])
    h, r = agents.index("human"), agents.index("robot")
    out = {"run": str(run), "shared_reward_max_diff": float(np.abs(recs["sampled"]["reward"][:, :, r]
                                                                   - recs["sampled"]["reward"][:, :, h]).max())}
    for mode in ("sampled", "mean"):
        rec = recs[mode]
        raw = rec["action"][:, :, h, :, :len(acts)]  # (S, T, E, k)
        ctrl = np.clip(raw, -lim, lim)
        mask = np.stack([first_mask(rec["done"][s]) for s in range(raw.shape[0])])[..., None]
        out[f"{mode}_effort"] = float((np.abs(ctrl) / lim * mask).sum() / (mask.sum() * len(acts)))
        out[f"{mode}_pinned"] = float(((np.abs(raw) >= lim) * mask).sum() / (mask.sum() * len(acts)))
        out[f"{mode}_change"] = float(np.mean(np.std(ctrl / lim, axis=1)))
        out[f"{mode}_raw_mean_abs"] = float(np.mean(np.abs(rec["mean"][:, :, h, :, :len(acts)])))
    joints = {}
    rec = recs["mean"]
    for a in acts:
        j = m.actuator_trnid[a, 0]
        q = np.degrees(rec["qpos"][..., m.jnt_qposadr[j]])  # (S, T, E)
        v = np.abs(np.degrees(rec["qvel"][..., m.jnt_dofadr[j]]))
        span = [np.ptp(q[s, first_mask(rec["done"][s])[:, e], e]) for s in range(q.shape[0]) for e in range(q.shape[2])]
        joints[m.joint(j).name] = {"range_deg": float(np.median(span)), "speed_deg_s": float(v.mean())}
    out["joints"] = joints
    for mode in ("sampled", "mean", "human_zero"):
        rec = recs[mode]
        shares, dists = [], []
        for s in range(rec["qpos"].shape[0]):
            mask = first_mask(rec["done"][s])
            tool, target = positions(m, rec["qpos"][s], rec, scene, s)
            shares.append(gap_share(tool, target, mask))
            dists.append(np.median(np.linalg.norm(tool - target, axis=-1)[mask]))
        out[f"{mode}_gap_share"] = float(np.nanmean(shares))
        out[f"{mode}_tool_target_cm"] = float(100 * np.mean(dists))
    for mode in ("sampled", "human_zero"):
        rec = recs[mode]
        ret = [(rec["reward"][s, :, r] * first_mask(rec["done"][s])).sum(0).mean() for s in range(rec["reward"].shape[0])]
        out[f"{mode}_return"] = float(np.mean(ret))
    if "itch_arm" in recs["mean"]:
        rec = recs["mean"]
        elbow = np.degrees(rec["qpos"][..., m.jnt_qposadr[m.joint("right_elbow").id]])[:, 200:].mean(1)  # (S, E)
        lower = rec["itch_arm"][:, 0].astype(bool)
        out["elbow_deg_lower_arm_itch"] = float(elbow[lower].mean())
        out["elbow_deg_upper_arm_itch"] = float(elbow[~lower].mean())
        out["by_itch_side_mean"] = itch_side(m, recs["mean"])
        out["by_itch_side_sampled"] = itch_side(m, recs["sampled"])
    out["capability"] = capability(m, acts)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("scene")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    res = audit(a.run, a.scene)
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
