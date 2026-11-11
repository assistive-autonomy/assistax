"""Video of one episode from an audit_run.py recording: scene camera | close-up, the itch drawn where it is (scratch
itch), and the human-driven joint angles and the tool-target distance printed under each frame.

Usage: python -m reward_design.audit_video <run dir> <scene xml> <mode> <seed> <episode> <out.mp4>
"""
import sys
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from reward_design import render
from reward_design.human_audit import TOOL_TARGET, run_model


def main() -> None:
    run, scene, mode, seed, ep, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    m, human_act = run_model(run, scene)
    for name in ("target-u", "target-l"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            m.geom_rgba[gid, 3] = 0.0
    rec = dict(np.load(run / f"audit_{mode}.npz"))
    qpos, done = rec["qpos"][seed][:, ep], rec["done"][seed][:, ep]
    n = int(np.argmax(done)) if done.any() else len(done)
    dt = 0.004 if "armmanip" in scene else 0.008
    stride = max(1, int(np.ceil(1.0 / dt / 30)))
    joints = [m.actuator_trnid[a, 0] for a in human_act]
    tool = m.site(TOOL_TARGET[scene][0]).id
    d = mujoco.MjData(m)

    def itch(t: int):
        if "itch_arm" not in rec:
            return None
        g = m.geom("right_larm" if rec["itch_arm"][seed, t, ep] else "right_uarm1").id
        return d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ rec["itch_pos"][seed, t, ep]

    d.qpos[:] = qpos[0]
    mujoco.mj_kinematics(m, d)
    focus = itch(0) if "itch_arm" in rec else d.xpos[m.body("right_lower_arm").id]
    fwd = -d.cam_xmat[m.camera("default").id].reshape(3, 3)[:, 2]
    front = d.xmat[m.body("torso").id].reshape(3, 3)[:, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance = focus, 0.7
    cam.azimuth = np.degrees(np.arctan2(-front[1], -front[0])) + 240.0 if "itch_arm" in rec else np.degrees(np.arctan2(fwd[1], fwd[0]))
    cam.elevation = -35.0
    ren = mujoco.Renderer(m, height=render.PANEL_H, width=render.PANEL_W)
    frames = []
    for t in range(0, n, stride):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        x = itch(t)
        panels = []
        for c in ("default", cam):
            ren.update_scene(d, camera=c)
            if x is not None:
                render.add_sphere(ren.scene, x, 0.012, (1, 0, 0, 1))
            panels.append(ren.render())
        target = x if x is not None else None
        dist = f" | tool–itch {100 * np.linalg.norm(d.site_xpos[tool] - target):4.1f} cm" if target is not None else ""
        angles = "  ".join(f"{m.joint(j).name.replace('right_', '')} {np.degrees(d.qpos[m.jnt_qposadr[j]]):+4.0f}°" for j in joints)
        img = Image.fromarray(np.concatenate([np.concatenate(panels, 1), np.zeros((render.BAR, 2 * render.PANEL_W, 3), np.uint8)], 0))
        ImageDraw.Draw(img).text((10, render.PANEL_H + 6), f"t={t * dt:4.2f}s  human: {angles}{dist}", fill=(255, 255, 255),
                                 font=render.FONT)
        frames.append(np.asarray(img))
    ren.close()
    media.write_video(out, np.array(frames), fps=1.0 / (dt * stride), crf=23, codec="h264")
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main()
