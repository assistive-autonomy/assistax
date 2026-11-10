"""Videos for reward-design runs: default camera | close-up, task markers drawn where they really are, and a
per-step text overlay of the numbers the fitness uses.

Markers: scratch itch = the episode's itch (red; the scene's two fixed red markers are hidden); bed bathing = the 52
wipe targets (red: not wiped, yellow: wiped by the env's rule but pressing > 20 N, green: wiped gently).
"""
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parents[1] / "assistax" / "envs" / "assets"
SCENES = {"teethbrushing": "teethbrushing_scene.xml", "feeding": "feeding_scene.xml",
          "scratchitch": "wheelchair_scene.xml", "bedbathing": "bed_scene.xml",
          "armmanipulation": "bed_scene_armmanip.xml"}
PANEL_W, PANEL_H, BAR = 640, 480, 34
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
except OSError:
    FONT = ImageFont.load_default()


def overlay_text(env: str, mt: dict, t: int, dt: float) -> str:
    """One line of the per-step numbers behind the task flag x_ok."""
    ok = "YES" if mt["x_ok"] > 0.5 else "no"
    head = f"t={t * dt:4.2f}s  "
    if env == "teethbrushing":
        return head + (f"brush-mouth {mt['x_task_d'] * 100:4.1f} cm | bristles {mt['x_f_bristle']:5.1f} N | "
                       f"rest of tool on head {mt['x_f_head_other']:5.1f} N | brushing: {ok}")
    if env == "feeding":
        return head + (f"spoon tip-mouth {mt['x_tip_d'] * 100:4.1f} cm | on head {mt['x_f_head']:5.1f} N | "
                       f"head turn {mt['x_neck_w']:3.1f} rad/s | at mouth: {ok}")
    if env == "scratchitch":
        return head + (f"tip-itch {mt['x_tip_itch_d'] * 100:4.1f} cm | scratcher {mt['x_f_tool']:5.1f} N | "
                       f"other robot contact {mt['x_f_other']:5.1f} N | scratching: {ok}")
    if env == "bedbathing":
        return head + (f"wiped {mt['x_n_wiped']:2.0f}/52 (gently {mt['x_n_gentle']:2.0f}) | pad {mt['x_f_tool']:5.1f} N | "
                       f"other contact {mt['x_f_other']:5.1f} N")
    return head + (f"arm-target {mt['x_task_d'] * 100:4.1f} cm | hook on body {mt['x_f_rh']:5.1f} N | "
                   f"hook speed {mt['x_speed']:4.2f} m/s | at target: {ok}")


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
    """Append a decorative sphere to an already-updated scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        pos.astype(np.float64), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _targets_local(m: mujoco.MjModel, geom: str, n: int = 26) -> np.ndarray:
    """Bed-bathing wipe targets in a capsule's geom frame, as BedBathing._initialize_targets makes them."""
    radius, half = m.geom_size[m.geom(geom).id][:2]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.linspace(-half, half, n)], -1)


def markers(env: str, m: mujoco.MjModel, d: mujoco.MjData, rec: dict, t: int, ep: int) -> list:
    """(pos, radius, rgba) of the task markers at step t."""
    if env == "scratchitch":
        arm = bool(rec["info"]["scratch"]["arm"][t, ep])
        g = m.geom("right_larm" if arm else "right_uarm1").id
        pos = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ rec["info"]["scratch"]["pos"][t, ep]
        return [(pos, 0.012, (1, 0, 0, 1))]
    if env == "bedbathing":
        out = []
        cv, gcv = rec["info"]["contact_vector"][t, ep], rec["info"]["gentle_cv"][t, ep]
        pts = []
        for geom in ("right_uarm", "right_larm"):
            g = m.geom(geom).id
            pts.append(_targets_local(m, geom) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        for i, p in enumerate(np.concatenate(pts)):
            rgba = (0, 0.9, 0, 1) if gcv[i] == 0 else (1, 0.85, 0, 1) if cv[i] == 0 else (1, 0, 0, 1)
            out.append((p, 0.007, rgba))
        return out
    return []


def render_episode(env: str, m: mujoco.MjModel, rec: dict, ep: int, n: int, stride: int, dt: float) -> np.ndarray:
    """Frames (T, H + BAR, 2 W, 3) of one episode."""
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=PANEL_H, width=PANEL_W)
    qpos, qvel = rec["qpos"][:, ep], rec["qvel"][:, ep]
    look = []
    for t in range(0, n, stride):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        mk = markers(env, m, d, rec, t, ep)
        if env in ("teethbrushing", "feeding"):
            look.append(d.site_xpos[m.site("mouth").id].copy())
        elif mk and env == "scratchitch":
            look.append(mk[0][0])
        else:
            look.append(d.xpos[m.body("hand").id].copy())
    fwd = -d.cam_xmat[m.camera("default").id].reshape(3, 3)[:, 2]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.mean(look, 0)
    cam.distance = 0.6 if env in ("teethbrushing", "feeding", "scratchitch") else 0.9
    cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0]))
    cam.elevation = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    if env in ("teethbrushing", "feeding"):  # face on: look back along the head's forward (+x) axis at step 0
        d.qpos[:] = qpos[0]
        mujoco.mj_kinematics(m, d)
        face = d.xmat[m.body("head").id].reshape(3, 3)[:, 0]
        cam.azimuth = np.degrees(np.arctan2(-face[1], -face[0])) + 25.0  # turned a little toward the robot
        cam.elevation = -10.0
    elif env == "scratchitch":  # from the human's right-front and above, where the robot's arm does not block it
        d.qpos[:] = qpos[0]
        mujoco.mj_kinematics(m, d)
        front = d.xmat[m.body("torso").id].reshape(3, 3)[:, 0]
        cam.azimuth = np.degrees(np.arctan2(-front[1], -front[0])) + 240.0
        cam.elevation, cam.distance = -35.0, 0.7
    metrics = {k: np.asarray(v)[:, ep] for k, v in rec["metrics"].items() if k.startswith("x_")}
    frames = []
    for t in range(0, n, stride):
        d.qpos[:], d.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_forward(m, d)
        panels = []
        for c in ("default", cam):
            ren.update_scene(d, camera=c)
            for pos, r, rgba in markers(env, m, d, rec, t, ep):
                add_sphere(ren.scene, pos, r, rgba)
            panels.append(ren.render())
        img = Image.fromarray(np.concatenate([np.concatenate(panels, 1), np.zeros((BAR, 2 * PANEL_W, 3), np.uint8)], 0))
        ImageDraw.Draw(img).text((10, PANEL_H + 6), overlay_text(env, {k: v[t] for k, v in metrics.items()}, t, dt),
                                 fill=(255, 255, 255), font=FONT)
        frames.append(np.asarray(img))
    ren.close()
    return np.array(frames)


def render_picks(env: str, version: str, out_dir: Path, rec: dict, picks: dict, dt: float) -> None:
    """Videos of the picked episodes of seed 0 (real time, <= 30 fps) and an 8-frame sheet of the median one."""
    m = mujoco.MjModel.from_xml_path(str(ASSETS / SCENES[env]))
    for name in ("target-u", "target-l"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            m.geom_rgba[gid, 3] = 0.0
    stride = max(1, int(np.ceil(1.0 / dt / 30)))
    (out_dir / "videos").mkdir(exist_ok=True)
    for which, ep in picks.items():
        done = np.asarray(rec["done"])[:, ep]
        n = int(np.argmax(done)) if done.any() else len(done)
        frames = render_episode(env, m, rec, ep, n, stride, dt)
        media.write_video(str(out_dir / "videos" / f"seed0_{which}.mp4"), frames, fps=1.0 / (dt * stride), crf=23,
                          codec="h264")
        if which == "median":
            idx = np.linspace(0, len(frames) - 1, 8).astype(int)
            rows = [np.concatenate(list(frames[idx[i:i + 2]]), axis=1) for i in range(0, 8, 2)]
            media.write_image(str(out_dir / "videos" / "seed0_median_sheet.png"), np.concatenate(rows, axis=0))
