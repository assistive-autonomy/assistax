"""Instrumented assistax human-robot envs for reward design.

Each class subclasses one assistax env. Version "v0" keeps the env exactly as it is (dynamics, observations,
reward) and only adds per-step diagnostics to ``state.metrics`` under ``x_*`` keys: contact normal forces,
distances and a per-step task flag ``x_ok`` that ``fitness.py`` turns into episode fitness and success.

Later versions replace the reward (``_reward_<version>``) and, for feeding and teeth brushing, give the human its
neck instead of its abdomen and a passive neck spring (``install`` patches the action map; ``__init__`` the model).
Nothing in the assistax package is modified: ``install`` swaps the env class in the registry for this process.
"""
from functools import lru_cache, partial

import mujoco
import numpy as np
from brax import math as bmath
from brax.envs.base import State
from jax import numpy as jp

import assistax.envs as aenvs
from assistax.envs import base_env
from assistax.envs.armmanipulation import ArmManipulation
from assistax.envs.bedbathing import BedBathing
from assistax.envs.contact_utils import contact_geom_pairs
from assistax.envs.feeding import Feeding
from assistax.envs.scratchitch import ScratchItch
from assistax.envs.teethbrushing import TeethBrushing

HUMAN, ROBOT = "humanoid_container", "panda_container"
NECK_HUMAN_ACT = [17, 18]  # neck_yaw, neck_pitch actuators (feeding, teeth brushing)
NECK_STIFFNESS, NECK_DAMPING = 10.0, 2.0  # N m/rad, N m s/rad: a head that holds itself up (v1+)
# Weight of the out-of-range-action cost per version. 0.1 left the round-2 bed-bathing policy at |a| ~ 2.2: its
# joint targets sit at their limits, which is what makes contact forces spike, so v4 raises it to 1.
BOUND_W = {"v1": 0.1, "v2": 0.1, "v3": 0.1, "v4": 1.0, "v5": 1.0}
# Action-rate cost weight per version: base + near * gate, where gate rises linearly from 0 at GATE_RANGE metres
# from the task target to 1 at it. v1 applied 0.5 everywhere: random early actions then cost ~1.6 per step, far
# more than the approach reward pays, and the teeth-brushing robot learned to stay still 56 cm from the mouth.
SMOOTH_W = {"v1": (0.5, 0.0), "v2": (0.02, 0.5), "v3": (0.02, 0.5), "v4": (0.02, 0.5), "v5": (0.02, 0.5)}
GATE_RANGE = 0.2


@lru_cache(maxsize=None)
def _layout(xml_path: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """(contact slot -> geom pair ids, geom -> top-level body name) for a scene, cached per process."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    top = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b and m.body_parentid[b] != 0:
            b = m.body_parentid[b]
        top.append(m.body(b).name)
    return contact_geom_pairs(m), tuple(top)


def normal_forces(ps, ids: np.ndarray) -> jp.ndarray:
    """Normal force (N) in each listed contact slot, 0 when inactive (pyramidal cones: sum of the pyramid rows)."""
    c = ps._impl.contact
    addr, dim = np.asarray(c.efc_address)[ids], np.asarray(c.dim)[ids]
    rows = np.where(dim == 1, 1, 2 * (dim - 1))
    width = int(rows.max())
    idx = addr[:, None] + np.arange(width)[None]
    mask = (np.arange(width)[None] < rows[:, None]) & (addr[:, None] >= 0)
    return (ps._impl.efc_force[np.where(mask, idx, 0)] * mask).sum(-1)


def peak1(x: jp.ndarray) -> jp.ndarray:
    """x * exp(1 - x): 0 at 0, peaks at 1 when x = 1, then decays."""
    return x * jp.exp(1.0 - x)


def zeros(keys) -> dict:
    """Float32 zero scalars for metric keys (reset values)."""
    return {k: jp.zeros(()) for k in keys}


class Diag:
    """Mixin: robot-human contact slots, human bookkeeping and the common diagnostics."""

    COMMON = ("x_ok", "x_task_d", "x_f_tool", "x_f_rh", "x_f_rh_max", "x_f_other", "x_human_act", "x_robot_act",
              "x_root_disp", "x_root_tilt", "x_torso_lean", "x_speed")
    EXTRA: tuple = ()

    def _rd_setup(self, version: str, tool_slots: list[int], human_act: list[int], robot_act: list[int]) -> None:
        """Contact groups and human root reference, computed once from the scene."""
        self.version = version
        pairs, top = _layout(str(self.path))
        self.pairs = pairs
        self.rh = np.array([i for i, (a, b) in enumerate(pairs) if {top[a], top[b]} == {HUMAN, ROBOT}])
        self.tool_in_rh = np.array([int(np.flatnonzero(self.rh == s)[0]) for s in tool_slots])
        self.other_in_rh = np.array([i for i, s in enumerate(self.rh) if s not in tool_slots])
        self.human_act, self.robot_act = np.array(human_act), np.array(robot_act)
        m = self.sys.mj_model
        # The human's free body is the torso; the pelvis (a child) is what sits on the chair or lies on the bed.
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso").id
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("init").qpos
        mujoco.mj_kinematics(m, d)
        self.pelvis_pos0 = d.xpos[self.pelvis].copy()
        self.pelvis_z0 = d.xmat[self.pelvis].reshape(3, 3)[:, 2].copy()
        self.torso_z0 = d.xmat[self.torso].reshape(3, 3)[:, 2].copy()

    def _common(self, ps, action: jp.ndarray, f: jp.ndarray) -> dict:
        """Diagnostics shared by every task; f = normal forces of the robot-human slots. Root = the pelvis
        (displacement from the keyframe pose and tilt); x_torso_lean = the torso's tilt from the keyframe pose."""
        def tilt(body, z0):
            return jp.arccos(jp.clip(jp.dot(ps.xmat[body][:, 2], z0), -1.0, 1.0))
        return {
            "x_f_tool": f[self.tool_in_rh].sum(),
            "x_f_rh": f.sum(),
            "x_f_rh_max": f.max(),
            "x_f_other": f[self.other_in_rh].sum(),
            "x_human_act": jp.abs(action[self.human_act]).mean(),
            "x_robot_act": jp.abs(action[self.robot_act]).mean(),
            "x_root_disp": jp.linalg.norm(ps.xpos[self.pelvis] - self.pelvis_pos0),
            "x_root_tilt": tilt(self.pelvis, self.pelvis_z0),
            "x_torso_lean": tilt(self.torso, self.torso_z0),
        }

    def _neck_setup(self) -> None:
        """v1+: a passive neck spring and damper, so the head holds itself up when the human does not act."""
        m = self.sys.mj_model
        jids = [m.joint(n).id for n in ("neck_z", "neck_y")]
        dofs = [int(m.jnt_dofadr[j]) for j in jids]
        stiff = np.asarray(self.sys.jnt_stiffness).copy()
        damp = np.asarray(self.sys.dof_damping).copy()
        stiff[jids], damp[dofs] = NECK_STIFFNESS, NECK_DAMPING
        # jax (not numpy) arrays: MJX treats numpy model fields as static and groups joints by their values
        self.sys = self.sys.tree_replace({"jnt_stiffness": jp.asarray(stiff), "dof_damping": jp.asarray(damp)})

    GATE_KEY = "x_task_d"  # diagnostic distance (m) to the task target that gates the action-rate cost

    def reset(self, rng):
        state = super().reset(rng)
        keys = self.COMMON + self.EXTRA + (("p_smooth", "p_bound", "w_smooth") if self.version != "v0" else ())
        return state.replace(metrics={**state.metrics, **zeros(keys)}, info=self._info0(state.info))

    def _info0(self, info: dict) -> dict:
        """Extra info entries (v1+: the previous action, for the action-rate penalty)."""
        return info if self.version == "v0" else {**info, "prev_act": jp.zeros(self.sys.nu)}

    def _pre_step(self, state: State, first: jp.ndarray) -> State:
        """v1+ hook before the env step (bed bathing resets its wipe state at an episode's first step)."""
        return state

    def step(self, rng, state: State, action: jp.ndarray) -> State:
        ps0 = state.pipeline_state
        first = state.info["steps"] == 0 if "steps" in state.info else jp.zeros((), bool)
        if self.version != "v0":
            state = self._pre_step(state, first)
        info0 = dict(state.info)
        new = super().step(rng, state, action)
        diag = self._diag(ps0, new, action)
        reward, done, terms, info = new.reward, new.done, {}, new.info
        if self.version != "v0":
            smooth, bound = self._smooth(info0, action, first), self._bound(action)
            reward, done, terms = getattr(self, f"_reward_{self.version}")(info0, new, diag, smooth)
            base, near = SMOOTH_W[self.version]
            w_smooth = base + near * jp.clip(1.0 - diag[self.GATE_KEY] / GATE_RANGE, 0.0, 1.0)
            reward = reward - w_smooth * smooth - BOUND_W[self.version] * bound
            terms = {**terms, "p_smooth": smooth, "p_bound": bound, "w_smooth": w_smooth}
            info = {**new.info, "prev_act": self._ctrl(action)}
        return new.replace(reward=reward, done=done, info=info, metrics={**new.metrics, **diag, **terms})

    def _ctrl(self, action: jp.ndarray) -> jp.ndarray:
        """The controls MuJoCo applies: actions clipped to each actuator's control range."""
        rng = np.asarray(self.sys.mj_model.actuator_ctrlrange)
        return jp.clip(action, rng[:, 0], rng[:, 1])

    def _smooth(self, info0: dict, action: jp.ndarray, first: jp.ndarray) -> jp.ndarray:
        """Action-rate cost: mean squared step-to-step change of each agent's applied controls, in units of half the
        actuator's control range (0 at an episode's first step). Penalises jitter without biasing the posture."""
        rng = np.asarray(self.sys.mj_model.actuator_ctrlrange)
        da = jp.where(first, 0.0, (self._ctrl(action) - info0["prev_act"]) / (0.5 * (rng[:, 1] - rng[:, 0])))
        return jp.mean(da[self.robot_act] ** 2) + jp.mean(da[self.human_act] ** 2)

    def _bound(self, action: jp.ndarray) -> jp.ndarray:
        """How far the agents' raw actions leave the [-1, 1] action space (mean over actuated dims). Clipped actions
        give PPO no signal, so policies drift to saturated, bang-bang means without this."""
        idx = np.concatenate([self.robot_act, self.human_act])
        return jp.mean(jp.maximum(jp.abs(action[idx]) - 1.0, 0.0))

    @staticmethod
    def _over(force: jp.ndarray, comfort: float) -> jp.ndarray:
        """Overload: force above `comfort`, in units of `comfort`, capped at 4."""
        return jp.minimum(jp.maximum(force - comfort, 0.0) / comfort, 4.0)


def _neck_dofs(m: mujoco.MjModel) -> np.ndarray:
    return np.array([int(m.jnt_dofadr[m.joint(n).id]) for n in ("neck_z", "neck_y")])


class RDTeeth(Diag, TeethBrushing):
    """Teeth brushing. x_ok: the bristle side touches the face within 2.5 cm of the mouth with 0.3-5 N normal force
    while moving 0.5-10 cm/s along the face (relative to the head)."""

    EXTRA = ("x_f_bristle", "x_f_head_other", "x_contact_mouth_d", "x_v_tan", "x_neck_w", "x_align")
    REWARD_KEYS = ("r_far", "r_mid", "r_near", "r_force", "r_brush", "r_align", "p_over", "p_harm", "p_other", "p_smooth",
                   "p_bound", "p_neck")

    def __init__(self, version: str = "v0", **kwargs):
        super().__init__(**kwargs)
        m = self.sys.mj_model
        self.head_geom = m.geom("head").id
        self.neck_dofs = _neck_dofs(m)
        human = [0, 1] if version == "v0" else NECK_HUMAN_ACT
        self._rd_setup(version, [self.TOOTHBRUSH_RSIDE_CONTACT_ID], human, list(range(19, 26)))
        head_slots = [s for s in self.rh if self.head_geom in self.pairs[s]]
        self.head_other_in_rh = np.array([int(np.flatnonzero(self.rh == s)[0]) for s in head_slots
                                          if s != self.TOOTHBRUSH_RSIDE_CONTACT_ID])
        if version != "v0":
            self._neck_setup()
            self.EXTRA = self.EXTRA + self.REWARD_KEYS

    def _reward_v1(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """Bristles on the mouth with ~1 N, slow strokes along the face, nothing else pressing on the head."""
        near = (d["x_contact_mouth_d"] < 0.025).astype(jp.float32)
        touching = (d["x_f_bristle"] > 0.05).astype(jp.float32)
        t = {"r_far": jp.exp(-3.0 * d["x_task_d"]), "r_near": jp.exp(-d["x_task_d"] / 0.01),
             "r_force": peak1(d["x_f_bristle"] / 1.0) * near, "r_brush": peak1(d["x_v_tan"] / 0.03) * touching * near,
             "p_over": self._over(d["x_f_bristle"] + d["x_f_head_other"], 5.0),
             "p_other": jp.minimum(d["x_f_other"], 4.0), "p_neck": d["x_neck_w"] ** 2}
        t["r_align"] = d["x_align"] * t["r_near"]
        reward = (1.0 * t["r_far"] + 2.0 * t["r_near"] + 3.0 * t["r_force"] + 3.0 * t["r_brush"] + 1.0 * t["r_align"]
                  - 1.0 * t["p_over"] - 0.5 * t["p_other"] - 0.05 * t["p_neck"])
        return reward, new.done, t

    def _reward_v2(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v1 with v0's approach weight (2 exp(-3 d)) restored; the action-rate cost is gated by distance (SMOOTH_W)."""
        reward, done, t = self._reward_v1(info0, new, d, smooth)
        return reward + 1.0 * t["r_far"], done, t

    def _reward_v3(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v2 policies point the brush tip at the mouth from ~3.5 cm: the bristle-alignment term was weighted by the
        1 cm near term, so it paid nothing there. v3 weights alignment by the far term (as v0 does), adds a 3 cm-scale
        distance term, and penalises head speed 4x more (v2's human swings its head at up to 2.5 rad/s)."""
        reward, done, t = self._reward_v2(info0, new, d, smooth)
        align_far = d["x_align"] * t["r_far"]
        reward = reward - 1.0 * t["r_align"] + 1.0 * align_far
        t = {**t, "r_align": align_far, "r_mid": jp.exp(-d["x_task_d"] / 0.03)}
        return reward + 2.0 * t["r_mid"] - 0.15 * t["p_neck"], done, t

    _reward_v4 = _reward_v3  # v4 = v3's reward with the stronger out-of-range-action cost (BOUND_W)

    def _reward_v5(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v3 policies line the bristles up but stop ~6 cm out: contact only paid near exactly 1 N and any other
        touch cost a flat 2, so with a noisy policy touching the face lost reward. v5 pays bristle contact on the
        mouth at full credit up to 5 N and 5/F above (at least 1/4), sets the head's comfort force to 15 N, charges
        5 for steps above 60 N, and charges incidental contact by force (0.05 per N, at most 1)."""
        f = d["x_f_bristle"]
        on_mouth = ((d["x_contact_mouth_d"] < 0.025) & (f > 0.05)).astype(jp.float32)
        quality = jp.where(f <= 5.0, 1.0, jp.maximum(0.25, 5.0 / jp.maximum(f, 1e-6)))
        r_far = jp.exp(-3.0 * d["x_task_d"])
        t = {"r_far": r_far, "r_mid": jp.exp(-d["x_task_d"] / 0.03), "r_near": jp.exp(-d["x_task_d"] / 0.01),
             "r_align": d["x_align"] * r_far, "r_force": on_mouth * quality,
             "r_brush": peak1(d["x_v_tan"] / 0.03) * on_mouth * quality,
             "p_over": self._over(f + d["x_f_head_other"], 15.0), "p_harm": (d["x_f_rh"] > 60.0).astype(jp.float32),
             "p_other": jp.minimum(d["x_f_other"] / 10.0, 2.0), "p_neck": d["x_neck_w"] ** 2}
        reward = (2.0 * t["r_far"] + 2.0 * t["r_mid"] + 2.0 * t["r_near"] + 1.0 * t["r_align"] + 4.0 * t["r_force"]
                  + 3.0 * t["r_brush"] - 1.0 * t["p_over"] - 5.0 * t["p_harm"] - 0.5 * t["p_other"] - 0.15 * t["p_neck"])
        return reward, new.done, t

    def _diag(self, ps0, new: State, action) -> dict:
        ps = new.pipeline_state
        f = normal_forces(ps, self.rh)
        mouth, tb = ps.site_xpos[self.human_mouth], ps.site_xpos[self.panda_toothbrush_centre]
        f_b = f[self.tool_in_rh].sum()
        cpos = ps._impl.contact.pos[self.TOOTHBRUSH_RSIDE_CONTACT_ID]
        d_cm = jp.where(f_b > 0, jp.linalg.norm(cpos - mouth), 1.0)
        head, head0 = ps.geom_xpos[self.head_geom], ps0.geom_xpos[self.head_geom]
        v_rel = (tb - ps0.site_xpos[self.panda_toothbrush_centre] - (head - head0)) / self.dt
        n = (cpos - head) / (jp.linalg.norm(cpos - head) + 1e-6)
        v_tan = jp.linalg.norm(v_rel - jp.dot(v_rel, n) * n)
        bristle = bmath.rotate(jp.array([-1.0, 0.0, 0.0]), ps.xquat[self.panda_toothbrush_body_idx])
        to_mouth = (mouth - tb) / (jp.linalg.norm(mouth - tb) + 1e-6)
        ok = (f_b >= 0.3) & (f_b <= 5.0) & (d_cm < 0.025) & (v_tan >= 0.005) & (v_tan <= 0.10)
        return {**self._common(ps, action, f), "x_ok": ok.astype(jp.float32), "x_task_d": jp.linalg.norm(tb - mouth),
                "x_speed": jp.linalg.norm(v_rel), "x_f_bristle": f_b, "x_f_head_other": f[self.head_other_in_rh].sum(),
                "x_contact_mouth_d": d_cm, "x_v_tan": v_tan, "x_neck_w": jp.linalg.norm(ps.qvel[self.neck_dofs]),
                "x_align": jp.dot(bristle, to_mouth)}


class RDFeeding(Diag, Feeding):
    """Feeding. x_ok: the spoon tip is within 2.5 cm of the mouth, pointing at it (cos > 0.7), moving < 5 cm/s
    relative to the head, with < 5 N on the head."""

    EXTRA = ("x_tip_d", "x_aim", "x_level", "x_f_head", "x_neck_w")
    REWARD_KEYS = ("r_far", "r_mid", "r_near", "r_touch", "r_shape", "p_over", "p_other", "p_smooth", "p_bound", "p_neck")
    TIP = 0.036  # spoon bowl half-length along the spoon's z axis

    def _reward_v1(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """Spoon tip brought to the mouth slowly and held there (v0's orientation and speed shaping kept), a 3-per-step
        bonus while x_ok, a gentle touch, and nothing pressing on the head."""
        m = new.metrics
        t = {"r_far": m["reward_dist"], "r_near": jp.exp(-d["x_tip_d"] / 0.01), "r_touch": peak1(d["x_f_tool"] / 1.0),
             "p_over": self._over(d["x_f_head"], 5.0), "p_other": jp.minimum(d["x_f_other"], 4.0),
             "p_neck": d["x_neck_w"] ** 2}
        reward = (1.0 * t["r_far"] + 2.0 * t["r_near"] + 1.0 * m["reward_orientation"] + 1.0 * m["reward_velocity"]
                  + 3.0 * d["x_ok"] + 1.0 * t["r_touch"] - 1.0 * t["p_over"] - 0.5 * t["p_other"]
                  - 0.05 * t["p_neck"])
        return reward, new.done, t

    def _reward_v2(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v1 with v0's approach weight (2 exp(-3 d)) restored; the action-rate cost is gated by distance (SMOOTH_W)."""
        reward, done, t = self._reward_v1(info0, new, d, smooth)
        return reward + 1.0 * t["r_far"], done, t

    def _reward_v4(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v2 robots carry the spoon to 1.1 m from the mouth: v0's orientation and speed terms pay ~1.6 per step at
        any distance, and v2 charges for brushing past the human's arm on the way in. v4 scales those two terms by
        exp(-3 d) (so they pay only near the mouth), adds a 3 cm-scale tip term, raises the head's comfort force to
        10 N and charges incidental contact by force (0.05 per N, at most 1) instead of a flat 2 above 4 N."""
        m = new.metrics
        near_mouth = m["reward_dist"]  # exp(-3 d), d = spoon centre to mouth
        t = {"r_far": near_mouth, "r_mid": jp.exp(-d["x_tip_d"] / 0.03), "r_near": jp.exp(-d["x_tip_d"] / 0.01),
             "r_touch": peak1(d["x_f_tool"] / 1.0), "r_shape": near_mouth * (m["reward_orientation"] + m["reward_velocity"]),
             "p_over": self._over(d["x_f_head"], 10.0), "p_other": jp.minimum(d["x_f_other"] / 10.0, 2.0),
             "p_neck": d["x_neck_w"] ** 2}
        reward = (2.0 * t["r_far"] + 2.0 * t["r_mid"] + 2.0 * t["r_near"] + 1.0 * t["r_shape"] + 3.0 * d["x_ok"]
                  + 1.0 * t["r_touch"] - 1.0 * t["p_over"] - 0.5 * t["p_other"] - 0.05 * t["p_neck"])
        return reward, new.done, t

    _reward_v5 = _reward_v4  # v5 changes teeth brushing and the arm-manipulation human's arm gear only

    def __init__(self, version: str = "v0", **kwargs):
        super().__init__(**kwargs)
        m = self.sys.mj_model
        self.head_geom = m.geom("head").id
        self.neck_dofs = _neck_dofs(m)
        human = [0, 1] if version == "v0" else NECK_HUMAN_ACT
        self._rd_setup(version, [self.SPOON_HEAD_CONTACT_ID, self.SPOON_RSIDE_CONTACT_ID], human, list(range(19, 26)))
        self.head_in_rh = np.array([i for i, s in enumerate(self.rh) if self.head_geom in self.pairs[s]])
        if version != "v0":
            self._neck_setup()
            self.EXTRA = self.EXTRA + self.REWARD_KEYS

    def _diag(self, ps0, new: State, action) -> dict:
        ps = new.pipeline_state
        f = normal_forces(ps, self.rh)
        mouth, sc = ps.site_xpos[self.human_mouth], ps.site_xpos[self.panda_spoon_centre]
        rot = ps.xmat[self.panda_spoon_body_idx]
        tip = sc + self.TIP * rot[:, 2]
        d = jp.linalg.norm(sc - mouth)
        aim = jp.dot(rot[:, 2], (mouth - sc) / (d + 1e-6))
        level = jp.dot(rot @ jp.array([-1.0, 0.0, 0.0]), jp.array([0.0, 0.0, 1.0]))
        head, head0 = ps.geom_xpos[self.head_geom], ps0.geom_xpos[self.head_geom]
        speed = jp.linalg.norm((sc - ps0.site_xpos[self.panda_spoon_centre] - (head - head0)) / self.dt)
        f_head = f[self.head_in_rh].sum()
        tip_d = jp.linalg.norm(tip - mouth)
        ok = (tip_d < 0.025) & (aim > 0.7) & (speed < 0.05) & (f_head < 5.0)
        return {**self._common(ps, action, f), "x_ok": ok.astype(jp.float32), "x_task_d": d, "x_speed": speed,
                "x_tip_d": tip_d, "x_aim": aim, "x_level": level, "x_f_head": f_head,
                "x_neck_w": jp.linalg.norm(ps.qvel[self.neck_dofs])}


class RDScratch(Diag, ScratchItch):
    """Scratch itch. x_ok: the scratcher stick touches the arm within 2.5 cm of the itch with 0.1-10 N normal force
    while moving 1-25 cm/s relative to the itch (Assistive Gym: < 10 N, within 2.5 cm, moving)."""

    EXTRA = ("x_contact_itch_d", "x_tip_itch_d")
    REWARD_KEYS = ("r_far", "r_mid", "r_near", "r_force", "r_scratch", "p_over", "p_harm", "p_other", "p_smooth",
                   "p_bound")

    def __init__(self, version: str = "v0", **kwargs):
        super().__init__(**kwargs)
        self._rd_setup(version, [self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID], [11, 12, 13], list(range(17, 24)))
        if version != "v0":
            self.EXTRA = self.EXTRA + self.REWARD_KEYS

    def _reward_v1(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """Distance shaping down to the last centimetre, ~3 N on the itch, strokes at ~5 cm/s, no hard pressing."""
        near = (d["x_contact_itch_d"] < 0.025).astype(jp.float32)
        touching = (d["x_f_tool"] > 0.1).astype(jp.float32)
        x = d["x_tip_itch_d"]
        t = {"r_far": jp.exp(-x ** 2 / 0.1), "r_mid": jp.exp(-x ** 2 / 0.01), "r_near": jp.exp(-x / 0.01),
             "r_force": peak1(d["x_f_tool"] / 3.0) * near, "r_scratch": peak1(d["x_speed"] / 0.05) * touching * near,
             "p_over": self._over(d["x_f_rh"], 10.0), "p_other": jp.minimum(d["x_f_other"], 4.0)}
        reward = (1.0 * t["r_far"] + 1.0 * t["r_mid"] + 2.0 * t["r_near"] + 3.0 * t["r_force"] + 3.0 * t["r_scratch"]
                  - 1.0 * t["p_over"] - 0.5 * t["p_other"])
        return reward, new.done, t

    _reward_v2 = _reward_v1  # v2 changes only the distance gating of the action-rate cost
    GATE_KEY = "x_tip_itch_d"

    def _reward_v4(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v2's terms with a 3 cm-scale distance term in place of the 10 cm Gaussian, a 15 N comfort force, incidental
        contact charged by force (0.05 per N, at most 1), and a 5-point charge for any step above 60 N on the human."""
        near = (d["x_contact_itch_d"] < 0.025).astype(jp.float32)
        touching = (d["x_f_tool"] > 0.1).astype(jp.float32)
        x = d["x_tip_itch_d"]
        t = {"r_far": jp.exp(-x ** 2 / 0.1), "r_mid": jp.exp(-x / 0.03), "r_near": jp.exp(-x / 0.01),
             "r_force": peak1(d["x_f_tool"] / 3.0) * near, "r_scratch": peak1(d["x_speed"] / 0.05) * touching * near,
             "p_over": self._over(d["x_f_rh"], 15.0), "p_harm": (d["x_f_rh"] > 60.0).astype(jp.float32),
             "p_other": jp.minimum(d["x_f_other"] / 10.0, 2.0)}
        reward = (1.0 * t["r_far"] + 2.0 * t["r_mid"] + 2.0 * t["r_near"] + 3.0 * t["r_force"] + 3.0 * t["r_scratch"]
                  - 1.0 * t["p_over"] - 5.0 * t["p_harm"] - 0.5 * t["p_other"])
        return reward, new.done, t

    _reward_v5 = _reward_v4  # v5 changes teeth brushing and the arm-manipulation human's arm gear only

    def _diag(self, ps0, new: State, action) -> dict:
        ps = new.pipeline_state
        f = normal_forces(ps, self.rh)
        itch, itch0 = self._get_scratch_xpos(ps, new.info), self._get_scratch_xpos(ps0, new.info)
        tip, tip0 = ps.site_xpos[self.panda_scratcher_tip_idx], ps0.site_xpos[self.panda_scratcher_tip_idx]
        fu, fl = f[self.tool_in_rh[0]], f[self.tool_in_rh[1]]
        pos = ps._impl.contact.pos
        cpos = jp.where(fu >= fl, pos[self.UARM_TOOL_CONTACT_ID], pos[self.LARM_TOOL_CONTACT_ID])
        f_s = fu + fl
        d_c = jp.where(f_s > 0, jp.linalg.norm(cpos - itch), 1.0)
        speed = jp.linalg.norm((tip - tip0 - (itch - itch0)) / self.dt)
        ok = (f_s >= 0.1) & (f_s <= 10.0) & (d_c < 0.025) & (speed >= 0.01) & (speed <= 0.25)
        return {**self._common(ps, action, f), "x_ok": ok.astype(jp.float32), "x_task_d": jp.linalg.norm(tip - itch),
                "x_speed": speed, "x_contact_itch_d": d_c, "x_tip_itch_d": jp.linalg.norm(tip - itch)}


class RDBedBathing(Diag, BedBathing):
    """Bed bathing. x_ok: the wiper pad touches the arm with 0.5-20 N. x_gentle counts targets wiped (the env's rule:
    wiper centre within 10 cm) at a step whose pad force is at most 20 N."""

    EXTRA = ("x_n_wiped", "x_n_gentle")
    REWARD_KEYS = ("r_dist", "r_wipe", "r_contact", "p_over", "p_other", "p_smooth", "p_bound")
    F_GENTLE = 20.0

    def __init__(self, version: str = "v0", **kwargs):
        super().__init__(**kwargs)
        tool = [self.UARM_TOOL_CONTACT_ID1, self.UARM_TOOL_CONTACT_ID2, self.LARM_TOOL_CONTACT_ID1,
                self.LARM_TOOL_CONTACT_ID2]
        self._rd_setup(version, tool, [11, 12, 13], list(range(17, 24)))
        if version != "v0":
            self.EXTRA = self.EXTRA + self.REWARD_KEYS

    def _info0(self, info: dict) -> dict:
        return {**super()._info0(info), "gentle_cv": jp.ones(self.n_targets)}

    def _pre_step(self, state: State, first: jp.ndarray) -> State:
        """Start each episode with nothing wiped. assistax's AutoResetWrapper resets the physics but not `info`, so
        after an episode that wiped everything the next one would start (and stay) done."""
        cv = jp.where(first, 1.0, state.info["contact_vector"])
        return state.replace(info={**state.info, "contact_vector": cv,
                                   "gentle_cv": jp.where(first, 1.0, state.info["gentle_cv"])})

    def _reward_v1(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """Wipe credit only for gentle wipes (<= 20 N), distance shaping to the nearest target not yet wiped gently,
        a small bonus for light pad contact, and penalties for pressing hard or touching with anything but the pad.

        No early termination: v0 ends the episode when every target is wiped, which forfeits the remaining steps'
        distance reward, so finishing is punished (the v0 policy never wipes all 52). Here, once everything is
        wiped the distance term pays its maximum until the episode's time limit."""
        ps, gcv = new.pipeline_state, new.info["gentle_cv"]
        dist = self._target_distances(self._get_global_targets(ps), ps.site_xpos[self.panda_wiper_center_idx])
        x = jp.where(jp.all(gcv == 0), 0.0, jp.min(jp.where(gcv == 0, jp.inf, dist)))
        t = {"r_dist": jp.exp(-x ** 2 / 0.1) + jp.exp(-x ** 2 / 0.01),
             "r_wipe": d["x_n_gentle"] - jp.sum(info0["gentle_cv"] == 0).astype(jp.float32),
             "r_contact": peak1(d["x_f_tool"] / 5.0), "p_over": self._over(d["x_f_rh"], self.F_GENTLE),
             "p_other": jp.minimum(d["x_f_other"], 4.0)}
        reward = t["r_dist"] + 3.0 * t["r_wipe"] + 0.5 * t["r_contact"] - 1.0 * t["p_over"] - 0.5 * t["p_other"]
        return reward, jp.zeros_like(new.done), t

    _reward_v2 = _reward_v1  # v2 changes only the distance gating of the action-rate cost

    def _reward_v4(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v2 robots hover 7-10 cm from the arm: hovering earned ~1.25 per step, and its noisy contacts rarely stayed
        under 20 N, so wipes rarely paid. v4 pays every wipe (the env's rule) at 5, scaled by the pad force (full up
        to 20 N, 20/F above, at least 1/4), pays light pad contact (peak at 10 N), and halves the hover term."""
        ps, cv = new.pipeline_state, new.info["contact_vector"]
        dist = self._target_distances(self._get_global_targets(ps), ps.site_xpos[self.panda_wiper_center_idx])
        x = jp.where(jp.all(cv == 0), 0.0, jp.min(jp.where(cv == 0, jp.inf, dist)))
        f = d["x_f_tool"]
        quality = jp.where(f <= self.F_GENTLE, 1.0, jp.maximum(0.25, self.F_GENTLE / jp.maximum(f, 1e-6)))
        new_wipes = d["x_n_wiped"] - jp.sum(info0["contact_vector"] == 0).astype(jp.float32)
        t = {"r_dist": 0.5 * jp.exp(-x ** 2 / 0.1), "r_wipe": new_wipes * quality, "r_contact": peak1(f / 10.0),
             "p_over": self._over(d["x_f_rh"], 40.0), "p_other": jp.minimum(d["x_f_other"] / 10.0, 2.0)}
        reward = t["r_dist"] + 5.0 * t["r_wipe"] + 1.5 * t["r_contact"] - 1.0 * t["p_over"] - 0.5 * t["p_other"]
        return reward, jp.zeros_like(new.done), t

    _reward_v5 = _reward_v4  # v5 changes teeth brushing and the arm-manipulation human's arm gear only

    def _diag(self, ps0, new: State, action) -> dict:
        ps = new.pipeline_state
        f = normal_forces(ps, self.rh)
        f_pad = f[self.tool_in_rh].sum()
        targets = self._get_global_targets(ps)
        dist = self._target_distances(targets, ps.site_xpos[self.panda_wiper_center_idx])
        gentle = (f_pad > 0) & (f_pad <= self.F_GENTLE)
        cv = jp.where((dist < self.target_threshold) & gentle, 0.0, new.info["gentle_cv"])
        new.info["gentle_cv"] = cv
        wc, wc0 = ps.site_xpos[self.panda_wiper_center_idx], ps0.site_xpos[self.panda_wiper_center_idx]
        ok = (f_pad >= 0.5) & (f_pad <= self.F_GENTLE)
        return {**self._common(ps, action, f), "x_ok": ok.astype(jp.float32),
                "x_task_d": jp.min(jp.where(new.info["contact_vector"] == 0, jp.inf, dist)),
                "x_speed": jp.linalg.norm(wc - wc0) / self.dt,
                "x_n_wiped": jp.sum(new.info["contact_vector"] == 0).astype(jp.float32),
                "x_n_gentle": jp.sum(cv == 0).astype(jp.float32)}


class RDArmManip(Diag, ArmManipulation):
    """Arm manipulation. x_ok: the arm point (hook_target site) is within 5 cm of its target (arm_target site)."""

    EXTRA = ("x_hook_d",)
    REWARD_KEYS = ("r_near", "r_support", "p_rot", "p_over", "p_harm", "p_other", "p_speed", "p_smooth", "p_bound")
    # The scene gives the human's right-arm actuators (its whole action space) gear 0, so the human agent has no
    # effect. From v5 (Leo, 2026-11-10): gear 10, i.e. up to 4 N m; lifting the arm off the bed takes ~6-7 N m, so
    # the human can move and help but the robot still has to carry the arm. The other scenes use gear 25.
    HUMAN_ARM_GEAR = {"v5": 10.0}

    def __init__(self, version: str = "v0", **kwargs):
        super().__init__(**kwargs)
        self._rd_setup(version, list(range(56, 64)), [11, 12, 13], list(range(17, 24)))
        if version != "v0":
            self.EXTRA = self.EXTRA + self.REWARD_KEYS
        if version in self.HUMAN_ARM_GEAR:
            gear = np.asarray(self.sys.actuator_gear).copy()
            gear[self.human_act, 0] = self.HUMAN_ARM_GEAR[version]
            self.sys = self.sys.tree_replace({"actuator_gear": jp.asarray(gear)})

    def _reward_v1(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v0's waist and hook terms, the rotation term as a penalty (v0 adds it, rewarding misalignment), a
        near-target term, a bonus while the hook carries the arm (3-40 N), and penalties for pressing hard, touching
        with anything but the hook, and moving the hook faster than 0.5 m/s (the v0 robot swings it at up to 3 m/s
        while the human lifts its own arm)."""
        m = new.metrics
        f = d["x_f_tool"]
        t = {"r_near": jp.exp(-d["x_task_d"] / 0.02), "r_support": ((f >= 3.0) & (f <= 40.0)).astype(jp.float32),
             "p_rot": m["reward_rot"], "p_over": self._over(d["x_f_rh"], 40.0),
             "p_other": jp.minimum(d["x_f_other"], 4.0), "p_speed": jp.maximum(d["x_speed"] - 0.5, 0.0)}
        reward = (10.0 * m["reward_waist_dist"] + 1.0 * m["reward_hook_dist"] + 5.0 * t["r_near"] + 2.0 * t["r_support"]
                  - 0.1 * t["p_rot"] - 1.0 * t["p_over"] - 0.5 * t["p_other"] - 1.0 * t["p_speed"])
        return reward, new.done, t

    _reward_v2 = _reward_v1  # v2 changes only the distance gating of the action-rate cost (gated on the hook)
    GATE_KEY = "x_hook_d"

    def _reward_v4(self, info0: dict, new: State, d: dict, smooth: jp.ndarray):
        """v2 robots never touch the arm and nobody moves it. v4 pays 5 per step while the hook carries the arm at
        2-60 N, charges 5 for any step above 240 N (v0 flings the arm with a 300 N hit), charges incidental contact
        by force, and only slows the hook above 1 m/s."""
        m, f = new.metrics, d["x_f_tool"]
        t = {"r_near": jp.exp(-d["x_task_d"] / 0.02), "r_support": ((f >= 2.0) & (f <= 60.0)).astype(jp.float32),
             "p_rot": m["reward_rot"], "p_over": self._over(d["x_f_rh"], 60.0),
             "p_harm": (d["x_f_rh"] > 240.0).astype(jp.float32), "p_other": jp.minimum(d["x_f_other"] / 10.0, 2.0),
             "p_speed": jp.maximum(d["x_speed"] - 1.0, 0.0)}
        reward = (10.0 * m["reward_waist_dist"] + 1.0 * m["reward_hook_dist"] + 5.0 * t["r_near"] + 5.0 * t["r_support"]
                  - 0.1 * t["p_rot"] - 1.0 * t["p_over"] - 5.0 * t["p_harm"] - 0.5 * t["p_other"] - 0.5 * t["p_speed"])
        return reward, new.done, t

    _reward_v5 = _reward_v4  # v5: same reward; the human's arm gets gear 10 (HUMAN_ARM_GEAR)

    def _diag(self, ps0, new: State, action) -> dict:
        ps = new.pipeline_state
        f = normal_forces(ps, self.rh)
        arm, target = ps.site_xpos[self.hook_target_site], ps.site_xpos[self.arm_target_site]
        hook, hook0 = ps.site_xpos[self.panda_hook_center_idx], ps0.site_xpos[self.panda_hook_center_idx]
        d = jp.linalg.norm(target - arm)
        return {**self._common(ps, action, f), "x_ok": (d < 0.05).astype(jp.float32), "x_task_d": d,
                "x_speed": jp.linalg.norm(hook - hook0) / self.dt, "x_hook_d": jp.linalg.norm(arm - hook)}


CLASSES = {"teethbrushing": RDTeeth, "feeding": RDFeeding, "scratchitch": RDScratch, "bedbathing": RDBedBathing,
           "armmanipulation": RDArmManip}


def install(env_name: str, version: str) -> None:
    """Make assistax.make(env_name) build the instrumented env at `version` (this process only)."""
    aenvs._envs[env_name] = partial(CLASSES[env_name], version=version)
    if env_name in ("feeding", "teethbrushing"):
        base_env._agent_action_mapping[env_name]["human"] = jp.array([0, 1] if version == "v0" else NECK_HUMAN_ACT)
