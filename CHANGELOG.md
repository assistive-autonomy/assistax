# Changelog

## 0.1.0 (unreleased)

**Breaking: observations change for all five human–robot tasks and for handover.** Policies trained with earlier versions, including the published partner zoo, see different inputs under 0.1.0 and need retraining. The last pre-0.1.0 code is tagged `zoo-v1`.

### Fixed

- **Joint observations read the wrong `qpos`/`qvel` entries** (scratchitch, bedbathing, armmanipulation, feeding, teethbrushing).
  - **Cause:** joint ids were used as `qpos` addresses, but the human's root is a free joint (7 `qpos`, 6 dofs), so every later joint was shifted.
  - **Robot:** its "joint angles" were the human's six arm joints (plus Panda `joint1` on feeding/teethbrushing), so it never observed its own arm. It now observes all 7 Panda joints. On scratchitch, bedbathing and armmanipulation its joint slice was also one entry short (6 slots for a 7-joint arm).
  - **Human:** it got root-pose numbers and leg joints instead of its arm joints. It now observes all of its hinge joints.
  - **Velocities:** armmanipulation's joint velocities had the same problem and are fixed too.
  - Reported by BIRD (Logan Bhamidipaty's group).
- **The per-agent observation split now matches the observation layout.**
  - Feeding/teethbrushing: the robot lost its last entry to the human, and the human missed its last 3.
  - Armmanipulation: the robot's range included the human's first entry, and the human's ran 2 entries past the end of the observation.
- **Bed-bathing wipe targets are mapped through the arm capsules' frame** instead of the arm bodies' frame. Before, the 52 targets sat 0.4 mm–12 cm off the arm and 8 were inside the mattress. This changes bedbathing's distance reward, wiping credit and early termination. Reported by BIRD.
- **Handover velocity observations read `qvel` at the right addresses.** The panda joint velocities were offset into the object's free joint and the other arm, and the object's angular velocity was 4 entries wide. The handover observation is now 87 entries (was 88).
- **`DisabilityWrapper` restricts the range of the joint driven by the disabled actuator.** `joint_idx` indexes actuators (the action vector), but the range restriction used it as a joint id. In scratchitch, bedbathing and armmanipulation, `joint_idx: 13` weakened the `right_elbow` actuator but restricted `right_shoulder2`'s range; in feeding and teethbrushing it restricted `left_knee`'s.
- **Declared observation spaces** for `homogenisation_method=None` / `"concat"` (`None` is the `assistax.make` default) are no longer `num_agents` entries too wide.
- **Contact ids read the intended geom pairs.**
  - Feeding's head–spoon ids 186/187 read `(chair_seat, right_finger_collision)`. They are now 212/213: `(head, spoon_bowl)` and `(head, spoon_right_side)`. The wrong ids affected feeding's force observations, `r_force` and the force/touch preference rewards.
  - Armmanipulation's `UARM_HEND_CONTACT_ID` was 57, a second `(right_uarm, hook_platform)` slot. It is now 58, `(right_uarm, hook_end)`. This affects `ee_force` and the force preference reward.
- **`PreferenceRewardWrapper` accepts a partial `preference_weights` dict.** Leaving a weight out raised `KeyError` on the first step; the README's feeding example, which leaves out `touch_penalty`, hit this. A missing weight now counts as 0.0, the value the agent already observes for it.

### Added

- `assistax.envs.contact_utils`, plus a command-line interface (`python -m assistax.envs.contact_utils <env> [--pair GEOM_A GEOM_B]`), to map MJX contact-slot ids to geom names.
- A `tests/` regression suite using stdlib `unittest`: run `python -m unittest discover -s tests -t . -v` from the repo root. `tests/test_contacts.py` checks every hardcoded contact id; run it after any MuJoCo upgrade or scene-XML change.

### Changed

- `mujoco` and `mujoco-mjx` are pinned to `>=3.3.7,<3.5`. The hardcoded contact ids depend on MJX's contact layout.
- Docker CI only logs in to Docker Hub and pushes images for pushes to `main`. Pull requests from this repository build the image without publishing it. Pull requests from forks get no Docker Hub secrets, so their build fails, as it did before this change.

### Observation sizes

Entries in each env's observation vector (robot block, then human block):

| Task | robot | human | total (was) |
|---|---|---|---|
| scratchitch | 30 | 40 | 70 (69) |
| bedbathing | 26 | 36 | 62 (61) |
| armmanipulation | 30 | 38 | 68 (66) |
| feeding / teethbrushing | 23 | 35 | 58 (58; split was 22/33) |
| handover | 31 + 25 shared | 31 + 25 shared | 87 (88) |
| pushcoop | 27 | 27 | 54 (unchanged) |

With `homogenisation_method="max"` (as in the baselines), each agent sees `2 + max(robot, human)` entries, plus 7 with preference rewards. Under that padding, scratchitch and bedbathing keep their old widths (42 and 38), so old checkpoints for those two tasks still **load without error but receive different inputs**. Don't use pre-0.1.0 policies with 0.1.0.

### Provenance

These bugs were checked against the code and environment the published zoo was trained with. Its zoo-generation job snapshots on the training cluster (Feb–Jun 2026) ran MuJoCo/MJX 3.4.0, brax 0.14.0 and jax 0.8.0.

- **Every bug above that touches the zoo's tasks and settings was present in every zoo-generation job.** The two exceptions are handover (it isn't in the zoo) and the declared observation sizes (the zoo used `max` padding, which was already correct).
- **None came from a MuJoCo/MJX upgrade.** Joint addresses, capsule frames and contact layouts are identical under MuJoCo 3.2.6, 3.3.7 and 3.4.0.
- **Feeding's contact ids** were correct when written (`474bd3c`). They became wrong when left-arm collisions were enabled in the humanoid XML (`432fa79`).
- **Armmanipulation's `UARM_HEND_CONTACT_ID`** has pointed at the wrong slot since it was written.
