# Assistax Environments

## 1. Overview

Assistax provides 7 multi-agent environments built on Brax/MJX physics for assistive robotics research. All environments use SI units (meters, m/s, Newtons). Actions are `Box(-1, 1)` per actuator. Environments are created via `assistax.envs.create()`, which applies wrappers in sequence: base env &rarr; EpisodeWrapper &rarr; VmapWrapper &rarr; AutoResetWrapper &rarr; DisabilityWrapper &rarr; PreferenceRewardWrapper.

## 2. Environment Summary Table

| Env | Agents | n_frames | dt (s) | Episode Length | Terminates Early? | Max Per-Step (env only) |
|-----|--------|----------|--------|----------------|-------------------|------------------------|
| ScratchItch | Robot + Human | 4 | 0.008 | 1000 | No | ~1.0 |
| BedBathing | Robot + Human | 4 | 0.008 | 1000 | Yes (all wiped) | ~4.0 |
| ArmManipulation | Robot + Human | 4 | 0.004 | 1000 | No | ~11.28 |
| PushCoop | Robot1 + Robot2 | 5 | 0.01 | 1000 | Yes (target/fall) | varies |
| Handover | Panda1 + Panda2 | 4 | 0.008 | 1000 | Yes (success/drop/collision) | varies |
| Feeding | Robot + Human | 4 | 0.008 | 1000 | No | ~5.0 |
| TeethBrushing | Robot + Human | 4 | 0.008 | 1000 | No | ~3.14 |

---

## 3. Per-Environment Details

### 3.1 ScratchItch

**Description:** A Panda robot uses a scratcher tool to scratch a randomly placed itch target on a human's arm (upper or lower).

#### Physical Setup
- **Robot (Agent 0):** Panda arm with scratcher tool. Actuators identified by `actuator*` prefix.
- **Human (Agent 1):** Humanoid with target arm (right upper/lower arm).
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 4, effective dt = 0.008s.

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning | Range | Max Weighted/Step |
|-----------|---------|----------------|------------------|-------|-------------------|
| `r_dist` | `exp(-dist^2 / 0.1)` | 1.0 | Gaussian centered on target. `dist` = Euclidean distance in **m** between scratcher tip and itch target. Scale = 0.1 m^2 | [0, 1] | 1.0 |
| `r_scratching` | `(r_dist < 0.1) * (v/v_t)*exp(-v/v_t) * (f/f_t)*exp(-f/f_t)` | 4.0 | Boltzmann product of scratcher tip speed and contact force. `v_t = 0.1 m/s`, `f_t = 3.0 N`. Each Boltzmann factor peaks at 1/e when value = target. Product peaks at 1/e^2 approx 0.135 | [0, ~0.135] | ~0.54 |
| `ctrl_cost` | `-sum(action^2)` | 1e-6 | Penalises large actions (dimensionless). Negligible weight | (-inf, 0] | ~0 |

**Key constraint:** The scratching reward is gated by `r_dist < 0.1`, which activates when the robot is **far** from the target (dist > ~0.48m). Distance and scratching rewards cannot be simultaneously maximised.

#### Combined Max Per-Step and Per-Episode
- When close to target: r_dist dominates, max ~1.0/step
- When far (scratching active): ~0.64/step (4.0 * 0.135 + small r_dist)
- **Max/episode:** ~1000

#### Termination Conditions
- No early termination (`done = 0.0` always).

#### Tracked Metrics
`reward_dist`, `reward_ctrl`, `reward_scratching`

---

### 3.2 BedBathing

**Description:** A Panda robot uses a wiper pad to wipe 52 target points distributed across a human's right arm (26 upper, 26 lower).

#### Physical Setup
- **Robot (Agent 0):** Panda arm with wiper tool.
- **Human (Agent 1):** Humanoid lying on bed.
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 4, effective dt = 0.008s.

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning | Range | Max Weighted/Step |
|-----------|---------|----------------|------------------|-------|-------------------|
| `r_dist` | `exp(-closest_dist^2 / 0.1)` | 1.0 | Gaussian distance to closest **unvisited** target in **m**. Scale = 0.1 m^2 | [0, 1] | 1.0 |
| `new_contacts` | Count of newly contacted targets | 3.0 | Integer count of targets wiped this step (typically 0 or 1). Target counts as wiped when distance < 0.1m **and** non-zero contact force | {0, 1, ...} | 3.0 (one new) |
| `ctrl_cost` | `-sum(action^2)` | 0 | Control cost. Weight = 0, so inactive | 0 | 0 |

#### Combined Max Per-Step and Per-Episode
- Max/step: ~4.0 (1.0 distance + 3.0 for one new target)
- **Max/episode:** ~1000 * 1.0 + 52 * 3.0 = **~1156** (all targets wiped + distance reward throughout)

#### Termination Conditions
- **Early termination** when all 52 targets are wiped (`contact_vector` all zeros).

#### Tracked Metrics
`reward_dist`, `reward_ctrl`, `reward_wiping`

---

### 3.3 ArmManipulation

**Description:** A Panda robot uses a hook tool to lift and reposition a human's weak right arm towards a waist-level target.

#### Physical Setup
- **Robot (Agent 0):** Panda arm with hook tool.
- **Human (Agent 1):** Humanoid lying on bed with weak right arm.
- **Physics:** Timestep explicitly set to **0.001s**, n_frames = 4, effective dt = **0.004s**.

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning | Range | Max Weighted/Step |
|-----------|---------|----------------|------------------|-------|-------------------|
| `r_hook_dist` | `1 - tanh(dist / 0.1)` | 1 (hardcoded) | Smooth inverse distance from hook tool to arm target site in **m**. Scale = 0.1m | [0, 1] | 1.0 |
| `r_waist_dist` | `exp(-dist^2 / 0.1)` | 10 (hardcoded) | Gaussian distance from human's lower arm to waist target in **m** | [0, 1] | 10.0 |
| `r_rot` | `sqrt(sum(angular_diff^2))` | 0.1 (hardcoded) | Frobenius norm of 3x3 rotation matrix difference. Measures angular alignment of hook tool | [0, ~5.66] | ~0.57 |
| `ctrl_cost` | `-sum(action^2)` | 1e-6 | Penalises large actions. Negligible weight | (-inf, 0] | ~0 |

**Note:** `dist_reward_weight` is hardcoded to 1 inside `step()` regardless of the constructor parameter.

#### Combined Max Per-Step and Per-Episode
- Max/step: ~11.28 (10.0 waist + 1.0 hook + ~0.28 rotation)
- **Max/episode:** ~11,280

#### Termination Conditions
- No early termination.

#### Tracked Metrics
`reward_hook_dist`, `reward_rot`, `reward_waist_dist`, `reward_ctrl`, `weighted_reward_hook_dist`, `weighted_reward_waist_dist`, `weighted_reward_ctrl`, `weighted_reward_rot`

---

### 3.4 PushCoop

**Description:** Two Panda robots cooperatively push and drag a T-shaped object across a table to a random target location, navigating around obstacles.

#### Physical Setup
- **Robot 1 (Agent 0):** Panda arm with pusher tool. 7 actuators.
- **Robot 2 (Agent 1):** Panda arm with pusher tool. 7 actuators.
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 5, effective dt = **0.01s**.

#### Reward Structure
This environment uses **heterogeneous rewards** (each robot receives its own reward signal). The task has two phases:

- **Push phase** (`phase_weight` near 1.0): T-object has not yet reached the middle of the table.
- **Drag phase** (`phase_weight` near 0.0): T-object reached the middle; roles reverse.

| Component | Formula | Default Weight | Physical Meaning |
|-----------|---------|----------------|------------------|
| `target_dist_reward` | `exp(-dist^2 / 0.3)` | 1.0 | Gaussian distance of T-object to goal. Scale = 0.3 m^2 |
| `dist_reward` (per robot) | `exp(-ee_dist^2 / 0.1)` | 0.5 | Distance from robot's pusher to T-object |
| `staging_reward` (per robot) | `exp(-staging_dist^2 / 0.1)` | 0.5 | Distance from robot's pusher to staging area |
| `t_at_target` | `dist < 0.1` | +100 terminal | T-object reached target |
| `t_fell` | Floor contact detected | -10 terminal | T-object fell off table |

**Robot 1:** `push_phase = target_dist + dist1 + ctrl`, `drag_phase = staging + ctrl + target_dist` (x2 multiplier)
**Robot 2:** `push_phase = staging + ctrl`, `drag_phase = target_dist + dist2 + ctrl`

#### Termination Conditions
- **Early termination** when T-object reaches target (dist < 0.1m) or falls off table.

#### Tracked Metrics
`robo1_reward_dist`, `robo2_reward_dist`, `robo1_reward_ctrl`, `robo2_reward_ctrl`, `reward_t_to_goal`, `robot1_staging_reward`, `robot2_staging_reward`, `drag_phase_locked`, `phase_weight`, `robo1_push_reward`, `robo1_drag_reward`, `robo2_push_reward`, `robo2_drag_reward`

---

### 3.5 Handover

**Description:** Two Panda robots perform a cooperative object handover. Panda1 picks up an object, transfers it to Panda2, which then places it at a goal location.

#### Physical Setup
- **Panda 1 (Agent 0):** 7 arm joints + 1 gripper = 8 actuators.
- **Panda 2 (Agent 1):** 7 arm joints + 1 gripper = 8 actuators.
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 4, effective dt = 0.008s.

#### 6 Phases
1. **APPROACH:** Panda1 moves toward the object
2. **GRASP:** Panda1 grasps the object (force threshold = 0.5 N)
3. **TRANSFER:** Panda1 lifts and moves object to handover location
4. **HANDOVER:** Panda2 approaches, both robots grip the object
5. **RETREAT:** Panda2 takes the object while Panda1 releases
6. **PLACE:** Panda2 places the object at the goal location

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning |
|-----------|---------|----------------|------------------|
| `dist` | `exp(-dist^2 / 0.1)` | 1.0 | Phase-dependent distance reward (approach/transfer/retreat) |
| `grasp` | `2 * (both_touching + force_in_range)` | 2.0 | Grasping with appropriate force (phase 1 only) |
| `maintain_grip` | `gripping + stable` | 1.0 | Maintaining grip on object (phases 2-5) |
| `handover` | `3 * (approach + dual_grip)` | 3.0 | Successful dual-grip handover (phase 3 only) |
| `place` | `2 * (dist + on_table + upright)` | 2.0 | Placing at goal (phase 5 only) |
| `ctrl` | `-1e-4 * sum(action^2)` | 1e-4 | Control cost (always active) |
| `drop_penalty` | `-10` | -- | Object hits the floor |
| `collision_penalty` | `-5` | -- | Robot hands too close (< 0.08m) |
| `phase_transition_bonus` | `+10` per transition | -- | Up to 5 transitions = +50 total |

#### Termination Conditions
- **Success:** Object placed at goal (dist < 0.02m, correct height, PLACE phase)
- **Drop:** Object z-position < 0.1m
- **Collision:** Severe robot collision (penalty < -2.0)

#### Tracked Metrics
`reward_dist`, `reward_grasp`, `reward_maintain_grip`, `reward_handover`, `reward_place`, `reward_ctrl`, `penalty_drop`, `penalty_collision`, `phase_transition_bonus`, `phase`

---

### 3.6 Feeding

**Description:** A Panda robot uses a spoon to feed a human by navigating to the mouth with correct orientation, speed, and gentle contact.

#### Physical Setup
- **Robot (Agent 0):** Panda arm with spoon tool. Joints: indices 20-27.
- **Human (Agent 1):** Humanoid seated in wheelchair. Joints: indices 1-20.
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 4, effective dt = 0.008s.

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning | Range | Max Weighted/Step |
|-----------|---------|----------------|------------------|-------|-------------------|
| `r_dist` | `exp(-3 * dist)` | 2.0 | Exponential decay distance to mouth in **m**. Steeper falloff (scale = 3) than Gaussian | [0, 1] | 2.0 |
| `r_orientation` | `1.0 * r_pour + (0.3 + 0.7*proximity) * r_aim` | 1.0 | Blend of spoon tilt alignment (upright far, toward-mouth close) and yaw alignment | [~-1.3, ~1.3] | ~1.3 |
| `r_velocity` | `exp(-(speed_error^2 / 0.01))` | 1.0 | Gaussian around target speed. Far: 0.2 m/s, close: 0.0 m/s. Sigma = 0.1 m/s | [0, 1] | 1.0 |
| `r_force` | `(f/1.0) * exp(-f/1.0)` | 1.0 | Boltzmann contact force on mouth. Target = 1.0 N. Peaks at 1/e | [0, 1/e] | ~0.368 |
| `ctrl_cost` | `-sum(action^2)` | 1e-6 | Negligible | ~0 | ~0 |

Reward = `2.0 * r_dist + 1.0 * (r_orientation + r_velocity + r_force) + 1e-6 * ctrl_cost`

#### Combined Max Per-Step and Per-Episode
- Max/step: ~4.7 (2.0 dist + ~1.3 orientation + 1.0 velocity + 0.368 force)
- **Max/episode:** ~4700. Realistically ~5000 accounting for orientation range.

#### Termination Conditions
- No early termination.

#### Tracked Metrics
`reward_dist`, `reward_ctrl`, `reward_orientation`, `reward_velocity`, `reward_force`

---

### 3.7 TeethBrushing

**Description:** A Panda robot uses a toothbrush to brush a human's teeth, requiring approach, alignment, and appropriate brushing motion with contact force.

#### Physical Setup
- **Robot (Agent 0):** Panda arm with toothbrush tool. Joints: indices 20-27.
- **Human (Agent 1):** Humanoid seated. Joints: indices 1-20.
- **Physics:** MuJoCo default timestep = 0.002s, n_frames = 4, effective dt = 0.008s.

#### Reward Components

| Component | Formula | Default Weight | Physical Meaning | Range | Max Weighted/Step |
|-----------|---------|----------------|------------------|-------|-------------------|
| `r_dist` | `exp(-3 * dist)` | 2.0 | Exponential decay distance to mouth in **m**. Same as Feeding | [0, 1] | 2.0 |
| `r_align_weighted` | `r_align * r_dist` | 1.0 | Alignment reward (bristle-toward-mouth + handle-horizontal, averaged) weighted by distance. Only significant when close | [0, 1] | ~1.0 |
| `r_brush` | `(dist < 0.05) * (v_tan/0.1)*exp(-v_tan/0.1) * (f/1.0)*exp(-f/1.0)` | 1.0 | Brushing reward: Boltzmann product of tangential speed (target 0.1 m/s) and contact force (target 1.0 N). Gated by proximity (dist < 0.05m). Peaks at 1/e^2 | [0, ~0.135] | ~0.135 |
| `ctrl_cost` | `-sum(action^2)` | 1e-6 | Negligible | ~0 | ~0 |

Reward = `2.0 * r_dist + 1.0 * (r_brush + r_align_weighted) + 1e-6 * ctrl_cost`

#### Combined Max Per-Step and Per-Episode
- Max/step: ~3.14 (2.0 dist + ~1.0 align + 0.135 brush)
- **Max/episode:** ~3140

#### Termination Conditions
- No early termination.

#### Tracked Metrics
`reward_dist`, `reward_ctrl`, `reward_align`, `reward_brushing`, `reward_force`

---

## 4. Preference Reward System

### 4a. Overview

`PreferenceRewardWrapper` (`wrappers/training.py:270`) augments any environment's reward with human-preference-based components. Applied after AutoResetWrapper in the `create()` pipeline. The pure function `compute_preference_reward()` (`wrappers/training.py:520`) is used in the AHT path for vmapped partner preferences.

### 4b. Physical Meaning of Input Variables

All environments expose these in `state.info`:

- **`ee_speed`** (m/s): end-effector speed = `norm(pos_new - pos_old) / dt`. How fast the robot's tool tip moves through space.
- **`ee_force`** (N): contact force magnitude on human body = `norm(force_vector)`. How much force the tool exerts on the human.
- **`action_magnitude`** (dimensionless): `norm(action_vector)` where each action element is in [-1, 1]. For a 7-DOF robot arm, max norm = sqrt(7) approx 2.65.

### 4c. Component Formulas and Physical Interpretation

**Speed Preference** -- "Does the robot move at the human's preferred speed?"
- `_gaussian_preference(ee_speed, [speed_min, speed_max])`
- Returns 1.0 if `speed_min <= ee_speed <= speed_max` (within preferred range)
- Outside range: Gaussian decay `exp(-((v - center)^2) / (2 * width^2))` where center = (min+max)/2, width = (max-min)/2
- Default range: **[0.06, 0.14] m/s** -- the human prefers the robot to move between 6 and 14 cm/s
- Sweep range: min sampled from [0.03, 0.08], max from [sampled_min, 0.20]

**Force Preference** -- "Is the robot applying the right amount of contact force?"
- Same Gaussian preference formula applied to `ee_force`
- Default range: **[1.5, 3.5] N** -- the human prefers between 1.5 and 3.5 Newtons of contact force
- Sweep range: min from [1.0, 2.0], max from [sampled_min, 5.0]

**Action Efficiency** -- "Is the robot using minimal effort?"
- `exp(-action_magnitude / max_action_magnitude)`
- Rewards smaller actions (less energy, smoother motion). Value = 1.0 at zero action, decays exponentially
- Default `max_action_magnitude = 0.8` -- at action norm = 0.8, reward = exp(-1) approx 0.368
- Sweep range: [0.5, 1.0]

**Touch Penalty** -- "Penalize unexpected new contacts"
- Binary: 1.0 if `prev_force < threshold AND current_force >= threshold` (new touch transition), else 0.0
- Weight is negative (default -0.03), so this subtracts from total pref reward
- Default threshold: 0.1 N (IPPO training) or 0.3 N (zoo gen)

### 4d. Budget Normalization (default mode)

```
positive_weight_sum = w_speed + w_force + w_action
norm_factor         = reward_budget / positive_weight_sum

speed_component   = norm_factor * w_speed * speed_pref       in [0, norm_factor * w_speed]
force_component   = norm_factor * w_force * force_pref       in [0, norm_factor * w_force]
action_component  = norm_factor * w_action * action_eff      in [0, norm_factor * w_action]
touch_component   = norm_factor * w_touch * touch_penalty    in [norm_factor * w_touch, 0]

KEY PROPERTY: max positive total = reward_budget (always, regardless of individual weights)
total_pref_reward = overall_weight * (sum of all 4 components)
```

### 4e. Per-Step Max Tables for Common Configs

**IPPO training** (`config/ippo.yaml`): budget=2.0, weights: 0.25/0.35/0.15/-0.03, touch_threshold=0.1

```
norm_factor    = 2.0 / 0.75 = 2.6667
Max speed      = 2.6667 * 0.25 * 1.0 = 0.6667 /step
Max force      = 2.6667 * 0.35 * 1.0 = 0.9333 /step
Max action_eff = 2.6667 * 0.15 * 1.0 = 0.4000 /step
Max positive   = 2.0 /step  ->  2000 /episode
Touch penalty  = 2.6667 * (-0.03) * 1 = -0.08 per new-touch event
```

**Zoo gen** (`config/ippo_zoo_gen.yaml`): budget=1.5, weights: 0.25/0.35/0.15/-0.03, touch_threshold=0.3

```
norm_factor    = 1.5 / 0.75 = 2.0
Max speed      = 0.50 /step
Max force      = 0.70 /step
Max action_eff = 0.30 /step
Max positive   = 1.5 /step  ->  1500 /episode
Touch penalty  = -0.06 per event
```

**Combined max (env + pref) per episode:**

| Env | Env-only Max/Episode | + IPPO Pref (budget=2) | + Zoo Pref (budget=1.5) |
|-----|---------------------|------------------------|------------------------|
| ScratchItch | ~1000 | ~3000 | ~2500 |
| BedBathing | ~1156 | ~3156 | ~2656 |
| ArmManipulation | ~11280 | ~13280 | ~12780 |
| PushCoop | varies | N/A (robot-robot) | N/A |
| Feeding | ~4700 | ~6700 | ~6200 |
| TeethBrushing | ~3140 | ~5140 | ~4640 |
| Handover | varies | N/A (robot-robot) | N/A |

> **Note:** These are theoretical upper bounds. In practice, rewards are significantly lower because optimizing one component (e.g., staying still for max action efficiency) conflicts with others (e.g., moving to match speed preference, or reaching the target for distance reward).

---

## 5. Zoo Generation Preference Sampling

### 5a. Overview

Zoo generation trains a diverse population of agents, each with different preference reward configurations. This creates a partner pool for Zero-Shot Coordination (ZSC) training where the learning agent must adapt to partners with varied preferences.

### 5b. Two-Level Sampling Architecture

1. **`generate_preference_configs()`** (`baselines/utils.py:1265`) samples `num_configs` unique preference parameter sets
2. Each parameter set is used to train `NUM_SEEDS` agents
3. Total zoo agents = `num_configs * NUM_SEEDS * num_agents_per_team`

### 5c. What "Meta-Ranges" Mean

The `PREFERENCE_SWEEP` config in `ippo_zoo_gen.yaml` defines ranges-of-ranges (meta-ranges). Each preference parameter is independently sampled:

- **Weights** (w_speed, w_force, w_action, w_touch): sampled uniformly from their meta-range. These determine the *relative importance* of each preference component
- **Preference ranges** (speed_range, force_range): define what physical values the simulated human "prefers". Sampled as two values (min, max) with the constraint min < max
- **max_action_magnitude**: defines how strict the action efficiency preference is. Lower = stricter
- **reward_budget, overall_weight, touch_threshold**: NOT swept in default config (use base_config values)

### 5d. Sampling Mechanics (from code)

```python
# For each parameter with a {min, max} spec:
value = jax.random.uniform(rng, shape=(num_configs,), minval=spec["min"], maxval=spec["max"])

# For range parameters (ensure min < max):
speed_range_min = uniform(rng, [0.03, 0.08])              # sample min first
speed_range_max = uniform(rng, [speed_range_min, 0.20])    # max >= sampled min

# Parameters without {min, max} spec use fixed base_config value:
reward_budget = broadcast(base_config_value, shape=(num_configs,))  # e.g., 1.5 for all
```

### 5e. PREFERENCE_SWEEP Meta-Ranges Table

| Parameter | Min | Max | Units / Meaning |
|-----------|-----|-----|-----------------|
| w_speed | 0.1 | 0.5 | Relative weight for speed preference |
| w_force | 0.1 | 0.6 | Relative weight for force preference |
| w_action | 0.05 | 0.3 | Relative weight for action efficiency |
| w_touch | -0.1 | -0.01 | Touch penalty weight (always negative) |
| speed_range_min | 0.03 | 0.08 | Lower bound of preferred speed in **m/s** (3-8 cm/s) |
| speed_range_max | 0.10 | 0.20 | Upper bound of preferred speed in **m/s** (10-20 cm/s, constrained >= sampled min) |
| force_range_min | 1.0 | 2.0 | Lower bound of preferred force in **N** |
| force_range_max | 3.0 | 5.0 | Upper bound of preferred force in **N** (constrained >= sampled min) |
| max_action_mag | 0.5 | 1.0 | Action efficiency decay constant (dimensionless). Lower = stricter |

Fixed (not swept): `reward_budget=1.5`, `overall_weight=1.0`, `touch_threshold=0.3`

### 5f. How Sampled Weights Affect Max Reward

Because budget normalization divides by `positive_weight_sum`, the individual component maxima change with sampled weights, but the **total positive max is always = reward_budget**:

| Scenario | w_speed | w_force | w_action | pos_sum | norm_factor | Max Speed | Max Force | Max Action | Max Total |
|----------|---------|---------|----------|---------|-------------|-----------|-----------|------------|-----------|
| Min weights | 0.1 | 0.1 | 0.05 | 0.25 | 6.00 | 0.60 | 0.60 | 0.30 | **1.50** |
| Max weights | 0.5 | 0.6 | 0.3 | 1.40 | 1.07 | 0.54 | 0.64 | 0.32 | **1.50** |
| Speed-heavy | 0.5 | 0.1 | 0.05 | 0.65 | 2.31 | 1.15 | 0.23 | 0.12 | **1.50** |
| Force-heavy | 0.1 | 0.6 | 0.05 | 0.75 | 2.00 | 0.20 | 1.20 | 0.10 | **1.50** |

Touch penalty varies: `norm_factor * |w_touch|`, ranging from:
- Mildest: 1.07 * 0.01 = **0.011** per event
- Harshest: 6.00 * 0.10 = **0.600** per event

### 5g. Tuning Guide: Reward Budget Scaling

To calibrate preference reward signal relative to environment reward:

```
reward_budget / env_max_per_step = preference signal ratio

ScratchItch:      2.0 / 1.0  = 2.0x    (prefs dominate env reward)
BedBathing:       2.0 / 4.0  = 0.5x    (prefs are secondary)
ArmManipulation:  2.0 / 11.3 = 0.18x   (prefs are minor)
Feeding:          2.0 / 4.7  = 0.43x   (prefs are secondary)
TeethBrushing:    2.0 / 3.14 = 0.64x   (prefs are meaningful)
```

---

## 6. Episode Mechanics

- **EpisodeWrapper:** `episode_length=1000` (default), `action_repeat=1` (default in `create()`)
- **n_frames** is physics substeps per `pipeline_step()` call -- NOT RL steps. Reward is computed once per `step()` call
- **dt** = MuJoCo_timestep * n_frames. MuJoCo default timestep = 0.002s for all envs except ArmManipulation which sets 0.001s
- **Effective:** 1000 reward computations per episode
- **Episode duration** in simulated time: 1000 * dt seconds (e.g., ScratchItch: 1000 * 0.008 = 8.0s)

| Env | MuJoCo timestep (s) | n_frames | dt (s) | Simulated episode duration (s) |
|-----|---------------------|----------|--------|-------------------------------|
| ScratchItch | 0.002 | 4 | 0.008 | 8.0 |
| BedBathing | 0.002 | 4 | 0.008 | 8.0 |
| ArmManipulation | 0.001 | 4 | 0.004 | 4.0 |
| PushCoop | 0.002 | 5 | 0.010 | 10.0 |
| Handover | 0.002 | 4 | 0.008 | 8.0 |
| Feeding | 0.002 | 4 | 0.008 | 8.0 |
| TeethBrushing | 0.002 | 4 | 0.008 | 8.0 |

---

## Source Files

| File | Contents |
|------|----------|
| `assistax/envs/scratchitch.py` | ScratchItch environment |
| `assistax/envs/bedbathing.py` | BedBathing environment |
| `assistax/envs/armmanipulation.py` | ArmManipulation environment |
| `assistax/envs/pushcoop.py` | PushCoop environment |
| `assistax/envs/handover.py` | CooperativeHandover environment |
| `assistax/envs/feeding.py` | Feeding environment |
| `assistax/envs/teethbrushing.py` | TeethBrushing environment |
| `assistax/envs/__init__.py` | `create()` pipeline |
| `assistax/wrappers/training.py` | PreferenceRewardWrapper, compute_preference_reward, EpisodeWrapper |
| `assistax/envs/base_env.py` | Action spaces Box(-1, 1) |
| `assistax/baselines/utils.py` | `generate_preference_configs()` |
| `assistax/baselines/IPPO/config/ippo.yaml` | IPPO training defaults |
| `assistax/baselines/IPPO/config/ippo_zoo_gen.yaml` | Zoo gen defaults + PREFERENCE_SWEEP |
