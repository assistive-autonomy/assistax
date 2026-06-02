# Preference Meta-Range Analysis

## Motivation

Train and test partner sets in AHT show similar performance, suggesting the current preference meta-ranges may be too narrow. Most sampled preference configurations produce near-identical rewards at the robot's natural operating point, limiting discrimination between train and test partners.

---

## 1. How the Preference Reward Works

### Gaussian Preference Function

From `assistax/wrappers/training.py:490-506`:

```python
def _gaussian_preference(self, value, preferred_range):
    min_pref, max_pref = preferred_range
    center = (min_pref + max_pref) / 2
    width = (max_pref - min_pref) / 2    # sigma = half the range width

    in_range = (value >= min_pref) & (value <= max_pref)
    gaussian_reward = exp(-((value - center)^2) / (2 * width^2))

    return 1.0 if in_range else gaussian_reward
```

- Returns **1.0** when the value falls inside `[min_pref, max_pref]`.
- Outside the range, decays as a Gaussian with `sigma = width = (max_pref - min_pref) / 2`.

### Budget Normalization

From `assistax/wrappers/training.py:325-326, 466-473`:

```
norm_factor = reward_budget / (w_speed + w_force)
```

Only positive weights are summed (touch penalty is negative, excluded from the denominator). Each preference term is then:

```
reward_i = norm_factor * w_i * gaussian_pref_i(value)
```

With `reward_budget = 2`, the maximum total positive preference reward is always 2.0, regardless of weight magnitudes.

---

## 2. Physical Units

| Quantity | Unit | Computation |
|----------|------|-------------|
| Speed | m/s | `norm(site_xpos[t] - site_xpos[t-1]) / dt`, where `dt = pipeline_dt * n_frames` |
| Force | N | `norm(contact_force)` from the MuJoCo contact solver |

### ScratchItch Environment Targets

From `assistax/envs/scratchitch.py:44-45, 113`:

- `target_scratcher_speed = 0.1` m/s
- `target_scratcher_force = 3.0` N
- `n_frames = 4`

### Boltzmann Reward Shape

From `assistax/envs/scratchitch.py:241-244`:

```
r_scratching = (|dist| < dist_scale)
    * (speed / target_speed) * exp(-speed / target_speed)
    * (force / target_force) * exp(-force / target_force)
```

This is the product of two `x * e^{-x}` terms, each peaking at `x = 1` (i.e., at `speed = 0.1 m/s` and `force = 3.0 N`). The base environment reward therefore naturally drives the robot toward these exact target values.

---

## 3. Current Meta-Ranges

From `assistax/baselines/IPPO/config/ippo_zoo_gen.yaml:65-89`:

| Parameter | min | max | Default |
|-----------|-----|-----|---------|
| `w_speed` | 0.1 | 0.5 | 0.25 |
| `w_force` | 0.1 | 0.6 | 0.35 |
| `w_touch` | -0.1 | -0.01 | -0.03 |
| `speed_range_min` | 0.03 | 0.08 | 0.06 |
| `speed_range_max` | 0.10 | 0.20 | 0.14 |
| `force_range_min` | 1.0 | 2.0 | 1.5 |
| `force_range_max` | 3.0 | 5.0 | 3.5 |

---

## 4. Why Current Ranges Produce Low Discrimination

### Speed ranges cluster around the natural operating point

- `speed_range_min` is sampled from [0.03, 0.08].
- `speed_range_max` is sampled from [0.10, 0.20].
- Every sampled speed range therefore contains or is very close to the overlap region ~0.06-0.10 m/s.
- The robot's natural operating speed (driven by the Boltzmann reward peaking at 0.1 m/s) falls inside or near every sampled preferred range.
- Result: `_gaussian_preference(0.1, [any_sampled_range])` returns approximately **1.0** for all sampled configs.

### Force ranges cluster around the natural operating point

- `force_range_min` is sampled from [1.0, 2.0].
- `force_range_max` is sampled from [3.0, 5.0].
- Every sampled force range contains or nearly contains the overlap region ~2.0-3.0 N.
- The robot's natural operating force (~3.0 N from the Boltzmann target) falls inside or near every sampled preferred range.
- Result: `_gaussian_preference(3.0, [any_sampled_range])` returns approximately **1.0** for all sampled configs.

### Budget normalization compresses weight differences

With `reward_budget = 2`:

| Config | w_speed | w_force | norm_factor | Effective speed contrib | Effective force contrib |
|--------|---------|---------|-------------|------------------------|------------------------|
| A | 0.1 | 0.6 | 2.0/0.7 = 2.86 | 0.286 | 1.714 |
| B | 0.5 | 0.1 | 2.0/0.6 = 3.33 | 1.667 | 0.333 |

When both Gaussian preferences return ~1.0, both configs yield total positive reward ~2.0. The weight distribution changes the breakdown between speed and force components, but since the robot satisfies both preferences simultaneously, the total reward is indistinguishable.

### Consequence for AHT

If every preference config in the zoo produces near-identical reward signals at the robot's natural behavior, then:
1. All zoo partners effectively train the robot to do the same thing.
2. Train vs. test partner splits are not meaningfully different.
3. AHT cannot learn to adapt because there is nothing to adapt to.

---

## 5. Recommended Wider Meta-Ranges

```yaml
PREFERENCE_SWEEP:
  num_configs: 4
  w_speed:
    min: 0.05
    max: 0.8
  w_force:
    min: 0.05
    max: 0.8
  w_touch:
    min: -0.3
    max: -0.01
  speed_range_min:
    min: 0.01
    max: 0.15
  speed_range_max:
    min: 0.05
    max: 0.40
  force_range_min:
    min: 0.5
    max: 4.0
  force_range_max:
    min: 2.0
    max: 10.0
```

### Summary of changes

| Parameter | Old range | New range | Rationale |
|-----------|-----------|-----------|-----------|
| `w_speed` | [0.1, 0.5] | [0.05, 0.8] | Wider weight range allows near-single-objective configs |
| `w_force` | [0.1, 0.6] | [0.05, 0.8] | Same as above |
| `w_touch` | [-0.1, -0.01] | [-0.3, -0.01] | Allows strongly touch-averse partners |
| `speed_range_min` | [0.03, 0.08] | [0.01, 0.15] | Can now prefer speeds above the natural 0.1 target |
| `speed_range_max` | [0.10, 0.20] | [0.05, 0.40] | 0.40 m/s is 4x the environment target |
| `force_range_min` | [1.0, 2.0] | [0.5, 4.0] | Can prefer forces above the natural 3.0 target |
| `force_range_max` | [3.0, 5.0] | [2.0, 10.0] | 10.0 N is 3.3x the environment target |

The key insight is that some sampled ranges should now be **entirely above** or **entirely below** the robot's natural operating point, forcing the robot to deviate from its default behavior to satisfy the preference.

---

## 6. Expected Behavioral Archetypes

With wider ranges, the zoo should produce partners requiring genuinely different robot behaviors:

| Archetype | speed_range | force_range | Weights | Robot behavior |
|-----------|-------------|-------------|---------|----------------|
| Slow & gentle | [0.01, 0.05] | [0.5, 2.0] | balanced | Slow, light scratching |
| Fast & firm | [0.15, 0.40] | [4.0, 10.0] | balanced | Vigorous, deep scratching |
| Speed-sensitive | any | any | w_speed=0.8, w_force=0.05 | Speed dominates reward signal |
| Force-sensitive | any | any | w_speed=0.05, w_force=0.8 | Force dominates reward signal |
| Touch-averse | any | any | w_touch=-0.3 | Strongly avoids unintended contact |

These archetypes create genuine behavioral diversity: a robot trained with a "slow & gentle" partner cannot zero-shot transfer to a "fast & firm" partner without adaptation, which is exactly the challenge AHT should learn to solve.

---

## 7. Caveats

1. **Physical achievability**: Very extreme ranges (e.g., preferred speed > 0.3 m/s or force > 8 N) may be difficult or impossible for the robot to achieve within the physics simulation. If the robot can never satisfy a preference, it receives near-zero preference reward, which could cause training collapse or reward hacking. Recommended: run a short untrained rollout with extreme configs to verify the achievable operating range.

2. **Effective single-objective configs**: With `w_speed=0.8, w_force=0.05`, the force component contributes only ~6% of the total preference reward. The robot may learn to ignore force entirely. This is intentional (it creates diversity) but may require more training steps for convergence.

3. **Touch penalty scaling**: Increasing `w_touch` to -0.3 makes the touch penalty up to 10x stronger than the current default of -0.03. This could dominate the reward signal if the robot frequently makes unintended contact. Monitor touch penalty magnitude relative to total reward during training.

4. **Sanity checking**: Before running a full zoo generation with new ranges, run a few individual training runs with extreme corner-case configs (e.g., slowest + gentlest, fastest + firmest) to verify stable training.

5. **Number of configs**: With wider ranges, `num_configs: 4` may not adequately cover the space. Consider increasing to 6-8 configs if compute budget allows, to ensure the train/test split captures meaningfully different behavioral regions.

---

## 8. Original Settings (Pre-Widening Reference)

The following meta-ranges were in use prior to the widening applied on 2026-03-01. Recorded here for reproducibility of earlier zoo generations.

```yaml
# Original PREFERENCE_SWEEP meta-ranges (all algorithms)
PREFERENCE_SWEEP:
  num_configs: 4
  w_speed:     {min: 0.1,  max: 0.5}
  w_force:     {min: 0.1,  max: 0.6}
  w_touch:     {min: -0.1, max: -0.01}
  speed_range_min: {min: 0.03, max: 0.08}
  speed_range_max: {min: 0.10, max: 0.20}
  force_range_min: {min: 1.0,  max: 2.0}
  force_range_max: {min: 3.0,  max: 5.0}

# Dead entries removed (MAPPO & MASAC only):
  w_action:              {min: 0.05, max: 0.3}
  max_action_magnitude:  {min: 0.5,  max: 1.0}
# Dead ENV_KWARGS entries removed (MAPPO & MASAC only):
#   preference_weights.action_efficiency: 0.15
#   preference_ranges.max_action_magnitude: 0.8
#   variable_names.action_magnitude: "action_magnitude"  (also removed from IPPO)
```

### Original pop_gen seed assignment
All algorithms shared the same seeds, causing identical random streams:

| Algorithm | Seeds |
|-----------|-------|
| IPPO | 0, 1, 2 |
| MAPPO | 0, 1, 2 |
| MASAC | 0, 1 (2 commented out) |

New assignment: IPPO 0-2, MAPPO 3-5, MASAC 6-8.
