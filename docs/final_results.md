# Final Results

# Robust Manipulator Trajectory Tracking with Model-Based Control and Residual Reinforcement Learning in MuJoCo

## 1. Problem Formulation

The project evaluates robust joint-space trajectory tracking for a Franka Panda
arm in MuJoCo. The key question is whether a bounded Residual PPO policy can
reduce tracking error caused by inertial model mismatch while preserving a
Computed Torque Controller (CTC) as the primary controller.

The final control law is:

```text
tau_total = tau_CTC + delta_tau_RL
```

The reinforcement learning policy does not replace the model-based controller.
It only contributes a bounded residual torque.

## 2. Computed Torque Control

The nominal CTC computes:

```text
tau_CTC = M_nominal(q) * qddot_cmd + qfrc_bias_nominal
```

with:

```text
qddot_cmd = qddot_des + Kd * (qdot_des - qdot) + Kp * (q_des - q)
```

The nominal model is used for dynamics terms. The plant model is stepped
separately and can contain inertial perturbations.

## 3. Dual-Model Design

The dual-model design separates:

- Plant model: receives torque and evolves the simulated system.
- Nominal model: receives synchronized `q, qdot` and computes controller dynamics.

This prevents the controller from reading perturbed plant dynamics. Validation
confirmed that perturbing the plant does not modify the nominal model.

## 4. Model Mismatch Benchmark

Nominal CTC is nearly exact:

```text
Nominal overall RMSE = 0.00002389
Nominal motion RMSE  = 0.00002792
```

Inertial mismatch causes clear CTC degradation:

| Inertial scale | CTC overall RMSE | CTC motion RMSE | Max error |
|---:|---:|---:|---:|
| 1.00 | 0.00002389 | 0.00002792 | 0.00005411 |
| 1.25 | 0.01368397 | 0.01410428 | 0.02975740 |
| 1.50 | 0.02591773 | 0.02667415 | 0.05868047 |
| 1.75 | 0.03700884 | 0.03803673 | 0.08680130 |
| 2.00 | 0.04722092 | 0.04847359 | 0.11413561 |

This validates inertial mismatch as a meaningful target for residual learning.

## 5. Residual PPO Formulation

Final policy: Residual PPO v3.

Observation:

```text
q, qdot, q_des, qdot_des, qddot_des, q_des - q, qdot_des - qdot
```

Total dimension: `49`.

Action:

```text
7D normalized residual action in [-1, 1]
```

Residual torque limits:

```text
[8.0, 8.0, 8.0, 8.0, 1.2, 1.2, 1.2] Nm
```

The policy does not observe the inertial scale, plant mass, plant inertia,
plant mass matrix, plant bias, or plant passive force.

## 6. Training Distribution

Final training distribution:

```text
P(scale = 1.00) = 0.25
P(scale ~ Uniform(1.25, 1.75)) = 0.75
```

Each episode samples the scale once at reset. The scale is fixed throughout the
episode.

## 7. Reward

The final reward is:

```text
reward =
    - 1.0   * p_position
    - 0.1   * p_velocity
    - 0.03  * p_action
    - 0.005 * p_smooth
```

The action penalty regularizes residual control effort. It is an engineering
constraint on policy behavior, not a formal stability proof.

## 8. Multi-Seed Protocol

Final seeds:

```text
7, 17, 27
```

All seeds use the fixed-budget final model:

```text
requested timesteps = 30000
actual timesteps    = 30720
```

The nominal guard:

```text
NOMINAL_MOTION_RMSE_MAX = 0.005 rad
```

was kept as a diagnostic during training, but the final multi-seed comparison
does not select different checkpoints per seed. This avoids selection bias.

Guard-passing checkpoint count:

| Seed | Guard-passing checkpoint exists |
|---:|---|
| 7 | NO |
| 17 | NO |
| 27 | NO |

## 9. Per-Seed Raw Results

Motion RMSE:

| Scale | CTC | Seed 7 | Seed 17 | Seed 27 |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00002792 | 0.00867985 | 0.01253659 | 0.00694576 |
| 1.25 | 0.01410428 | 0.00885348 | 0.01060255 | 0.00847750 |
| 1.50 | 0.02667415 | 0.01198066 | 0.01255042 | 0.01186351 |
| 1.75 | 0.03803673 | 0.01626917 | 0.01684278 | 0.01585196 |
| 2.00 | 0.04847359 | 0.02092464 | 0.02200304 | 0.02004454 |

## 10. Mean and Sample Standard Deviation

Residual PPO mean +/- sample std (`ddof=1`):

| Scale | CTC motion RMSE | PPO mean | PPO std |
|---:|---:|---:|---:|
| 1.00 | 0.00002792 | 0.00938740 | 0.00286179 |
| 1.25 | 0.01410428 | 0.00931118 | 0.00113405 |
| 1.50 | 0.02667415 | 0.01213153 | 0.00036747 |
| 1.75 | 0.03803673 | 0.01632130 | 0.00049746 |
| 2.00 | 0.04847359 | 0.02099074 | 0.00098092 |

Improvement is computed per seed, then averaged:

| Scale | Mean improvement | Std |
|---:|---:|---:|
| 1.25 | 33.98% | 8.04% |
| 1.50 | 54.52% | 1.38% |
| 1.75 | 57.09% | 1.31% |
| 2.00 | 56.70% | 2.02% |

The nominal case is not reported as a relative percentage because the CTC
baseline is extremely close to zero. Its absolute RMSE increase is:

```text
0.00935948 +/- 0.00286179 rad
```

## 11. Residual Torque Analysis

Residual torque RMS:

| Scale | Mean residual RMS | Std |
|---:|---:|---:|
| 1.00 | 1.5302 | 0.1745 |
| 1.25 | 1.5696 | 0.1404 |
| 1.50 | 1.6836 | 0.0873 |
| 1.75 | 1.8574 | 0.0545 |
| 2.00 | 2.0731 | 0.0998 |

Residual / total torque RMS ratio:

| Scale | Mean ratio | Std |
|---:|---:|---:|
| 1.00 | 4.43% | 0.51% |
| 1.25 | 4.35% | 0.39% |
| 1.50 | 4.48% | 0.23% |
| 1.75 | 4.75% | 0.13% |
| 2.00 | 5.10% | 0.24% |

The policy remains a residual compensator rather than a replacement torque
controller.

## 12. Nominal Interference

At `scale=1.00`, nominal CTC is nearly perfect:

```text
CTC motion RMSE = 0.00002792
```

Residual PPO is worse for all three seeds:

```text
Residual PPO mean motion RMSE = 0.00938740 +/- 0.00286179
```

No seed passed the `0.005 rad` nominal safety guard. This is the main limitation
of the final algorithm.

## 13. Extrapolation

The policy was trained on mismatch scales in `[1.25, 1.75]`, plus nominal
episodes. The `scale=2.00` evaluation is outside the mismatch training range.

All three seeds still improve over CTC at `scale=2.00`:

```text
mean improvement = 56.70 +/- 2.02%
```

This is useful extrapolation within the tested inertial mismatch family, not a
claim of general robust control under arbitrary dynamics.

## 14. Safety Checks

Final deterministic evaluations:

```text
total torque clipping = 0
residual action saturation = 0
NaN / Inf = none observed
simulation instability = none observed
```

## 15. Limitations

- Simulation only.
- No Sim2Real validation.
- Only one explicit inertial mismatch family.
- No broad domain randomization.
- PPO is memoryless.
- Nominal-policy interference remains.
- No formal stability guarantee.

## 16. Final Conclusion

The final results support the following engineering conclusion:

> A bounded Residual PPO policy can consistently reduce joint trajectory tracking error under end-effector inertial model mismatch while keeping Computed Torque Control as the primary controller.

The strongest evidence is at `scale=1.50` and `scale=1.75`, where all three
seeds improve over CTC by roughly 55-57%. The residual torque remains small
relative to total torque and does not cause saturation.

The main caveat is nominal-policy interference: when plant and nominal dynamics
match, the residual policy still introduces unnecessary torque and performs
worse than near-perfect nominal CTC.
