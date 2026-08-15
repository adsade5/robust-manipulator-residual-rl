# Phase 6.2 Action-Regularized Residual PPO

Phase 6 v1 trained residual PPO only on end-effector inertial mismatch:

```text
scale ~ Uniform(1.25, 1.75)
```

It improved medium and strong mismatch tracking, including the unseen 2.00
scale, but it damaged the nominal case:

```text
scale 1.00 CTC motion RMSE = 0.00002792
scale 1.00 PPO v1 motion RMSE = 0.02258049
```

Phase 6.1 v2 added nominal episodes to the training distribution:

```text
P(scale = 1.00) = 0.25
P(scale ~ Uniform(1.25, 1.75)) = 0.75
```

This reduced nominal degradation:

```text
scale 1.00 PPO v2 motion RMSE = 0.00971576
```

However, v2 still did not pass the nominal safety guard:

```text
NOMINAL_MOTION_RMSE_MAX = 0.005 rad
```

## Control Interpretation

The PPO policy outputs a bounded residual action, not a joint target angle:

```text
action in [-1, 1]^7
delta_tau_RL = action * RESIDUAL_TORQUE_LIMIT
tau_total = tau_CTC + delta_tau_RL
```

The base CTC controller remains the primary controller. PPO can only add a
limited torque residual. The action magnitude penalty is an engineering
regularizer on residual control effort. It does not prove stability and should
not be interpreted as a Lyapunov argument.

The intended trade-off is:

```text
tracking improvement vs residual effort
```

When nominal CTC already tracks nearly perfectly, residual action should be less
attractive. When inertial mismatch causes large tracking error, residual action
can still be worthwhile.

## Single Variable Change

Phase 6.2 keeps all v2 settings fixed:

- same Panda model
- same dual-model architecture
- same CTC controller and gains
- same trajectory and timing
- same 49D observation
- same residual torque limits
- same actuator torque limits
- same policy frequency and MuJoCo timestep
- same PPO network and hyperparameters
- same training budget and seed
- same scale mixture distribution
- same nominal safety guard and best-model rule

The only reward change is:

```text
ACTION_PENALTY_WEIGHT: 0.01 -> 0.03
```

The action penalty was increased only modestly. A much larger penalty could
drive the residual policy toward a near-zero policy and erase mismatch
compensation, which would make the experiment less informative.

## Validation

Zero-action dynamics are unchanged:

| Scale | Overall RMSE diff | Motion RMSE diff | Max error diff | Max torque diff | Result |
|---:|---:|---:|---:|---:|---|
| 1.00 | 0 | 0 | 0 | 0 | PASS |
| 1.50 | 0 | 0 | 0 | 0 | PASS |

Reward sanity with `ACTION_PENALTY_WEIGHT = 0.03`:

| Case | Position | Velocity | Action | Smoothness | Total |
|---|---:|---:|---:|---:|---:|
| scale 1.00 zero | -0.000000 | -0.000000 | 0.000000 | 0.000000 | -0.000000 |
| scale 1.50 zero | -0.268692 | -0.000169 | 0.000000 | 0.000000 | -0.268860 |
| scale 1.00 random | -3.926902 | -0.285777 | -0.010125 | -0.003391 | -4.226196 |
| scale 1.50 random | -3.280725 | -0.296893 | -0.010137 | -0.003369 | -3.591125 |

The action penalty is active, but it is not orders of magnitude larger than the
tracking terms.

## Training

Training used seed 7 and the same 30,000 requested timesteps as v2. Stable
Baselines3 completed the rollout at 30,720 actual timesteps.

The reset distribution was:

```text
nominal episodes = 17
mismatch episodes = 52
nominal fraction = 0.2464
mismatch fraction = 0.7536
```

No checkpoint passed the nominal safety guard, so no accepted best model exists
for v3. The final model was evaluated only as a diagnostic model.

## Results

| Scale | CTC motion RMSE | PPO v1 | PPO v2 | PPO v3 final diagnostic |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00002792 | 0.02258049 | 0.00971576 | 0.00867985 |
| 1.25 | 0.01410428 | 0.01436022 | 0.00970849 | 0.00885348 |
| 1.50 | 0.02667415 | 0.01268607 | 0.01275967 | 0.01198066 |
| 1.75 | 0.03803673 | 0.01759034 | 0.01712203 | 0.01626917 |
| 2.00 | 0.04847359 | 0.02461465 | 0.02192680 | 0.02092464 |

v3 improves the diagnostic final model relative to v2 across all evaluated
scales. It still fails nominal safety because:

```text
scale 1.00 PPO v3 motion RMSE = 0.00867985 > 0.005
```

Mismatch compensation is retained:

```text
scale 1.50 improvement vs CTC = 55.09%
scale 1.75 improvement vs CTC = 57.23%
```

The unseen 2.00 case also remains improved:

```text
scale 2.00 improvement vs CTC = 56.83%
```

## Residual Analysis

Nominal residual RMS:

| Policy | Residual torque RMS | Action RMS |
|---|---:|---:|
| PPO v1 | 1.4466 | 0.0991 |
| PPO v2 | 1.9361 | 0.0962 |
| PPO v3 | 1.5853 | 0.0781 |

The desired strict ordering `v3 < v2 < v1` for residual RMS did not occur.
Instead, v3 reduced residual RMS versus v2 but remained above v1. Action RMS did
drop monotonically, which is consistent with the increased action penalty.

At scale 1.50:

| Policy | Motion RMSE | Residual torque RMS |
|---|---:|---:|
| PPO v1 | 0.01268607 | 1.5943 |
| PPO v2 | 0.01275967 | 2.0674 |
| PPO v3 | 0.01198066 | 1.7477 |

v3 did not collapse to a zero-residual policy. It still uses meaningful
residual compensation while remaining much smaller than the total torque RMS.

## Safety

Across the deterministic v3 diagnostic evaluation:

```text
total torque clipping = 0
residual action saturation = 0
NaN / Inf = none observed
instability = none observed
```

The residual nature is preserved. At scale 1.50:

```text
residual RMS / total torque RMS = 1.7477 / 37.6689 = 4.64%
```

## Verdict

Phase 6.2 is a useful improvement over v2, but it does not satisfy the hard
nominal guard. The final milestone remains:

```text
Residual PPO milestone = PARTIAL
```

No further reward sweep or hyperparameter search was performed.
