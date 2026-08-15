# Phase 6.1 Nominal-Safe Residual PPO

Phase 6 v1 showed that residual PPO can compensate end-effector inertial
mismatch when it is trained on:

```text
scale ~ Uniform(1.25, 1.75)
```

It improved the medium and strong mismatch cases, and also improved the unseen
2.00 scale. However, v1 degraded the nominal 1.00 case because the nominal CTC
baseline is already nearly exact. The policy never saw scale 1.00 during
training, and it also does not observe the true inertial scale. A trajectory
phase dependent residual can therefore become a useful feedforward-like
correction for mismatch while still over-compensating when the plant is nominal.

The policy observation is unchanged:

```text
q, qdot, q_des, qdot_des, qddot_des, e_q, e_qdot
```

The `qddot_des` term is a reference trajectory signal, not privileged plant
information. It is useful because inertial mismatch is directly coupled to
acceleration demand. The policy still does not receive plant mass, plant
inertia, plant M(q), plant bias, plant passive force, or the sampled scale.

## Nominal-Safe Change

Phase 6.1 keeps the Panda model, dual-model architecture, CTC controller,
trajectory, observation definition, residual torque limits, reward formula, PPO
network, PPO hyperparameters, actuator torque limits, timestep, and policy
frequency unchanged.

Only the episode scale sampling distribution changes:

```text
P(scale = 1.00) = 0.25
P(scale ~ Uniform(1.25, 1.75)) = 0.75
```

The scale is sampled once at reset and remains fixed for the episode. It is
recorded in `info` and logs as `sample_source = nominal` or `mismatch`, but it is
not included in the observation.

## Nominal Safety Guard

Checkpoint evaluation now uses:

```text
1.00, 1.25, 1.50, 1.75
```

The engineering guard is:

```text
NOMINAL_MOTION_RMSE_MAX = 0.005 rad
```

A checkpoint with nominal motion RMSE above this threshold cannot become the
best model. For checkpoints that pass the guard, the score is the mean motion
RMSE over only the mismatch scales:

```text
1.25, 1.50, 1.75
```

Nominal is therefore a safety constraint, not the optimization score.

## Validation

Zero residual still exactly reproduces Phase 5 CTC:

| Scale | Overall RMSE diff | Motion RMSE diff | Max error diff | Max torque diff | Result |
|---:|---:|---:|---:|---:|---|
| 1.00 | 0 | 0 | 0 | 0 | PASS |
| 1.50 | 0 | 0 | 0 | 0 | PASS |

The training reset distribution produced 17 nominal and 52 mismatch episodes:

```text
nominal fraction = 0.2464
mismatch fraction = 0.7536
```

This is close to the intended 25% / 75% mixture.

## Checkpoint Results

No checkpoint passed the nominal safety guard.

| Timestep | Nominal motion RMSE | 1.25 | 1.50 | 1.75 | Mean mismatch | Guard |
|---:|---:|---:|---:|---:|---:|---|
| 4500 | 0.02445212 | 0.02843313 | 0.03609058 | 0.04437689 | 0.03630020 | FAIL |
| 9000 | 0.01522917 | 0.02032016 | 0.02992214 | 0.03947265 | 0.02990499 | FAIL |
| 13500 | 0.01759607 | 0.02124373 | 0.02951363 | 0.03822136 | 0.02965957 | FAIL |
| 18000 | 0.01692508 | 0.01884917 | 0.02598487 | 0.03408737 | 0.02630714 | FAIL |
| 22500 | 0.01133155 | 0.01306162 | 0.01939849 | 0.02650110 | 0.01965374 | FAIL |
| 27000 | 0.01229142 | 0.01304107 | 0.01622671 | 0.02055754 | 0.01660844 | FAIL |

Because no guard-passing checkpoint exists, Phase 6.1 has no valid `best_model`.
The final model was evaluated only as a diagnostic model, not as an accepted
best checkpoint.

## v1 vs v2 Diagnostic Evaluation

| Scale | CTC motion RMSE | PPO v1 motion RMSE | PPO v2 final motion RMSE |
|---:|---:|---:|---:|
| 1.00 | 0.00002792 | 0.02258049 | 0.00971576 |
| 1.25 | 0.01410428 | 0.01436022 | 0.00970849 |
| 1.50 | 0.02667415 | 0.01268607 | 0.01275967 |
| 1.75 | 0.03803673 | 0.01759034 | 0.01712203 |
| 2.00 | 0.04847359 | 0.02461465 | 0.02192680 |

The mixture training reduced nominal degradation relative to v1, but not enough
to satisfy the nominal safety guard:

```text
v1 nominal motion RMSE = 0.02258049
v2 final nominal motion RMSE = 0.00971576
guard threshold = 0.005
```

At scale 1.50, v2 final keeps essentially the same mismatch compensation as v1:

```text
v1 scale 1.50 motion RMSE = 0.01268607
v2 scale 1.50 motion RMSE = 0.01275967
```

This means the nominal episodes helped reduce nominal damage, but with the
unchanged reward and hyperparameters they did not fully enforce near-zero
residual behavior under nominal dynamics.

## Result

Phase 6.1 is a useful diagnostic step, but it does not meet the defined PASS
criteria because no checkpoint passed the nominal safety guard. The current
milestone remains:

```text
Residual PPO milestone = PARTIAL
```

This result should not be interpreted as solving all model uncertainty. The
current residual PPO is still only targeted at one explicit end-effector
inertial mismatch family.
