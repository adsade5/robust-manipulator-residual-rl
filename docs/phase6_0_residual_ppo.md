# Phase 6: Residual PPO

## Why RL Starts After Phase 5

Phase 5 established that nominal Computed Torque Control works extremely well
when the plant and nominal dynamics match, and that end-effector inertial
mismatch creates stable, interpretable tracking degradation. That gives
Residual PPO a clear job: compensate a known class of model mismatch without
replacing the model-based controller.

## Residual Control

The controller remains:

```text
tau = tau_CTC + delta_tau_RL
```

The base CTC torque dominates. PPO outputs only a bounded residual torque. This
keeps the learned policy from becoming a pure black-box torque controller and
preserves the value of the model-based baseline.

## Observation

The policy observes:

```text
q
qdot
q_des
qdot_des
qddot_des
q_des - q
qdot_des - qdot
```

Each term has seven dimensions, for a 49-dimensional observation. `qddot_des`
is included because inertial mismatch is directly related to acceleration
demand. It is part of the reference trajectory and is not privileged plant
information.

The policy does not receive inertial scale, plant mass, plant inertia, plant
mass matrix, plant bias force, or plant passive force.

## Training Distribution

Each episode samples one fixed inertial scale:

```text
scale ~ Uniform(1.25, 1.75)
```

The scale is fixed for the whole episode. This avoids mixing within-episode
time variation with model mismatch adaptation.

## Residual Torque Limit

The action is normalized:

```text
a in [-1, 1]^7
```

and converted to:

```text
delta_tau = a * residual_torque_limit
```

The residual limits are small relative to the full actuator limits. This forces
PPO to make limited corrections instead of taking over the controller.

## Reward

The reward has four terms:

```text
- position tracking penalty
- velocity tracking penalty
- residual action magnitude penalty
- action smoothness penalty
```

The position and velocity terms drive trajectory tracking. The action magnitude
and smoothness terms discourage unnecessary or noisy residual torques.

## PPO

The policy network maps observations to residual actions. The value network
estimates expected return and helps reduce variance in policy-gradient updates.
PPO clipping limits how far each policy update can move from the behavior that
generated the rollout, improving training stability.

## Evaluation

Evaluation uses deterministic actions at:

```text
1.00, 1.25, 1.50, 1.75, 2.00
```

Scales 1.25, 1.50, and 1.75 are seen or interpolation settings. Scale 2.00 is
unseen extrapolation. Scale 1.00 is the nominal sanity check: residual PPO
should not significantly damage the nearly perfect nominal CTC baseline.

## Limitations

This is still simulation-only. It is not Sim2Real. The residual policy is
trained only for end-effector inertial mismatch and does not include damping
randomization, external disturbance, sensor noise, delay, contact, force
control, or broad domain randomization.
