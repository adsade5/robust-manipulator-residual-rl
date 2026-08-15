# Phase 2: Joint-Space Trajectory Tracking

## Purpose

Phase 2 builds a joint-space trajectory tracking baseline on the derived Panda
torque model. It intentionally stays short of computed torque, Cartesian
control, impedance control, feedforward acceleration, ROS, and reinforcement
learning. The goal is to compare joint-space PD with joint-space PD plus
MuJoCo bias compensation while following a smooth point-to-point trajectory.

## Posture Regulation Versus Tracking

Phase 1 was posture regulation:

```text
q_des = constant
qdot_des = 0
```

Phase 2 is trajectory tracking:

```text
q_des = q_des(t)
qdot_des = qdot_des(t)
```

The desired position and velocity now change during the motion, so the
controller must use the actual velocity target rather than assuming it is zero.

## Quintic Trajectory

For a segment with duration `T`, normalized time is:

```text
s = t / T
```

The blend is:

```text
h(s) = 10s^3 - 15s^4 + 6s^5
```

For joint vector displacement `dq = q_goal - q_start`:

```text
q_des = q_start + h(s)dq
qdot_des = (30s^2 - 60s^3 + 30s^4)dq / T
qddot_des = (60s - 180s^2 + 120s^3)dq / T^2
```

The implementation clamps outside the segment: before the segment starts it
returns `q_start` with zero velocity and acceleration; after it ends it returns
`q_goal` with zero velocity and acceleration.

## Why Quintic Instead Of Linear Interpolation

Linear interpolation has discontinuous velocity at the beginning and end of a
move and gives no smooth acceleration profile. A quintic point-to-point
trajectory provides zero initial velocity, zero final velocity, zero initial
acceleration, and zero final acceleration, producing a smoother command for a
torque-controlled manipulator.

## Controllers

The PD trajectory controller is:

```text
tau = Kp(q_des - q) + Kd(qdot_des - qdot)
```

The PD plus bias compensation controller is:

```text
tau = Kp(q_des - q) + Kd(qdot_des - qdot) + qfrc_bias
```

Here `qfrc_bias` is MuJoCo's generalized bias force. It generally includes
gravity, Coriolis, and centrifugal bias terms. During motion it should not be
simplified to pure `g(q)`.

Both controllers use the same `Kp` and `Kd`. That is essential for a fair
comparison: the measured difference should come from the added bias
compensation, not from different feedback stiffness or damping.

## Metrics

Per-joint RMSE is:

```text
RMSE_j = sqrt(mean_t((q_des_j(t) - q_j(t))^2))
```

Overall RMSE is computed across all timesteps and all seven joints:

```text
overall RMSE = sqrt(mean_t,j((q_des_j(t) - q_j(t))^2))
```

RMSE summarizes typical tracking error. Maximum absolute error captures the
worst transient tracking error and can reveal brief lag or overshoot hidden by
the average.

Torque saturation must be counted because clipping changes the controller that
is actually applied. If a controller spends much of the experiment saturated,
its tracking error reflects actuator limits as much as feedback design.

## Acceleration Is Logged But Not Used Yet

The quintic generator already computes `qddot_des`, but the Phase 2 PD and
PD+GC controllers do not use it. The desired acceleration is logged now because
it will be used in the next phase: Computed Torque Control.
