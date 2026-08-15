# Phase 3: Computed Torque Control

## Purpose

Phase 3 adds a joint-space Computed Torque Controller and compares it against
the Phase 2 PD plus bias-compensation baseline. The model, torque limits,
trajectory, timing, and simulation settings are kept the same.

## Why PD+GC Still Needs Tracking Error

PD plus bias compensation cancels MuJoCo's current bias force, but it does not
explicitly command the inertial torque needed to follow the desired
acceleration. The inertial part of the motion is still generated indirectly by
tracking error through the PD torque terms.

Computed torque adds the model inertia term directly. It tries to make the
closed-loop joint acceleration behave like a chosen acceleration command.

## Dynamics And MuJoCo Terms

A useful form of the robot dynamics here is:

```text
M(q) qddot + c(q, qdot) = tau + passive
```

In MuJoCo terms:

`M(q)` is the joint-space inertia matrix stored internally in sparse form and
multiplied with `mj_mulM` or expanded with `mj_fullM`.

`qfrc_bias` is the generalized bias force. It generally includes gravity,
Coriolis, and centrifugal terms.

`qfrc_passive` is the passive generalized force. In the Panda model this
includes the official joint damping that was intentionally preserved.

## Computed Torque

The acceleration command is:

```text
a_cmd = qddot_des + Kp_acc e + Kd_acc edot
```

where:

```text
e = q_des - q
edot = qdot_des - qdot
```

The commanded torque is:

```text
tau = M(q) a_cmd + qfrc_bias - qfrc_passive
```

Under ideal simulation conditions with no clipping, no unmodeled forces, and no
active contact constraints:

```text
qddot = a_cmd
```

Substituting the acceleration command gives:

```text
e_ddot + Kd_acc e_dot + Kp_acc e = 0
```

This is a second-order error system in acceleration space.

## Why M(q) Is A Matrix

The arm inertia is not seven independent scalar inertias. `M(q)` is a full
matrix because accelerating one joint can require torque at other joints. The
off-diagonal elements represent joint dynamic coupling caused by the robot's
linked-body geometry and mass distribution.

## Acceleration-Domain Gains

Computed torque uses `Kp_acc` and `Kd_acc` inside an acceleration command. Their
units and physical meaning differ from Phase 2 torque-domain PD gains. They
should not be copied from the torque gains or described as numerically fair in
the same way. This project defines them from a conservative natural frequency
and damping ratio:

```text
Kp_acc = wn^2
Kd_acc = 2 zeta wn
```

## Torque Saturation

Torque clipping breaks the ideal cancellation because the torque sent to the
actuator no longer equals `M(q)a_cmd + qfrc_bias - qfrc_passive`. Saturation is
therefore recorded per joint and must be reported rather than hidden by raising
torque limits.

## Current Limitations

This phase assumes perfect model knowledge in simulation. It does not include
model uncertainty, contact tasks, sensor noise, actuator latency, force control,
Cartesian control, impedance control, MPC, WBC, ROS, or RL. A simulation CTC
result should not be claimed to transfer to a real robot without additional
robustness and validation work.
