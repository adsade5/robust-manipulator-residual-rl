# Phase 4: Cartesian Impedance Control

## Purpose

Phase 4 moves from joint-space control to end-effector Cartesian impedance. The
controller holds a desired end-effector pose and reacts to an external force
disturbance with a spring-damper behavior. This is not force control, hybrid
position/force control, MPC, WBC, ROS, or RL.

## Joint Space Versus Cartesian Space

Joint-space control commands behavior directly in the robot's joint
coordinates. Cartesian-space control commands behavior of the end-effector in
task coordinates such as world-frame position and orientation.

Forward kinematics maps:

```text
q -> end-effector pose
```

The Jacobian maps joint velocity to Cartesian twist:

```text
xdot = J(q) qdot
```

The translational rows map to linear velocity and the rotational rows map to
angular velocity.

## Jacobian Transpose

A Cartesian wrench `W = [F, M]` can be mapped into joint torques with:

```text
tau = J^T W
```

This expresses virtual work consistency: a Cartesian force at the end-effector
corresponds to generalized forces at the joints.

## Cartesian Impedance

The translational impedance law is:

```text
F = K_pos e_pos + D_pos e_vel
```

where `e_pos = p_des - p` and `e_vel = v_des - v`. In this phase the target pose
is static, so `v_des = 0`.

The rotational part uses the same idea:

```text
M = K_rot e_rot + D_rot e_omega
```

The orientation error is computed on SO(3), not by subtracting Euler angles.

## Why This Is Not Ordinary Position Control

Impedance control explicitly defines a desired force-displacement behavior. A
low stiffness behaves softly and allows larger displacement under the same
external force. A high stiffness behaves harder and allows smaller
displacement.

Damping dissipates energy and suppresses oscillation. The damping values here
are conservative engineering initial values inspired by critical damping, but
the true Cartesian virtual mass is configuration dependent, not unit mass.

## Disturbance Verification

It is not enough to show that the end-effector can hold a target pose. A real
impedance claim should be tested by applying an external force and measuring the
resulting displacement. For a simple one-dimensional static spring:

```text
delta x ~= F / K
```

This is only a sanity check. The MuJoCo result can differ because of 6D
coupling, redundancy, damping, bias/passive compensation, changing Jacobians,
finite transient windows, and torque limits.

## Current Limitations

This phase is simulation only. It assumes model-perfect bias compensation and
does not use a physical force sensor. There is no environment contact task, no
force regulation target, and no hybrid position-force controller yet. The next
phase will build a contact force control or hybrid position-force experiment.
