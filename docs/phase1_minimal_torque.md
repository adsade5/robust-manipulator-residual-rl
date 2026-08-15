# Phase 1: Minimal Torque Control

## Experiment Purpose

This phase creates a minimal torque-control baseline for the Franka Emika Panda
model from MuJoCo Menagerie. The only model change is made in a copied,
derived model: the first seven arm actuators are converted from position-servo
style general actuators into direct joint torque motors. The official model
files under `assets/mujoco_menagerie/franka_emika_panda` are not modified.

The goal is to understand the dynamics/control interface before adding later
controllers such as trajectory tracking, computed torque, or Cartesian
impedance.

## MuJoCo State And Force Terms

`qpos` is the generalized position vector. For this Panda model, the first
seven entries are the arm joint angles and the last two entries are the two
finger slider positions.

`qvel` is the generalized velocity vector. For hinge and slide joints in this
model, each controlled joint has one corresponding velocity entry.

`ctrl` is the actuator control input vector. Its physical meaning depends on
the actuator definition. In the official Panda model, the first seven `ctrl`
values are position targets for high-gain affine actuators, not torque commands.
In the derived torque model, the first seven `ctrl` values are direct torque
commands because those actuators are joint motors with `gear="1"`.

`actuator_force` is the actuator-space force produced by each actuator after
MuJoCo applies actuator dynamics, gains, bias terms, and limits.

`qfrc_actuator` is the generalized force contribution applied to the model by
the actuators. With a joint motor, scalar gear of 1, and one actuator per hinge
joint, the first seven actuator controls map directly to the corresponding
joint generalized forces.

`qfrc_bias` is MuJoCo's generalized bias force. It generally includes gravity,
Coriolis, and centrifugal terms. It should not be documented as simply `g(q)`
for all states, although at zero velocity its arm entries are commonly used as
gravity compensation torques.

## Why The Official Actuators Are Not Torque Inputs

The official Panda model uses `<general>` actuators with affine bias parameters,
for example nonzero `gainprm` and `biasprm`. Those actuators implement a
position-servo-like behavior: `ctrl` is interpreted as a desired joint position,
and MuJoCo converts the position error and velocity feedback into force. Sending
a value through `ctrl` therefore does not mean applying that many Nm at the
joint.

## Why The Derived Motor Actuators Are Torque Inputs

The derived model replaces only the first seven arm actuators with:

```xml
<motor name="torqueN" joint="jointN" gear="1" ctrllimited="true"
       ctrlrange="-limit limit" forcelimited="true"
       forcerange="-limit limit"/>
```

A MuJoCo motor actuator has no position-servo bias. With `gear="1"` on a hinge
joint, the actuator control maps directly to the joint generalized force. The
first seven `ctrl` entries can therefore be used as joint torque commands in Nm,
subject to both control and force limits.

## Four Minimal Experiments

`none` sends zero torque to the arm. It verifies the passive response of the
robot under gravity and the model's native damping/armature.

`gravity` reads the current `qfrc_bias` arm entries every simulation step and
sends them as torque commands. At the home state with zero velocity, this
approximately cancels gravity and tests whether the torque interface is wired
with the correct sign and joint addresses.

`pd` holds the official home posture with joint-space PD only:

```text
tau = Kp(q_des - q) - Kd qdot
```

This intentionally omits gravity compensation so the steady-state error caused
by gravity load is visible.

`pd_gc` adds bias-force compensation:

```text
tau = Kp(q_des - q) - Kd qdot + qfrc_bias
```

It verifies that adding MuJoCo's current bias force reduces the posture-holding
error compared with pure PD.

## Torque Limits

Both the XML and the experiment code limit commanded torque. Joints 1 through 4
are limited to `[-87, 87]` Nm. Joints 5 through 7 are limited to `[-12, 12]` Nm.
