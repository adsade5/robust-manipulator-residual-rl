# Phase 5: Model Mismatch Benchmark

## Purpose

Phase 3 Computed Torque Control is a perfect-model or nominal-condition
experiment: the controller uses the same dynamics that the simulated plant
actually follows. Phase 5 tests what happens when the plant dynamics differ
from the nominal dynamics used by the controller.

This phase does not implement PPO or any other RL method.

## Plant Versus Nominal Model

The benchmark uses a dual-model architecture:

```text
plant_model, plant_data
nominal_model, nominal_data
```

The plant model is the real simulation. The nominal model is an instantaneous
dynamics oracle for the Computed Torque Controller. Every control step follows:

```text
Plant q, qdot
-> copy state to Nominal Model
-> nominal mj_forward
-> nominal dynamics
-> CTC tau_base
-> torque clipping
-> Plant ctrl
-> plant mj_step
```

The nominal model is never integrated independently. It is synchronized from
the plant state each step.

The controller must not read perturbed plant inertia, plant `qfrc_bias`, or
plant `qfrc_passive` in mismatch scenarios. Reading those terms would leak the
true plant dynamics into the nominal controller and invalidate the benchmark.

## Scenarios

End-effector inertial mismatch is a payload-equivalent mismatch. It scales the
plant end-effector body mass and inertia together, while leaving the nominal
model unchanged. This is not claimed to be a fully modeled independent payload
body.

Joint damping mismatch scales only the plant arm DOF damping. It does not modify
armature, mass, torque limits, or dry friction.

External disturbance is not parameter mismatch. It is an unmodeled disturbance
applied only to the plant using MuJoCo external generalized force machinery.

The first benchmark perturbs one variable at a time so the result stays
interpretable.

## Metrics

Overall RMSE is computed across all timesteps and all seven joints. Motion RMSE
uses only the actual moving windows, 1-4 seconds and 5-8 seconds, so long hold
periods do not dilute the result.

Max error reports the largest absolute joint tracking error. Torque RMS is:

```text
sqrt(mean_t(||tau(t)||^2))
```

For external disturbance, the benchmark also reports maximum disturbance-window
error and recovery time. Recovery means the tracking-error norm returns below a
fixed threshold and stays there for 0.2 seconds.

## Next Phase

Residual PPO should only be introduced after nominal CTC is confirmed to work
and at least one mismatch or disturbance scenario produces a clear, explainable
performance degradation. The residual policy can then be trained on a selected
range of mismatch conditions rather than on a moving target of uncontrolled
experiment changes.
