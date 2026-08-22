# Visual admittance controllers

The `e2` experiment provides three PBVS controller variants:

- `cO`: visual servoing only.
- `cA`: feature-space admittance with proxy feature `s_p`.
- `cB`: base-frame Cartesian admittance with a proxy camera pose.

Run a controller with, for example:

```powershell
python e2\main.py --controller cB
```

`cB` implements

```text
measured wrench -> Cartesian admittance -> proxy camera pose -> PBVS/SOPD
```

<!-- Its Cartesian offset, velocity, and acceleration are saved as
`cart_offset0..5`, `cart_velocity0..5`, and `cart_acceleration0..5` in
`e2/data/log_cB.csv`.  The existing `s`, `sp`, and `sd` columns remain
available so that controller results can be plotted with the same tools. -->

The runtime is split into three independent execution chains:

1. `AprilTagVision` owns RealSense capture and AprilTag detection at the camera
   rate.
2. The RTDE chain reads robot pose, speed, and wrench; applies gravity
   compensation, force filtering, EKF prediction, admittance, and PBVS; then
   sends `speedL` commands at 200 Hz.
3. `OpenCVDisplay` renders the overlay and owns `imshow`/`waitKey` without
   running GUI work in the RTDE chain.

Vision and display exchange latest-only snapshots instead of queues. If
detection or rendering is slower than its producer, an old sample is replaced
rather than accumulated, so neither chain applies back-pressure to the RTDE
control loop. The RTDE loop reconstructs the camera-to-object pose from the
latest estimate and current robot pose. Commands stop when the last successful
tag estimate is older than `vision_stale_timeout`. Display key presses are
forwarded as events: `q` requests a safe stop and `z` requests force zeroing in
the RTDE chain.

## High-rate PBVS EKF and moving targets

The controller provides an optional 18-state PBVS EKF, following the process
model in Oliva et al., ICRA 2022. It is disabled by default and enabled with
`--ekf`:

```powershell
python e2\main.py --controller cB --ekf
```

Its process model is:

```text
x = [s, object twist u_o, object acceleration u_o_dot]
s_dot = L(s) * u_c + N(s) * u_o
u_o_dot = object acceleration
object acceleration_dot = 0
```

Actual TCP speed drives prediction at the 200 Hz RTDE rate. AprilTag poses at
the camera rate update the first six states. Frame capture and processing
timestamps are kept separate; delayed measurements update a buffered EKF state
and the stored 200 Hz predictions are replayed to the present. The resulting
high-rate feature estimate is used as `s`. Estimated object twist `u_o` is used
in the error derivative through `N*u_o`, and `N*u_o_dot` is compensated in the
SOPD acceleration command. Motion compensation is smoothly enabled after the
first accepted measurements to avoid a startup acceleration transient. Without
`--ekf`, the controller uses the stationary-target reconstruction.

Logs include `vision_s0..5`, `object_twist0..5`,
`object_acceleration0..5`, the corresponding unscaled
`ekf_object_*_raw0..5`,
`ekf_motion_compensation_scale`, `ekf_innovation_norm`, and `ekf_nis` for
tuning and comparison.

The visualization reports three independently measured instantaneous rates:
`Total` is the RTDE main-loop rate, `Vision` is the valid AprilTag pose rate,
and `Force` is the rate at which a valid wrench is consumed by the cA/cB
admittance and control update. They are also logged as `total_hz`, `vision_hz`,
and `force_control_hz`; the older `control_hz` column remains the `speedL`
command-send rate.

## Static gravity compensation

The single-pose `z` zero cannot compensate an orientation-dependent payload
wrench. Keep the normal tool/camera assembly installed, remove all external
contact, and collect at least 12, preferably 20--30, diverse static attitudes:

```powershell
python e2\calibrate_static_gravity.py collect
```

Use pendant freedrive to place the robot, release it, and press Enter to record
each attitude. Press `f` to fit and save
`e2/data/static_gravity_model.json`. Do not touch the tool during a capture.
The normal controller automatically loads this file on its next start. Use
`--no-gravity` to bypass it for a comparison run.

The fitted model predicts the no-contact static RTDE wrench from the TCP
rotation. Runtime processing subtracts this prediction before the deadband
and low-pass filter. With a model active, `z` estimates only the remaining
model/sensor residual at the current pose instead of absorbing the full
gravity wrench. The CSV log includes `static_wrench0..5` and
`tcp_force_static_comp0..5` for validation.
