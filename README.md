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

The runtime uses two rates: a background RealSense/AprilTag worker updates
the object estimate at the camera rate (60 Hz), while the RTDE loop reads
wrench data, updates admittance, and sends `speedL` commands at 200 Hz.  The
RTDE loop reconstructs the camera-to-object pose from the latest base-frame
object estimate and the current robot pose.  Commands stop when the last
successful tag estimate is older than `vision_stale_timeout`.

## High-rate PBVS EKF and moving targets

The controller enables an 18-state PBVS EKF by default, following the process
model in Oliva et al., ICRA 2022:

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
SOPD acceleration command. Motion compensation is smoothly enabled after the first accepted
measurements to avoid a startup acceleration transient. The old
stationary-target reconstruction remains available with:

```powershell
python e2\main.py --controller cB --no-pbvs-ekf
```

Logs include `vision_s0..5`, `object_twist0..5`,
`object_acceleration0..5`, the corresponding unscaled
`ekf_object_*_raw0..5`,
`ekf_motion_compensation_scale`, `ekf_innovation_norm`, and `ekf_nis` for
tuning and comparison.

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
`--no-gravity-compensation` to bypass it for a comparison run.

The fitted model predicts the no-contact static RTDE wrench from the TCP
rotation. Runtime processing subtracts this prediction before the deadband
and low-pass filter. With a model active, `z` estimates only the remaining
model/sensor residual at the current pose instead of absorbing the full
gravity wrench. The CSV log includes `static_wrench0..5` and
`tcp_force_static_comp0..5` for validation.
