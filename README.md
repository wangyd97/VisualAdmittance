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

Its Cartesian offset, velocity, and acceleration are saved as
`cart_offset0..5`, `cart_velocity0..5`, and `cart_acceleration0..5` in
`e2/data/log_cB.csv`.  The existing `s`, `sp`, and `sd` columns remain
available so that controller results can be plotted with the same tools.

The runtime uses two rates: a background RealSense/AprilTag worker updates
the object estimate at the camera rate (60 Hz), while the RTDE loop reads
wrench data, updates admittance, and sends `speedL` commands at 200 Hz.  The
RTDE loop reconstructs the camera-to-object pose from the latest base-frame
object estimate and the current robot pose.  Commands stop when the last
successful tag estimate is older than `vision_stale_timeout`.
