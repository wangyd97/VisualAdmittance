from dataclasses import dataclass

import numpy as np

from .Mathematic import to_vec3, to_vec6


@dataclass
class PBVSConfig:
    tag_size: float = 0.05
    detect_stride: int = 1
    apriltag_nthreads: int = 1
    apriltag_quad_decimate: float = 1.0
    apriltag_quad_sigma: float = 0.4
    enable_visualization: bool = True
    visualization_stride: int = 1
    # Stop robot commands when no successful AprilTag update has arrived
    # within this many seconds.
    vision_stale_timeout: float = 0.10

    pos_threshold: float = 0.005
    rot_threshold: float = 0.02
    stable_frames: int = 10
    slow_after_convergence: bool = False
    convergence_slowdown_frames: int = 5
    convergence_velocity_scale: float = 0.1

    max_runtime: float = 0.0

    max_linear_vel: float = 0.80
    max_angular_vel: float = 0.80
    enable_feature_lowpass: bool = True
    feature_lowpass_tau: float = 0.02
    use_commanded_tcp_pose_estimate: bool = False
    enable_Rc_lowpass: bool = False
    Rc_lowpass_tau: float = 0.01

    controller_mode: str = "SOPD"
    controller_name: str = "cA"

    enable_feature_admittance: bool = True
    feature_admittance_mass: object = 1.0
    feature_admittance_damping: object = 100.0
    feature_admittance_stiffness: object = 200.0
    feature_admittance_force: object = 0.0
    feature_admittance_Bc_inv: object = 1.0
    # Per-axis safety bounds in feature coordinates
    # [metres, metres, metres, quaternion-vector components].
    feature_admittance_max_offset: object = (0.03, 0.03, 0.03, 0.15, 0.15, 0.15)
    feature_admittance_max_velocity: object = (0.10, 0.10, 0.10, 0.50, 0.50, 0.50)
    feature_admittance_max_acceleration: object = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0)

    enable_cartesian_admittance: bool = False
    # Cartesian admittance state is a base-frame camera-pose offset
    # [dx, dy, dz, dRx, dRy, dRz].  Translational and rotational entries
    # therefore use physical Cartesian units rather than feature units.
    cartesian_admittance_mass: object = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
    cartesian_admittance_damping: object = (20.0, 20.0, 20.0, 2.0, 2.0, 2.0)
    cartesian_admittance_stiffness: object = (500.0, 500.0, 500.0, 20.0, 20.0, 20.0)
    cartesian_admittance_wrench_scale: object = 1.0
    cartesian_admittance_max_offset: object = (0.03, 0.03, 0.03, 0.15, 0.15, 0.15)
    cartesian_admittance_max_velocity: object = (0.10, 0.10, 0.10, 0.50, 0.50, 0.50)
    cartesian_admittance_max_acceleration: object = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0)

    enable_rtde_tcp_force: bool = True
    rtde_tcp_force_frame: str = "base"
    rtde_force_scale: float = 1.0
    enable_static_gravity_compensation: bool = False
    static_gravity_model_path: str = ""
    rtde_force_bias_samples: int = 30
    rtde_force_lowpass_tau: float = 0.05
    # Soft deadband [N, N, N, Nm, Nm, Nm], applied after bias removal.
    rtde_force_deadband: object = (1.0, 1.0, 1.0, 0.05, 0.05, 0.05)

    # R6 parameter order: [x, y, z, rx, ry, rz]
    kp: object = 1.0
    kd: object = 2.0

    accel_limit: object = None
    accel_limit_pos: object = float("inf")
    accel_limit_rot: object = float("inf")

    log_save_path: str = "log.csv"
    enable_memory_log: bool = True
    status_print_interval: int = 1

    def __post_init__(self):
        self.detect_stride = max(1, int(self.detect_stride))
        self.apriltag_nthreads = max(1, int(self.apriltag_nthreads))
        self.apriltag_quad_decimate = max(1.0, float(self.apriltag_quad_decimate))
        self.apriltag_quad_sigma = max(0.0, float(self.apriltag_quad_sigma))
        self.visualization_stride = max(1, int(self.visualization_stride))
        self.vision_stale_timeout = max(0.0, float(self.vision_stale_timeout))
        self.slow_after_convergence = bool(self.slow_after_convergence)
        self.convergence_slowdown_frames = max(0, int(self.convergence_slowdown_frames))
        self.convergence_velocity_scale = float(np.clip(self.convergence_velocity_scale, 0.0, 1.0))
        self.max_linear_vel = max(0.0, float(self.max_linear_vel))
        self.max_angular_vel = max(0.0, float(self.max_angular_vel))
        self.enable_feature_lowpass = bool(self.enable_feature_lowpass)
        self.feature_lowpass_tau = max(0.0, float(self.feature_lowpass_tau))
        self.use_commanded_tcp_pose_estimate = bool(self.use_commanded_tcp_pose_estimate)
        self.enable_Rc_lowpass = bool(self.enable_Rc_lowpass)
        self.Rc_lowpass_tau = max(0.0, float(self.Rc_lowpass_tau))
        self.enable_memory_log = bool(self.enable_memory_log)
        self.status_print_interval = max(1, int(self.status_print_interval))

        self.kp = to_vec6(self.kp, "kp")
        self.kd = to_vec6(self.kd, "kd")
        self.controller_name = str(self.controller_name)
        if self.controller_name not in ("cO", "cA", "cB"):
            raise ValueError("controller_name must be 'cO', 'cA', or 'cB'")
        # Controller names select exactly one admittance implementation.
        self.enable_feature_admittance = self.controller_name == "cA"
        self.enable_cartesian_admittance = self.controller_name == "cB"
        self.feature_admittance_mass = to_vec6(
            self.feature_admittance_mass, "feature_admittance_mass"
        )
        self.feature_admittance_damping = to_vec6(
            self.feature_admittance_damping, "feature_admittance_damping"
        )
        self.feature_admittance_stiffness = to_vec6(
            self.feature_admittance_stiffness, "feature_admittance_stiffness"
        )
        self.feature_admittance_force = to_vec6(
            self.feature_admittance_force, "feature_admittance_force"
        )
        self.feature_admittance_Bc_inv = to_vec6(
            self.feature_admittance_Bc_inv, "feature_admittance_Bc_inv"
        )
        self.feature_admittance_max_offset = np.maximum(
            to_vec6(self.feature_admittance_max_offset, "feature_admittance_max_offset"), 0.0
        )
        self.feature_admittance_max_velocity = np.maximum(
            to_vec6(self.feature_admittance_max_velocity, "feature_admittance_max_velocity"), 0.0
        )
        self.feature_admittance_max_acceleration = np.maximum(
            to_vec6(
                self.feature_admittance_max_acceleration,
                "feature_admittance_max_acceleration",
            ),
            0.0,
        )
        self.feature_admittance_mass = np.maximum(self.feature_admittance_mass, 1e-9)
        self.feature_admittance_damping = np.maximum(self.feature_admittance_damping, 0.0)
        self.feature_admittance_stiffness = np.maximum(self.feature_admittance_stiffness, 0.0)
        self.cartesian_admittance_mass = np.maximum(
            to_vec6(self.cartesian_admittance_mass, "cartesian_admittance_mass"), 1e-9
        )
        self.cartesian_admittance_damping = np.maximum(
            to_vec6(self.cartesian_admittance_damping, "cartesian_admittance_damping"), 0.0
        )
        self.cartesian_admittance_stiffness = np.maximum(
            to_vec6(self.cartesian_admittance_stiffness, "cartesian_admittance_stiffness"), 0.0
        )
        self.cartesian_admittance_wrench_scale = to_vec6(
            self.cartesian_admittance_wrench_scale,
            "cartesian_admittance_wrench_scale",
        )
        self.cartesian_admittance_max_offset = np.maximum(
            to_vec6(self.cartesian_admittance_max_offset, "cartesian_admittance_max_offset"),
            0.0,
        )
        self.cartesian_admittance_max_velocity = np.maximum(
            to_vec6(self.cartesian_admittance_max_velocity, "cartesian_admittance_max_velocity"),
            0.0,
        )
        self.cartesian_admittance_max_acceleration = np.maximum(
            to_vec6(
                self.cartesian_admittance_max_acceleration,
                "cartesian_admittance_max_acceleration",
            ),
            0.0,
        )
        self.enable_rtde_tcp_force = bool(self.enable_rtde_tcp_force)
        self.rtde_tcp_force_frame = str(self.rtde_tcp_force_frame).lower()
        if self.rtde_tcp_force_frame not in ("base", "tcp"):
            raise ValueError("rtde_tcp_force_frame must be 'base' or 'tcp'")
        self.rtde_force_scale = float(self.rtde_force_scale)
        self.enable_static_gravity_compensation = bool(
            self.enable_static_gravity_compensation
        )
        self.static_gravity_model_path = str(self.static_gravity_model_path)
        if self.enable_static_gravity_compensation and not self.static_gravity_model_path:
            raise ValueError(
                "static_gravity_model_path is required when static gravity "
                "compensation is enabled"
            )
        self.rtde_force_bias_samples = max(0, int(self.rtde_force_bias_samples))
        self.rtde_force_lowpass_tau = max(0.0, float(self.rtde_force_lowpass_tau))
        self.rtde_force_deadband = np.maximum(
            to_vec6(self.rtde_force_deadband, "rtde_force_deadband"), 0.0
        )

        if self.accel_limit is None:
            default_accel = float("inf") if self.controller_mode.upper() == "SOPD" else 1.5
            accel_pos_value = default_accel if self.accel_limit_pos is None else self.accel_limit_pos
            accel_rot_value = default_accel if self.accel_limit_rot is None else self.accel_limit_rot
            accel_pos = to_vec3(accel_pos_value, "accel_limit_pos")
            accel_rot = to_vec3(accel_rot_value, "accel_limit_rot")
            self.accel_limit = np.concatenate([accel_pos, accel_rot])
        else:
            self.accel_limit = to_vec6(self.accel_limit, "accel_limit")
        self.accel_limit_pos = self.accel_limit[:3].copy()
        self.accel_limit_rot = self.accel_limit[3:].copy()


@dataclass
class TargetPose:
    name: str
    desired_rotation: np.ndarray
    desired_translation: np.ndarray

    def __post_init__(self):
        self.T_des = np.eye(4)
        self.T_des[:3, :3] = self.desired_rotation
        self.T_des[:3, 3] = self.desired_translation
