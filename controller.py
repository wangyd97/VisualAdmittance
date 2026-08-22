import csv
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2 # pyright: ignore[reportMissingImports]
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl # pyright: ignore[reportMissingImports]
from rtde_receive import RTDEReceiveInterface as RTDEReceive # pyright: ignore[reportMissingImports]

from .config import PBVSConfig, TargetPose
from .geometry import (
    compute_L,
    compute_N,
    compute_b,
    get_tag_3d_corners,
    inv_T,
    project_3d_to_2d,
)
from .gravity_compensation import StaticGravityCompensator
from .pbvs_ekf import PBVSTargetMotionEKF
from .Mathematic import (
    euler_xyz_from_matrix,
    matrix_from_quat,
    matrix_from_rotvec,
    quat_from_matrix,
    rotvec_from_matrix,
)
from .vision import AprilTagEstimator, draw_axis


class FeatureSpaceAdmittance:
    """Discrete feature-space admittance."""

    def __init__(self, cfg: PBVSConfig, dt: float):
        self.cfg = cfg
        self.dt = float(dt)
        self.s_p: Optional[np.ndarray] = None
        self.s_p_dot = np.zeros(6)
        self.s_p_ddot = np.zeros(6)

    def reset(self):
        self.s_p = None
        self.s_p_dot = np.zeros(6)
        self.s_p_ddot = np.zeros(6)

    def compute(
            self,
            s_d: np.ndarray,
            L: np.ndarray,
            external_wrench_cam: Optional[np.ndarray] = None,
            feature_force: Optional[np.ndarray] = None,
            R_base_cam: Optional[np.ndarray] = None) -> np.ndarray:
        s_d = np.asarray(s_d, dtype=float).reshape(6)
        if not self.cfg.enable_feature_admittance:
            self.s_p = s_d.copy()
            self.s_p_dot[:] = 0.0
            self.s_p_ddot[:] = 0.0
            return s_d.copy()

        if self.s_p is None:
            self.s_p = s_d.copy()
            self.s_p_dot[:] = 0.0
            self.s_p_ddot[:] = 0.0

        f_s = np.asarray(self.cfg.feature_admittance_force, dtype=float).reshape(6).copy()
        if feature_force is not None:
            f_s += np.asarray(feature_force, dtype=float).reshape(6)
        if external_wrench_cam is not None:
            wrench = np.asarray(external_wrench_cam, dtype=float).reshape(6)
            if R_base_cam is None:
                raise ValueError(
                    "R_base_cam is required for a camera-frame external wrench"
                )
            R_base_cam = np.asarray(R_base_cam, dtype=float).reshape(3, 3)
            rotate_cam_to_base = np.block([
                [R_base_cam, np.zeros((3, 3))],
                [np.zeros((3, 3)), R_base_cam],
            ])
            bc_inv = np.diag(np.asarray(self.cfg.feature_admittance_Bc_inv, dtype=float).reshape(6))
            f_s += L @ (rotate_cam_to_base @ (bc_inv @ wrench))

        M = np.asarray(self.cfg.feature_admittance_mass, dtype=float).reshape(6)
        D = np.asarray(self.cfg.feature_admittance_damping, dtype=float).reshape(6)
        K = np.asarray(self.cfg.feature_admittance_stiffness, dtype=float).reshape(6)
        # Backward-Euler update of
        # M*s_p_ddot + D*s_p_dot + K*(s_p-s_d) = f_s.
        # The previous explicit update is unstable for the defaults at 60 Hz:
        # D*dt/M = 3.33, so velocity changed sign and grew every sample.
        previous_dot = self.s_p_dot.copy()
        denominator = M / self.dt + D + K * self.dt
        next_dot = (
            (M / self.dt) * previous_dot
            + K * (s_d - self.s_p)
            + f_s
        ) / denominator

        max_accel = np.asarray(
            self.cfg.feature_admittance_max_acceleration, dtype=float
        ).reshape(6)
        next_accel = np.clip(
            (next_dot - previous_dot) / self.dt, -max_accel, max_accel
        )
        next_dot = previous_dot + next_accel * self.dt

        max_velocity = np.asarray(
            self.cfg.feature_admittance_max_velocity, dtype=float
        ).reshape(6)
        next_dot = np.clip(next_dot, -max_velocity, max_velocity)
        next_p = self.s_p + next_dot * self.dt

        max_offset = np.asarray(
            self.cfg.feature_admittance_max_offset, dtype=float
        ).reshape(6)
        offset = next_p - s_d
        clipped_offset = np.clip(offset, -max_offset, max_offset)
        offset_saturated = np.abs(clipped_offset - offset) > 1e-12
        next_p = s_d + clipped_offset
        next_dot[offset_saturated] = 0.0

        if not (np.all(np.isfinite(next_p)) and np.all(np.isfinite(next_dot))):
            self.reset()
            raise FloatingPointError("Non-finite feature-admittance state")

        self.s_p_ddot = (next_dot - previous_dot) / self.dt
        self.s_p_dot = next_dot
        self.s_p = next_p

        qv_norm = float(np.linalg.norm(self.s_p[3:6]))
        if qv_norm > 1.0:
            self.s_p[3:6] /= qv_norm + 1e-12
            self.s_p_dot[3:6] = 0.0
        return self.s_p.copy()


class CartesianSpaceAdmittance:
    """Base-frame Cartesian admittance for a camera-pose offset."""

    def __init__(self, cfg: PBVSConfig, dt: float):
        self.cfg = cfg
        self.dt = float(dt)
        self.offset = np.zeros(6)
        self.velocity = np.zeros(6)
        self.acceleration = np.zeros(6)

    def reset(self):
        self.offset = np.zeros(6)
        self.velocity = np.zeros(6)
        self.acceleration = np.zeros(6)

    def compute(self, external_wrench_base: Optional[np.ndarray]) -> np.ndarray:
        if not self.cfg.enable_cartesian_admittance:
            self.reset()
            return self.offset.copy()

        wrench = np.zeros(6)
        if external_wrench_base is not None:
            wrench = np.asarray(external_wrench_base, dtype=float).reshape(6)
            if not np.all(np.isfinite(wrench)):
                wrench = np.zeros(6)
        wrench *= np.asarray(
            self.cfg.cartesian_admittance_wrench_scale, dtype=float
        ).reshape(6)

        mass = np.asarray(
            self.cfg.cartesian_admittance_mass, dtype=float
        ).reshape(6)
        damping = np.asarray(
            self.cfg.cartesian_admittance_damping, dtype=float
        ).reshape(6)
        stiffness = np.asarray(
            self.cfg.cartesian_admittance_stiffness, dtype=float
        ).reshape(6)

        # Backward-Euler update of
        # M*x_ddot + D*x_dot + K*x = wrench, where x is a base-frame
        # [translation, rotation-vector] offset of the desired camera pose.
        previous_velocity = self.velocity.copy()
        denominator = mass / self.dt + damping + stiffness * self.dt
        next_velocity = (
            (mass / self.dt) * previous_velocity
            - stiffness * self.offset
            + wrench
        ) / denominator

        max_acceleration = np.asarray(
            self.cfg.cartesian_admittance_max_acceleration, dtype=float
        ).reshape(6)
        next_acceleration = np.clip(
            (next_velocity - previous_velocity) / self.dt,
            -max_acceleration,
            max_acceleration,
        )
        next_velocity = previous_velocity + next_acceleration * self.dt

        max_velocity = np.asarray(
            self.cfg.cartesian_admittance_max_velocity, dtype=float
        ).reshape(6)
        next_velocity = np.clip(next_velocity, -max_velocity, max_velocity)
        next_offset = self.offset + next_velocity * self.dt

        max_offset = np.asarray(
            self.cfg.cartesian_admittance_max_offset, dtype=float
        ).reshape(6)
        clipped_offset = np.clip(next_offset, -max_offset, max_offset)
        offset_saturated = np.abs(clipped_offset - next_offset) > 1e-12
        next_velocity[offset_saturated] = 0.0

        if not (
            np.all(np.isfinite(clipped_offset))
            and np.all(np.isfinite(next_velocity))
        ):
            self.reset()
            raise FloatingPointError("Non-finite Cartesian-admittance state")

        self.acceleration = (next_velocity - previous_velocity) / self.dt
        self.velocity = next_velocity
        self.offset = clipped_offset
        return self.offset.copy()


class PBVSController:  
    def __init__(self, robot_ip: str, intrinsics,
                 hand_eye_calib: np.ndarray,
                 config: PBVSConfig = None):
        self.cfg = config or PBVSConfig()
        self.e_T_c = hand_eye_calib
        self.e_R_c = self.e_T_c[0:3, 0:3]
        # e_T_c contains e_p_ce = p_c - p_e expressed in frame e.
        # Use p_ec = p_e - p_c to match the manuscript convention.
        self.e_p_ec = -self.e_T_c[:3, 3]
        self.rtde_freq = 1000.0
        self.dt = 1.0 / self.rtde_freq

        self.rtde_c = RTDEControl(robot_ip, self.rtde_freq)
        receive_vars = ["actual_TCP_pose", "actual_TCP_speed", "actual_TCP_force"]
        self.rtde_r = RTDEReceive(
            robot_ip, self.rtde_freq,
            receive_vars,
            True, False, 1000
        )
        self.estimator = AprilTagEstimator(self.cfg, intrinsics)

        self.targets: List[TargetPose] = []
        self.cur_target: Optional[TargetPose] = None
        self.cur_target_idx = 0

        self._last_accel_saturated = False
        self._last_command_timestamp: Optional[float] = None
        self._instant_control_hz = float("nan")
        self._last_loop_timestamp: Optional[float] = None
        self._last_visual_measurement_timestamp: Optional[float] = None
        self._last_force_control_timestamp: Optional[float] = None
        self._instant_total_hz = float("nan")
        self._instant_vision_hz = float("nan")
        self._instant_force_control_hz = float("nan")

        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)

        self._feature_admittance = FeatureSpaceAdmittance(self.cfg, self.dt)
        self._cartesian_admittance = CartesianSpaceAdmittance(self.cfg, self.dt)
        self._pbvs_ekf = PBVSTargetMotionEKF(self.cfg)
        self._error_log: list = []
        self._t0: float = 0.0
        self._frame_idx = 0

        self._last_u_c = np.zeros(6)
        self._last_R_base_cam: Optional[np.ndarray] = None
        self._last_detection = None
        self._last_c_q_oc: Optional[np.ndarray] = None
        self._last_s = np.full(6, float("nan"))
        self._last_s_p = np.full(6, float("nan"))
        self._last_s_d = np.full(6, float("nan"))
        self._last_s_dot = np.zeros(6)
        self._last_s_p_dot = np.zeros(6)
        self._last_s_p_ddot = np.zeros(6)
        self._last_visual_measurement = np.full(6, float("nan"))
        self._last_object_twist = np.zeros(6)
        self._last_object_acceleration = np.zeros(6)
        self._last_ekf_measurement_accepted = False
        self._last_cartesian_proxy_quat: Optional[np.ndarray] = None
        self._force_bias = np.zeros(6)
        self._static_gravity_compensator: Optional[StaticGravityCompensator] = None
        if self.cfg.enable_static_gravity_compensation:
            self._static_gravity_compensator = StaticGravityCompensator.load(
                self.cfg.static_gravity_model_path
            )
            print(
                "Static gravity compensation loaded: "
                f"{self.cfg.static_gravity_model_path}"
            )
        self._force_bias_ready = self.cfg.rtde_force_bias_samples <= 0
        self._force_bias_count = 0
        self._force_zero_pending = False
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp_raw = np.full(6, float("nan"))
        self._last_static_wrench = np.zeros(6)
        self._last_wrench_static_compensated = np.full(6, float("nan"))
        self._last_wrench_tcp = np.full(6, float("nan"))
        self._last_wrench_cam = np.full(6, float("nan"))
        self._s_lowpass: Optional[np.ndarray] = None
        self._tcp_pose_cmd_est: Optional[np.ndarray] = None
        self._R_base_cam_lowpass: Optional[np.ndarray] = None
        self._vision_lock = threading.Lock()
        self._vision_stop_event = threading.Event()
        self._latest_vision_sample = None
        self._vision_error: Optional[BaseException] = None
        self._display_lock = threading.Lock()
        self._display_update_event = threading.Event()
        self._display_stop_event = threading.Event()
        self._display_quit_event = threading.Event()
        self._force_zero_event = threading.Event()
        self._latest_display_snapshot = None
        self._display_error: Optional[BaseException] = None
        self._last_vision_age = float("inf")

    def set_targets(self, targets: List[TargetPose]):
        self.targets = targets

    def _switch_target(self, idx: int) -> bool:
        if idx >= len(self.targets):
            return False
        self.cur_target_idx = idx
        self.cur_target = self.targets[idx]
        self._reset_controller_state()
        return True

    def _reset_controller_state(self):
        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)
        self._feature_admittance.reset()
        self._cartesian_admittance.reset()
        self._pbvs_ekf.reset()
        self._last_u_c = np.zeros(6)
        self._last_R_base_cam = None
        self._last_detection = None
        self._last_c_q_oc = None
        self._last_s = np.full(6, float("nan"))
        self._last_s_p = np.full(6, float("nan"))
        self._last_s_d = np.full(6, float("nan"))
        self._last_s_dot = np.zeros(6)
        self._last_s_p_dot = np.zeros(6)
        self._last_s_p_ddot = np.zeros(6)
        self._last_visual_measurement = np.full(6, float("nan"))
        self._last_object_twist = np.zeros(6)
        self._last_object_acceleration = np.zeros(6)
        self._last_ekf_measurement_accepted = False
        self._last_cartesian_proxy_quat = None
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp_raw = np.full(6, float("nan"))
        self._last_static_wrench = np.zeros(6)
        self._last_wrench_static_compensated = np.full(6, float("nan"))
        self._last_wrench_tcp = np.full(6, float("nan"))
        self._last_wrench_cam = np.full(6, float("nan"))
        self._s_lowpass = None
        self._tcp_pose_cmd_est = None
        self._R_base_cam_lowpass = None
        self._last_accel_saturated = False
        self._last_command_timestamp = None
        self._instant_control_hz = float("nan")
        self._last_loop_timestamp = None
        self._last_visual_measurement_timestamp = None
        self._last_force_control_timestamp = None
        self._instant_total_hz = float("nan")
        self._instant_vision_hz = float("nan")
        self._instant_force_control_hz = float("nan")
        self._last_vision_age = float("inf")

    def _desired_T(self) -> np.ndarray:
        if self.cur_target is not None:
            return self.cur_target.T_des
        return np.eye(4)

    @staticmethod
    def _nearer_quat(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        q_ref = np.asarray(q_ref, dtype=float).reshape(4)
        q = np.asarray(q, dtype=float).reshape(4).copy()
        if float(q_ref @ q) < 0.0:
            q = -q
        return q

    def _make_c_q_oc_continuous(self, c_q_oc: np.ndarray, update: bool = True) -> np.ndarray:
        c_q_oc = np.asarray(c_q_oc, dtype=float).reshape(4).copy()
        if self._last_c_q_oc is not None:
            c_q_oc = self._nearer_quat(self._last_c_q_oc, c_q_oc)
        if update:
            self._last_c_q_oc = c_q_oc.copy()
        return c_q_oc

    def _compute_features(self, T_current: np.ndarray):
        T_des = self._desired_T()
        q_des = quat_from_matrix(T_des[:3, :3])
        c_q_oc = quat_from_matrix(T_current[:3, :3])
        c_q_oc = self._make_c_q_oc_continuous(c_q_oc)
        c_p_oc = T_current[:3, 3]
        s_d = np.concatenate([T_des[:3, 3], q_des[1:4]])
        s = np.concatenate([c_p_oc, c_q_oc[1:4]])
        return s, s_d

    def _pose_to_ekf_measurement(self, T_current: np.ndarray) -> np.ndarray:
        """Convert c_T_o to the six PBVS states used by the EKF."""
        quaternion = quat_from_matrix(T_current[:3, :3])
        # The project reconstructs qw as the positive square root from qv.
        # Keep measurements on that same quaternion hemisphere.
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        return np.concatenate([T_current[:3, 3], quaternion[1:4]])

    def _ekf_feature_to_pose(self, feature: np.ndarray) -> np.ndarray:
        position, quaternion = self._feature_to_pose_parts(feature)
        transform = np.eye(4)
        transform[:3, :3] = matrix_from_quat(quaternion)
        transform[:3, 3] = position
        return transform

    def _filter_feature_s(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float).reshape(6)
        if not self.cfg.enable_feature_lowpass or self.cfg.feature_lowpass_tau <= 0.0:
            self._s_lowpass = s.copy()
            return s.copy()
        if self._s_lowpass is None:
            self._s_lowpass = s.copy()
            return s.copy()

        tau = self.cfg.feature_lowpass_tau
        beta = tau / (tau + self.dt)
        s_filtered = beta * self._s_lowpass + (1.0 - beta) * s
        qv_norm = float(np.linalg.norm(s_filtered[3:6]))
        if qv_norm > 1.0:
            s_filtered[3:6] /= qv_norm + 1e-12
        self._s_lowpass = s_filtered.copy()
        return s_filtered

    @staticmethod
    def _feature_to_pose_parts(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c_p_oc = np.asarray(s[:3], dtype=float).copy()
        qv = np.asarray(s[3:6], dtype=float).copy()
        qv_norm_sq = float(qv @ qv)
        if qv_norm_sq > 1.0:
            qv /= np.sqrt(qv_norm_sq + 1e-12)
            qv_norm_sq = float(qv @ qv)
        q0 = np.sqrt(max(0.0, 1.0 - qv_norm_sq))
        return c_p_oc, np.array([q0, qv[0], qv[1], qv[2]], dtype=float)

    def _compute_u_dot_c(
            self,
            alpha_c: np.ndarray,
            c_q_oc: np.ndarray,
            c_p_oc: np.ndarray,
            L_inv: np.ndarray,
            u_c: np.ndarray,
            R_base_cam: np.ndarray,
            object_twist: np.ndarray,
            object_acceleration: np.ndarray) -> np.ndarray:
        """Camera acceleration including estimated target-motion feedforward."""
        object_twist = np.asarray(object_twist, dtype=float).reshape(6)
        object_acceleration = np.asarray(
            object_acceleration, dtype=float
        ).reshape(6)
        N = compute_N(c_q_oc, R_base_cam)
        b = compute_b(
            c_p_oc, c_q_oc, u_c, object_twist, R_base_cam
        )
        # s_ddot = L*u_dot_c + N*u_o_dot + b.
        return L_inv @ (alpha_c - N @ object_acceleration - b)

    def _clip_direct_camera_acceleration(self, u_dot_c: np.ndarray) -> tuple[np.ndarray, bool]:
        limits = np.asarray(self.cfg.accel_limit, dtype=float).reshape(6)
        if np.all(np.isinf(limits)):
            return u_dot_c, False

        clipped = np.clip(u_dot_c, -limits, limits)
        saturated = bool(np.any(np.abs(clipped - u_dot_c) > 1e-12))
        return clipped, saturated

    def _clip_tcp_twist(self, v_ctrl: np.ndarray) -> tuple[np.ndarray, bool]:
        """Limit linear/angular vector norms while preserving each direction."""
        clipped = np.asarray(v_ctrl, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(clipped)):
            raise FloatingPointError("Non-finite TCP velocity command")

        saturated = False
        for part, limit in ((slice(0, 3), self.cfg.max_linear_vel),
                            (slice(3, 6), self.cfg.max_angular_vel)):
            limit = float(limit)
            norm = float(np.linalg.norm(clipped[part]))
            if np.isfinite(limit) and norm > limit:
                clipped[part] *= limit / max(norm, 1e-12)
                saturated = True
        return clipped, saturated

    def _compute_cartesian_proxy_feature(
            self,
            T_current: np.ndarray,
            T_base_cam: np.ndarray,
            R_base_cam: np.ndarray,
            external_wrench_cam: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Generate the cB proxy pose, then express it as the PBVS feature."""
        wrench_base = None
        if external_wrench_cam is not None:
            rotate_cam_to_base = np.block([
                [R_base_cam, np.zeros((3, 3))],
                [np.zeros((3, 3)), R_base_cam],
            ])
            wrench_base = rotate_cam_to_base @ np.asarray(
                external_wrench_cam, dtype=float
            ).reshape(6)

        offset = self._cartesian_admittance.compute(wrench_base)

        # Vision provides c_T_o.  Combining it with the measured camera pose
        # gives b_T_o, from which the nominal desired camera pose is obtained.
        T_base_object = T_base_cam @ T_current
        T_base_cam_desired = T_base_object @ inv_T(self._desired_T())

        # The six admittance coordinates are independent base-frame position
        # and rotation-vector offsets.  Apply rotation about the camera origin,
        # rather than rotating its position about the base origin.
        T_base_cam_proxy = T_base_cam_desired.copy()
        T_base_cam_proxy[:3, :3] = (
            matrix_from_rotvec(offset[3:6]) @ T_base_cam_desired[:3, :3]
        )
        T_base_cam_proxy[:3, 3] = T_base_cam_desired[:3, 3] + offset[:3]

        T_proxy = inv_T(T_base_cam_proxy) @ T_base_object
        proxy_quat = quat_from_matrix(T_proxy[:3, :3])
        if self._last_cartesian_proxy_quat is not None:
            proxy_quat = self._nearer_quat(
                self._last_cartesian_proxy_quat, proxy_quat
            )
        self._last_cartesian_proxy_quat = proxy_quat.copy()
        s_p = np.concatenate([T_proxy[:3, 3], proxy_quat[1:4]])

        # For the stationary-object PBVS model, the Cartesian proxy velocity
        # produces feature velocity through the same interaction matrix.  This
        # avoids differentiating noisy AprilTag pose estimates.
        previous_dot = self._last_s_p_dot.copy()
        L_proxy = compute_L(
            T_proxy[:3, 3],
            proxy_quat,
            T_base_cam_proxy[:3, :3],
        )
        s_p_dot = L_proxy @ self._cartesian_admittance.velocity
        self._last_s_p_ddot = (s_p_dot - previous_dot) / self.dt
        self._last_s_p_dot = s_p_dot.copy()
        return s_p, s_p_dot

    def _compute_control(self, T_current: np.ndarray,
                         R_base_cam: np.ndarray,
                         T_base_cam: Optional[np.ndarray] = None,
                         external_wrench_cam: np.ndarray = None,
                         feature_force: np.ndarray = None,
                         object_twist: np.ndarray = None,
                         object_acceleration: np.ndarray = None,
                         measured_camera_twist: np.ndarray = None):
        s, s_d = self._compute_features(T_current)
        if self.cfg.enable_pbvs_ekf:
            # The EKF already filters the measurement and predicts it at the
            # RTDE rate; another low-pass would add avoidable phase lag.
            self._s_lowpass = s.copy()
        else:
            s = self._filter_feature_s(s)
        if object_twist is None:
            object_twist = np.zeros(6)
        if object_acceleration is None:
            object_acceleration = np.zeros(6)
        object_twist = np.asarray(object_twist, dtype=float).reshape(6)
        object_acceleration = np.asarray(
            object_acceleration, dtype=float
        ).reshape(6)
        c_p_oc, c_q_oc = self._feature_to_pose_parts(s)
        L = compute_L(c_p_oc, c_q_oc, R_base_cam)
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.pinv(L)
        
        if self.cfg.controller_name == "cB":
            if T_base_cam is None:
                raise ValueError("T_base_cam is required by controller cB")
            s_p, s_p_dot = self._compute_cartesian_proxy_feature(
                T_current,
                T_base_cam,
                R_base_cam,
                external_wrench_cam,
            )
        else:
            s_p = self._feature_admittance.compute(
                s_d=s_d,
                L=L,
                external_wrench_cam=external_wrench_cam,
                feature_force=feature_force,
                R_base_cam=R_base_cam,
            )
            s_p_dot = self._feature_admittance.s_p_dot.copy()
            self._last_s_p_dot = s_p_dot.copy()
            self._last_s_p_ddot = self._feature_admittance.s_p_ddot.copy()
        e = s_p - s
        u_c = self._last_u_c.copy()
        u_c_feedback = u_c.copy()
        if measured_camera_twist is not None:
            u_c_feedback = np.asarray(
                measured_camera_twist, dtype=float
            ).reshape(6)
        
        s_dot_by_interaction_matrix = L @ u_c_feedback
        N = compute_N(c_q_oc, R_base_cam)
        estimated_s_dot = s_dot_by_interaction_matrix + N @ object_twist
        K = np.diag(self.cfg.kp)
        B = np.diag(self.cfg.kd)
        edot = s_p_dot - estimated_s_dot
        self._last_s = s.copy()
        self._last_s_p = s_p.copy()
        self._last_s_d = s_d.copy()
        self._last_s_dot = estimated_s_dot.copy()
        self._last_object_twist = object_twist.copy()
        self._last_object_acceleration = object_acceleration.copy()
        mode = self.cfg.controller_mode.upper()

        if mode == "SOPD":
            alpha_c = K @ e + B @ edot
            u_dot_c = self._compute_u_dot_c(
                alpha_c,
                c_q_oc,
                c_p_oc,
                L_inv,
                u_c_feedback,
                R_base_cam,
                object_twist,
                object_acceleration,
            )
            u_dot_c, self._last_accel_saturated = self._clip_direct_camera_acceleration(u_dot_c)

        else:
            raise ValueError(f"Unsupported controller mode: {mode}")

        self._u_c_integrated += u_dot_c * self.dt
        if not np.all(np.isfinite(self._u_c_integrated)):
            self._u_c_integrated[:] = 0.0
            raise FloatingPointError("Non-finite integrated camera velocity")
        # Clamp the integrator itself as anti-windup.  A second clamp after the
        # camera-to-TCP transform handles the hand-eye lever-arm contribution.
        self._u_c_integrated, velocity_saturated = self._clip_tcp_twist(
            self._u_c_integrated
        )
        self._last_accel_saturated = (
            self._last_accel_saturated or velocity_saturated
        )

        return self._u_c_integrated, u_dot_c

    def _u_c_to_tcp_twist_base(self, u_c: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        R_base_tcp = matrix_from_rotvec(tcp_pose[3:])
        b_p_ec = R_base_tcp @ self.e_p_ec

        v_c_base = u_c[:3]
        omega_base = u_c[3:]

        v_tcp_base = v_c_base + np.cross(omega_base, b_p_ec)

        return np.concatenate([v_tcp_base, omega_base])

    def _tcp_to_camera_twist_base(
            self, tcp_twist: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        """Convert the measured base-frame TCP twist to the camera origin."""
        tcp_twist = np.asarray(tcp_twist, dtype=float).reshape(6)
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        R_base_tcp = matrix_from_rotvec(tcp_pose[3:])
        b_p_ec = R_base_tcp @ self.e_p_ec
        omega_base = tcp_twist[3:]
        v_camera_base = tcp_twist[:3] - np.cross(omega_base, b_p_ec)
        return np.concatenate([v_camera_base, omega_base])

    def _control_tcp_pose(self, actual_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.cfg.use_commanded_tcp_pose_estimate:
            return actual_pose
        if actual_pose is None:
            return None if self._tcp_pose_cmd_est is None else self._tcp_pose_cmd_est.copy()
        actual_pose = np.asarray(actual_pose, dtype=float).reshape(6)
        if self._tcp_pose_cmd_est is None:
            self._tcp_pose_cmd_est = actual_pose.copy()
        return self._tcp_pose_cmd_est.copy()

    def _advance_commanded_tcp_pose(self, v_cmd: Optional[np.ndarray]):
        if not self.cfg.use_commanded_tcp_pose_estimate or self._tcp_pose_cmd_est is None:
            return
        if v_cmd is None:
            return

        v_cmd = np.asarray(v_cmd, dtype=float).reshape(6)
        pose = self._tcp_pose_cmd_est.copy()
        pose[:3] += v_cmd[:3] * self.dt

        R_base_tcp = matrix_from_rotvec(pose[3:])
        R_next = matrix_from_rotvec(v_cmd[3:] * self.dt) @ R_base_tcp
        pose[3:] = rotvec_from_matrix(R_next)
        self._tcp_pose_cmd_est = pose

    def _filter_R_base_cam(self, R_base_cam: np.ndarray) -> np.ndarray:
        if not self.cfg.enable_Rc_lowpass or self.cfg.Rc_lowpass_tau <= 0.0:
            self._R_base_cam_lowpass = R_base_cam.copy()
            return R_base_cam.copy()
        if self._R_base_cam_lowpass is None:
            self._R_base_cam_lowpass = R_base_cam.copy()
            return R_base_cam.copy()

        alpha = self.dt / (self.cfg.Rc_lowpass_tau + self.dt)
        R_prev = self._R_base_cam_lowpass
        rel_rotvec = rotvec_from_matrix(R_base_cam @ R_prev.T)
        R_filtered = matrix_from_rotvec(alpha * rel_rotvec) @ R_prev
        self._R_base_cam_lowpass = R_filtered.copy()
        return R_filtered

    def _base_camera_pose(
            self, tcp_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return b_T_c and b_R_c using the configured rotation filter."""
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        R_base_tcp = matrix_from_rotvec(tcp_pose[3:])
        R_base_cam = self._filter_R_base_cam(R_base_tcp @ self.e_R_c)

        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = R_base_tcp
        T_base_tcp[:3, 3] = tcp_pose[:3]
        T_base_cam = T_base_tcp @ self.e_T_c
        # The product above contains the unfiltered rotation. Replace only its
        # rotation block so the full pose and R_base_cam stay consistent.
        T_base_cam[:3, :3] = R_base_cam
        return T_base_cam, R_base_cam

    def _vision_worker(self, pipeline):
        """Acquire images and estimate AprilTag pose independently of RTDE."""
        frame_index = 0
        cached_detection = None
        try:
            while not self._vision_stop_event.is_set():
                frames = pipeline.wait_for_frames()
                if self._vision_stop_event.is_set():
                    break
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Own the image memory after the RealSense frame is released.
                image = np.asanyarray(color_frame.get_data()).copy()
                capture_timestamp = time.perf_counter()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                should_detect = (
                    cached_detection is None
                    or frame_index % self.cfg.detect_stride == 0
                )
                detection_updated = False
                if should_detect:
                    cached_detection = self.estimator.detect(gray)
                    detection_updated = True
                    if cached_detection[0] is None:
                        cached_detection = None

                sample = {
                    "sequence": frame_index,
                    "timestamp": capture_timestamp,
                    "processed_timestamp": time.perf_counter(),
                    "image": image,
                    "detection": cached_detection,
                    "detection_updated": detection_updated,
                }
                with self._vision_lock:
                    self._latest_vision_sample = sample
                frame_index += 1
        except BaseException as exc:
            with self._vision_lock:
                self._vision_error = exc

    def _latest_vision(self):
        with self._vision_lock:
            return self._latest_vision_sample, self._vision_error

    def _publish_display_snapshot(
            self, vision_sample, corners, R_cur, t_cur) -> None:
        """Publish one immutable, latest-only snapshot for the GUI thread."""
        target = self.cur_target
        snapshot = {
            "sequence": int(vision_sample["sequence"]),
            # The vision worker owns this immutable image buffer. Rendering
            # starts from image.copy(), so publishing the reference is safe and
            # avoids a full image copy in the RTDE thread.
            "image": vision_sample["image"],
            "corners": None if corners is None else np.asarray(corners).copy(),
            "R_cur": None if R_cur is None else np.asarray(R_cur).copy(),
            "t_cur": None if t_cur is None else np.asarray(t_cur).copy(),
            "K": self.estimator.K.copy(),
            "controller_name": self.cfg.controller_name,
            "target_name": "" if target is None else target.name,
            "desired_T": self._desired_T().copy(),
            "tag_size": float(self.cfg.tag_size),
            "s_p": self._last_s_p.copy(),
            "wrench_cam": self._last_wrench_cam.copy(),
            "enable_force": bool(self.cfg.enable_rtde_tcp_force),
            "force_bias_ready": bool(self._force_bias_ready),
            "force_bias_count": int(self._force_bias_count),
            "force_bias_samples": int(self.cfg.rtde_force_bias_samples),
            "gravity_enabled": self._static_gravity_compensator is not None,
            "accel_saturated": bool(self._last_accel_saturated),
            "total_hz": float(self._instant_total_hz),
            "vision_hz": float(self._instant_vision_hz),
            "force_control_hz": float(self._instant_force_control_hz),
        }
        with self._display_lock:
            # Replacing, rather than appending, guarantees that a slow GUI can
            # never build a backlog or exert back-pressure on control.
            self._latest_display_snapshot = snapshot
            self._display_update_event.set()

    def _display_worker(self) -> None:
        """Render the newest snapshot and own all OpenCV HighGUI calls."""
        last_sequence = -1
        try:
            while not self._display_stop_event.is_set():
                self._display_update_event.wait(timeout=0.10)
                if self._display_stop_event.is_set():
                    break
                with self._display_lock:
                    self._display_update_event.clear()
                    snapshot = self._latest_display_snapshot
                if snapshot is not None and snapshot["sequence"] != last_sequence:
                    last_sequence = snapshot["sequence"]
                    self._visualize(snapshot)
                # Pump window events even if vision has temporarily stopped,
                # so q/z remain responsive while the RTDE chain is running.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._display_quit_event.set()
                elif key == ord('z'):
                    self._force_zero_event.set()
        except BaseException as exc:
            with self._display_lock:
                self._display_error = exc
            self._display_quit_event.set()
        finally:
            try:
                cv2.destroyWindow("SO-PBVS View")
            except cv2.error:
                pass

    def _reset_force_bias(self, manual: bool = False):
        self._force_bias = np.zeros(6)
        self._force_bias_ready = self.cfg.rtde_force_bias_samples <= 0
        self._force_bias_count = 0
        self._force_zero_pending = manual and not self._force_bias_ready
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp = np.zeros(6)
        self._last_wrench_cam = np.full(6, float("nan"))
        if manual:
            zero_target = (
                "gravity-compensated residual"
                if self._static_gravity_compensator is not None
                else "raw wrench"
            )
            if self._force_bias_ready:
                print(f"\nForce sensor {zero_target} bias set to zero (sampling disabled).")
            else:
                print(
                    f"\nForce sensor {zero_target} zeroing started: keep the tool free "
                    f"of external contact "
                    f"for {self.cfg.rtde_force_bias_samples} samples."
                )

    def _read_rtde_tcp_force(self) -> Optional[np.ndarray]:
        if not self.cfg.enable_rtde_tcp_force:
            return None
        try:
            wrench = np.asarray(self.rtde_r.getActualTCPForce(), dtype=float).reshape(6)
        except Exception:
            return None
        self._last_wrench_tcp_raw = wrench.copy()
        return wrench

    def _update_force_bias_and_filter(
            self,
            wrench_raw: Optional[np.ndarray],
            tcp_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if wrench_raw is None:
            self._last_wrench_tcp = np.full(6, float("nan"))
            return None

        wrench_raw = np.asarray(wrench_raw, dtype=float).reshape(6)
        static_wrench = np.zeros(6)
        if self._static_gravity_compensator is not None:
            if tcp_pose is None:
                self._last_wrench_tcp = np.full(6, float("nan"))
                return None
            static_wrench = self._static_gravity_compensator.predict(tcp_pose)
        wrench_static_compensated = wrench_raw - static_wrench
        self._last_static_wrench = static_wrench.copy()
        self._last_wrench_static_compensated = wrench_static_compensated.copy()
        if not self._force_bias_ready:
            # With a model loaded, z estimates only the residual electronics /
            # model bias.  It no longer absorbs gravity at the current attitude.
            self._force_bias += wrench_static_compensated
            self._force_bias_count += 1
            if self._force_bias_count >= self.cfg.rtde_force_bias_samples:
                self._force_bias /= max(self._force_bias_count, 1)
                self._force_bias_ready = True
                self._force_lowpass = np.zeros(6)
                if self._force_zero_pending:
                    self._force_zero_pending = False
                    print("\nForce sensor zeroing complete.")
            self._last_wrench_tcp = np.zeros(6)
            return None

        wrench = self.cfg.rtde_force_scale * (
            wrench_static_compensated - self._force_bias
        )
        deadband = np.asarray(self.cfg.rtde_force_deadband, dtype=float).reshape(6)
        wrench = np.sign(wrench) * np.maximum(np.abs(wrench) - deadband, 0.0)
        if self.cfg.rtde_force_lowpass_tau > 0.0:
            beta = self.cfg.rtde_force_lowpass_tau / (self.cfg.rtde_force_lowpass_tau + self.dt)
            self._force_lowpass = beta * self._force_lowpass + (1.0 - beta) * wrench
            wrench = self._force_lowpass.copy()
        self._last_wrench_tcp = wrench.copy()
        return wrench

    def _tcp_wrench_to_camera(
            self,
            wrench_tcp: Optional[np.ndarray],
            tcp_pose: Optional[np.ndarray],
            R_base_cam: np.ndarray) -> Optional[np.ndarray]:
        if wrench_tcp is None or tcp_pose is None:
            self._last_wrench_cam = np.full(6, float("nan"))
            return None

        wrench_tcp = np.asarray(wrench_tcp, dtype=float).reshape(6)
        R_base_tcp = matrix_from_rotvec(np.asarray(tcp_pose, dtype=float).reshape(6)[3:])

        if self.cfg.rtde_tcp_force_frame == "tcp":
            f_tcp = wrench_tcp[:3]
            m_tcp = wrench_tcp[3:]
            m_cam_origin_tcp = m_tcp + np.cross(self.e_p_ec, f_tcp)
            R_cam_tcp = self.e_R_c.T
            wrench_cam = np.concatenate([
                R_cam_tcp @ f_tcp,
                R_cam_tcp @ m_cam_origin_tcp,
            ])
        else:
            f_base = wrench_tcp[:3]
            m_tcp_base = wrench_tcp[3:]
            b_p_ec = R_base_tcp @ self.e_p_ec
            m_cam_origin_base = m_tcp_base + np.cross(b_p_ec, f_base)
            R_cam_base = R_base_cam.T
            wrench_cam = np.concatenate([
                R_cam_base @ f_base,
                R_cam_base @ m_cam_origin_base,
            ])

        self._last_wrench_cam = wrench_cam.copy()
        return wrench_cam

    def _detect_or_reuse_tag(self, gray_img):
        should_detect = (
            self._last_detection is None
            or self._frame_idx % self.cfg.detect_stride == 0
        )
        if should_detect:
            detection = self.estimator.detect(gray_img)
            if detection[0] is not None:
                self._last_detection = detection
            else:
                self._last_detection = None
            return detection
        return self._last_detection

    def process_step(self, gray_img=None, tcp_pose: np.ndarray = None,
                     log_tcp_pose: np.ndarray = None,
                     external_wrench_cam: np.ndarray = None,
                     rtde_wrench_tcp: np.ndarray = None,
                     feature_force: np.ndarray = None,
                     object_twist: np.ndarray = None,
                     object_acceleration: np.ndarray = None,
                     measured_camera_twist: np.ndarray = None,
                     detection=None,
                     T_base_cam: np.ndarray = None,
                     R_base_cam: np.ndarray = None):
        if detection is None:
            if gray_img is None:
                raise ValueError("gray_img or a precomputed detection is required")
            detection = self._detect_or_reuse_tag(gray_img)
        T_cur, corners, R_cur, t_cur = detection
        
        if T_cur is None:
            self._last_u_c = np.zeros(6)
            self._last_c_q_oc = None
            self._frame_idx += 1
            return None, None, False, None, None, None

        if tcp_pose is None:
            actual_pose = np.array(self.rtde_r.getActualTCPPose())
        else:
            actual_pose = np.asarray(tcp_pose, dtype=float)
        logged_pose = (
            np.asarray(log_tcp_pose, dtype=float)
            if log_tcp_pose is not None else actual_pose
        )
        if T_base_cam is None or R_base_cam is None:
            T_base_cam, R_base_cam = self._base_camera_pose(actual_pose)
        self._last_R_base_cam = R_base_cam.copy()
        if external_wrench_cam is None:
            external_wrench_cam = self._tcp_wrench_to_camera(
                rtde_wrench_tcp,
                actual_pose,
                R_base_cam,
            )

        u_c, u_dot_c = self._compute_control(
            T_cur,
            R_base_cam,
            T_base_cam=T_base_cam,
            external_wrench_cam=external_wrench_cam,
            feature_force=feature_force,
            object_twist=object_twist,
            object_acceleration=object_acceleration,
            measured_camera_twist=measured_camera_twist,
        )
        self._last_u_c = u_c.copy()

        T_err = self._desired_T() @ inv_T(T_cur)
        t_err_vec = T_err[:3, 3]
        err_pos = float(np.linalg.norm(t_err_vec))
        err_rot = float(np.linalg.norm(rotvec_from_matrix(T_err[:3, :3])))
        r_euler = euler_xyz_from_matrix(T_err[:3, :3], degrees=True)

        mode = self.cfg.controller_mode.upper()
        controller_name = self.cfg.controller_name

        v_ctrl = self._u_c_to_tcp_twist_base(u_c, actual_pose)
        v_ctrl, velocity_saturated = self._clip_tcp_twist(v_ctrl)
        self._last_accel_saturated = (
            self._last_accel_saturated or velocity_saturated
        )
        converged_now = (err_pos < self.cfg.pos_threshold
                         and err_rot < self.cfg.rot_threshold)
        if converged_now:
            self.stable_cnt += 1
            if (self.cfg.slow_after_convergence
                    and self.stable_cnt > self.cfg.convergence_slowdown_frames):
                v_ctrl *= self.cfg.convergence_velocity_scale
        else:
            self.stable_cnt = 0
        converged = self.stable_cnt >= self.cfg.stable_frames

        t_now = time.time() - self._t0
        if self.cfg.enable_memory_log:
            des_corners_px = np.full((4, 2), float("nan"))
            if self.cur_target is not None:
                des_corners_3d = get_tag_3d_corners(self.cfg.tag_size, self._desired_T())
                if np.all(des_corners_3d[:, 2] > 0):
                    des_corners_px = project_3d_to_2d(des_corners_3d, self.estimator.K)

            self._error_log.append({
                "t": t_now,
                "err_pos": err_pos * 1000.0,
                "err_rot": float(np.rad2deg(err_rot)),
                "ex": float(t_err_vec[0]*1000), "ey": float(t_err_vec[1]*1000),
                "ez": float(t_err_vec[2]*1000),
                "rx": float(r_euler[0]), "ry": float(r_euler[1]), "rz": float(r_euler[2]),
                "udotc0": float(u_dot_c[0]), "udotc1": float(u_dot_c[1]), "udotc2": float(u_dot_c[2]),
                "udotc3": float(u_dot_c[3]), "udotc4": float(u_dot_c[4]), "udotc5": float(u_dot_c[5]),
                "target": self.cur_target.name,
                "cx0": float(corners[0,0]), "cy0": float(corners[0,1]),
                "cx1": float(corners[1,0]), "cy1": float(corners[1,1]),
                "cx2": float(corners[2,0]), "cy2": float(corners[2,1]),
                "cx3": float(corners[3,0]), "cy3": float(corners[3,1]),
                "dcx0": float(des_corners_px[0,0]), "dcy0": float(des_corners_px[0,1]),
                "dcx1": float(des_corners_px[1,0]), "dcy1": float(des_corners_px[1,1]),
                "dcx2": float(des_corners_px[2,0]), "dcy2": float(des_corners_px[2,1]),
                "dcx3": float(des_corners_px[3,0]), "dcy3": float(des_corners_px[3,1]),
            })
            row = self._error_log[-1]
            row.update({
                "frame_idx": int(self._frame_idx),
                "controller": controller_name,
                "controller_mode": mode,
                "accel_saturated": int(self._last_accel_saturated),
                "control_hz": float(self._instant_control_hz),
                "total_hz": float(self._instant_total_hz),
                "vision_hz": float(self._instant_vision_hz),
                "force_control_hz": float(self._instant_force_control_hz),
                "vision_age_ms": float(self._last_vision_age * 1000.0),
                "ekf_enabled": int(self.cfg.enable_pbvs_ekf),
                "ekf_measurement_accepted": int(
                    self._last_ekf_measurement_accepted
                ),
                "ekf_innovation_norm": float(
                    self._pbvs_ekf.last_innovation_norm
                ),
                "ekf_nis": float(self._pbvs_ekf.last_nis),
                "ekf_motion_compensation_scale": float(
                    self._pbvs_ekf.motion_compensation_scale
                ),
                "tcp_x": float(logged_pose[0]),
                "tcp_y": float(logged_pose[1]),
                "tcp_z": float(logged_pose[2]),
                "tcp_rx": float(logged_pose[3]),
                "tcp_ry": float(logged_pose[4]),
                "tcp_rz": float(logged_pose[5]),
                "ctrl_tcp_x": float(actual_pose[0]),
                "ctrl_tcp_y": float(actual_pose[1]),
                "ctrl_tcp_z": float(actual_pose[2]),
                "ctrl_tcp_rx": float(actual_pose[3]),
                "ctrl_tcp_ry": float(actual_pose[4]),
                "ctrl_tcp_rz": float(actual_pose[5]),
                "force_bias_ready": int(self._force_bias_ready),
            })
            for i in range(6):
                row[f"s{i}"] = float(self._last_s[i])
                row[f"sp{i}"] = float(self._last_s_p[i])
                row[f"sd{i}"] = float(self._last_s_d[i])
                row[f"spdot{i}"] = float(self._last_s_p_dot[i])
                row[f"spddot{i}"] = float(self._last_s_p_ddot[i])
                row[f"sdot{i}"] = float(self._last_s_dot[i])
                row[f"vision_s{i}"] = float(self._last_visual_measurement[i])
                row[f"object_twist{i}"] = float(
                    self._last_object_twist[i]
                )
                row[f"object_acceleration{i}"] = float(
                    self._last_object_acceleration[i]
                )
                row[f"ekf_object_twist_raw{i}"] = float(
                    self._pbvs_ekf.object_twist[i]
                )
                row[f"ekf_object_acceleration_raw{i}"] = float(
                    self._pbvs_ekf.object_acceleration[i]
                )
                row[f"uc{i}"] = float(u_c[i])
                row[f"udotc{i}"] = float(u_dot_c[i])
                row[f"tcp_force_raw{i}"] = float(self._last_wrench_tcp_raw[i])
                row[f"static_wrench{i}"] = float(self._last_static_wrench[i])
                row[f"tcp_force_static_comp{i}"] = float(
                    self._last_wrench_static_compensated[i]
                )
                row[f"tcp_force{i}"] = float(self._last_wrench_tcp[i])
                row[f"cam_force{i}"] = float(self._last_wrench_cam[i])
                row[f"cart_offset{i}"] = float(self._cartesian_admittance.offset[i])
                row[f"cart_velocity{i}"] = float(self._cartesian_admittance.velocity[i])
                row[f"cart_acceleration{i}"] = float(self._cartesian_admittance.acceleration[i])

        self._frame_idx += 1

        return v_ctrl, (err_pos, err_rot), converged, corners, R_cur, t_cur

    def save_log_csv(self):
        if not self._error_log:
            print("No log data to save.")
            return
        if not self.cfg.log_save_path:
            return

        log_path = Path(self.cfg.log_save_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._error_log[0].keys())
        for row in self._error_log[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._error_log)
        print(f"Log saved: {log_path}")

    def run(self, pipeline, init_pose, move_acc: float = 10.0):
        if not self.targets:
            print("Nothing detected!")
            return

        self.rtde_c.moveL(init_pose, 1.2, 1.0)
        self._switch_target(0)
        self._reset_force_bias()

        controller_name = self.cfg.controller_name

        self._t0 = time.time()
        self._error_log.clear()
        self._frame_idx = 0

        with self._vision_lock:
            self._latest_vision_sample = None
            self._vision_error = None
        with self._display_lock:
            self._latest_display_snapshot = None
            self._display_error = None
        self._vision_stop_event.clear()
        self._display_stop_event.clear()
        self._display_update_event.clear()
        self._display_quit_event.clear()
        self._force_zero_event.clear()
        vision_thread = threading.Thread(
            target=self._vision_worker,
            args=(pipeline,),
            name="AprilTagVision",
            daemon=True,
        )
        vision_thread.start()
        display_thread = None
        if self.cfg.enable_visualization:
            display_thread = threading.Thread(
                target=self._display_worker,
                name="OpenCVDisplay",
                daemon=True,
            )
            display_thread.start()

        last_vision_sequence = -1
        latest_target_timestamp: Optional[float] = None
        latest_T_base_object: Optional[np.ndarray] = None
        latest_corners = None
        command_active = False

        try:
            while True:
                loop_timestamp = time.perf_counter()
                if self._last_loop_timestamp is not None:
                    self._instant_total_hz = 1.0 / max(
                        loop_timestamp - self._last_loop_timestamp, 1e-9
                    )
                self._last_loop_timestamp = loop_timestamp

                now = time.time()
                if self.cfg.max_runtime > 0 and (now - self._t0) > self.cfg.max_runtime:
                    print("\nMax runtime reached; stopping.")
                    break
                with self._display_lock:
                    display_error = self._display_error
                if display_error is not None:
                    raise RuntimeError("Display worker failed") from display_error
                if self._display_quit_event.is_set():
                    break
                if self._force_zero_event.is_set():
                    self._force_zero_event.clear()
                    if self.cfg.enable_rtde_tcp_force:
                        self._reset_force_bias(manual=True)
                    else:
                        print(
                            "\nForce sensor zeroing unavailable: "
                            "RTDE force input is disabled."
                        )

                t_start = self.rtde_c.initPeriod()
                try:
                    tcp_pose_now = np.array(self.rtde_r.getActualTCPPose())
                except Exception:
                    tcp_pose_now = None
                try:
                    tcp_speed_now = np.array(self.rtde_r.getActualTCPSpeed())
                except Exception:
                    tcp_speed_now = None
                tcp_pose_ctrl = self._control_tcp_pose(tcp_pose_now)

                T_base_cam = None
                R_base_cam = None
                if tcp_pose_ctrl is not None:
                    T_base_cam, R_base_cam = self._base_camera_pose(tcp_pose_ctrl)

                camera_twist_actual = self._last_u_c.copy()
                if tcp_pose_now is not None and tcp_speed_now is not None:
                    camera_twist_actual = self._tcp_to_camera_twist_base(
                        tcp_speed_now, tcp_pose_now
                    )

                control_timestamp = time.perf_counter()
                if (
                    self.cfg.enable_pbvs_ekf
                    and self._pbvs_ekf.initialized
                    and R_base_cam is not None
                ):
                    self._pbvs_ekf.predict(
                        camera_twist_actual,
                        R_base_cam,
                        control_timestamp,
                    )

                wrench_raw = self._read_rtde_tcp_force()
                rtde_wrench_tcp = self._update_force_bias_and_filter(
                    wrench_raw, tcp_pose_now
                )

                vision_sample, vision_error = self._latest_vision()
                if vision_error is not None:
                    raise RuntimeError("Vision worker failed") from vision_error

                new_vision_sample = (
                    vision_sample is not None
                    and vision_sample["sequence"] != last_vision_sequence
                )
                self._last_ekf_measurement_accepted = False
                if new_vision_sample:
                    last_vision_sequence = vision_sample["sequence"]
                    detection = vision_sample["detection"]
                    if vision_sample["detection_updated"]:
                        if detection is not None and T_base_cam is not None:
                            measurement_timestamp = vision_sample["timestamp"]
                            if self._last_visual_measurement_timestamp is not None:
                                self._instant_vision_hz = 1.0 / max(
                                    measurement_timestamp
                                    - self._last_visual_measurement_timestamp,
                                    1e-9,
                                )
                            self._last_visual_measurement_timestamp = (
                                measurement_timestamp
                            )
                            if self.cfg.enable_pbvs_ekf:
                                measurement = self._pose_to_ekf_measurement(
                                    detection[0]
                                )
                                self._last_visual_measurement = measurement.copy()
                                accepted = self._pbvs_ekf.update(
                                    measurement, measurement_timestamp
                                )
                                self._last_ekf_measurement_accepted = accepted
                                if accepted:
                                    # A first measurement initializes the EKF
                                    # at capture time. Bring it to this RTDE
                                    # cycle immediately.
                                    self._pbvs_ekf.predict(
                                        camera_twist_actual,
                                        R_base_cam,
                                        control_timestamp,
                                    )
                                    latest_target_timestamp = measurement_timestamp
                                    latest_corners = detection[1].copy()
                            else:
                                latest_T_base_object = T_base_cam @ detection[0]
                                latest_target_timestamp = measurement_timestamp
                                latest_corners = detection[1].copy()

                if latest_target_timestamp is None:
                    self._last_vision_age = float("inf")
                else:
                    self._last_vision_age = max(
                        0.0, control_timestamp - latest_target_timestamp
                    )
                if self.cfg.enable_pbvs_ekf:
                    vision_fresh = (
                        self._pbvs_ekf.initialized
                        and T_base_cam is not None
                        and self._last_vision_age <= self.cfg.vision_stale_timeout
                    )
                else:
                    vision_fresh = (
                        latest_T_base_object is not None
                        and T_base_cam is not None
                        and self._last_vision_age <= self.cfg.vision_stale_timeout
                    )

                v_cmd = None
                errs = None
                converged = False
                if vision_fresh:
                    if self.cfg.enable_pbvs_ekf:
                        T_current = self._ekf_feature_to_pose(
                            self._pbvs_ekf.feature
                        )
                        object_twist = (
                            self._pbvs_ekf.object_twist
                            * self._pbvs_ekf.motion_compensation_scale
                        )
                        object_acceleration = (
                            self._pbvs_ekf.object_acceleration
                            * self._pbvs_ekf.motion_compensation_scale
                        )
                    else:
                        T_current = inv_T(T_base_cam) @ latest_T_base_object
                        object_twist = np.zeros(6)
                        object_acceleration = np.zeros(6)
                    detection_for_control = (
                        T_current,
                        latest_corners,
                        T_current[:3, :3],
                        T_current[:3, 3],
                    )
                    v_cmd, errs, converged, _, _, _ = self.process_step(
                        tcp_pose=tcp_pose_ctrl,
                        log_tcp_pose=tcp_pose_now,
                        rtde_wrench_tcp=rtde_wrench_tcp,
                        detection=detection_for_control,
                        T_base_cam=T_base_cam,
                        R_base_cam=R_base_cam,
                        object_twist=object_twist,
                        object_acceleration=object_acceleration,
                        measured_camera_twist=(
                            camera_twist_actual
                            if self.cfg.enable_pbvs_ekf else None
                        ),
                    )
                    if (
                        controller_name in ("cA", "cB")
                        and rtde_wrench_tcp is not None
                    ):
                        force_control_timestamp = time.perf_counter()
                        if self._last_force_control_timestamp is not None:
                            self._instant_force_control_hz = 1.0 / max(
                                force_control_timestamp
                                - self._last_force_control_timestamp,
                                1e-9,
                            )
                        self._last_force_control_timestamp = (
                            force_control_timestamp
                        )

                if v_cmd is not None:
                    self.rtde_c.speedL(v_cmd, move_acc, self.dt)
                    command_active = True
                    command_timestamp = time.perf_counter()
                    if self._last_command_timestamp is not None:
                        command_period = (
                            command_timestamp - self._last_command_timestamp
                        )
                        self._instant_control_hz = (
                            1.0 / max(command_period, 1e-9)
                        )
                    self._last_command_timestamp = command_timestamp
                    self._advance_commanded_tcp_pose(v_cmd)
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        status = "CONVERGED" if converged else "Running"
                        ep, er = errs
                        sat_s = "SAT" if self._last_accel_saturated else "---"
                        hz_s = (
                            f"{self._instant_control_hz:.1f}Hz"
                            if np.isfinite(self._instant_control_hz)
                            else "--.-Hz"
                        )
                        print(f"\r[{controller_name}] "
                              f"{self.cur_target.name} | "
                              f"P:{ep*1000:.1f}mm R:{np.rad2deg(er):.1f}deg"
                              f" | Ctrl:{hz_s} [{sat_s}] {status}   ",
                              end="", flush=True)
                else:
                    if command_active:
                        self.rtde_c.speedStop()
                        self._u_c_integrated[:] = 0.0
                        self._last_u_c[:] = 0.0
                    if (
                        self.cfg.enable_pbvs_ekf
                        and self._pbvs_ekf.initialized
                        and self._last_vision_age > self.cfg.vision_stale_timeout
                    ):
                        self._pbvs_ekf.reset()
                        self._last_object_twist[:] = 0.0
                        self._last_object_acceleration[:] = 0.0
                    command_active = False
                    self._last_command_timestamp = None
                    self._instant_control_hz = float("nan")
                    self._last_force_control_timestamp = None
                    self._instant_force_control_hz = float("nan")
                    if self._last_vision_age > self.cfg.vision_stale_timeout:
                        self._instant_vision_hz = float("nan")
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        age_s = (
                            f"{self._last_vision_age * 1000.0:.0f}ms"
                            if np.isfinite(self._last_vision_age)
                            else "unavailable"
                        )
                        print(
                            f"\rTag unavailable/stale ({age_s})...          ",
                            end="",
                            flush=True,
                        )
                    self._frame_idx += 1

                if (self.cfg.enable_visualization and new_vision_sample
                        and (vision_sample["sequence"]
                             % self.cfg.visualization_stride == 0)):
                    display_detection = vision_sample["detection"]
                    if display_detection is None:
                        display_corners = display_R = display_t = None
                    else:
                        _, display_corners, display_R, display_t = display_detection
                    self._publish_display_snapshot(
                        vision_sample,
                        display_corners,
                        display_R,
                        display_t,
                    )

                self.rtde_c.waitPeriod(t_start)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.rtde_c.speedStop()
            self._vision_stop_event.set()
            self._display_stop_event.set()
            self._display_update_event.set()
            vision_thread.join(timeout=1.0)
            if display_thread is not None:
                display_thread.join(timeout=1.0)
            self.rtde_c.stopScript()
            print("\nControl stopped.")
            self.save_log_csv()

    def _visualize(self, snapshot):
        img = snapshot["image"]
        corners = snapshot["corners"]
        R_cur = snapshot["R_cur"]
        t_cur = snapshot["t_cur"]
        vis = img.copy()
        K = snapshot["K"]
        controller_name = snapshot["controller_name"]
        target_name = snapshot["target_name"]
        desired_T = snapshot["desired_T"]
        tag_size = snapshot["tag_size"]
        s_p = snapshot["s_p"]
        wrench_cam = snapshot["wrench_cam"]
        enable_force = snapshot["enable_force"]
        force_bias_ready = snapshot["force_bias_ready"]
        force_bias_count = snapshot["force_bias_count"]
        force_bias_samples = snapshot["force_bias_samples"]
        gravity_enabled = snapshot["gravity_enabled"]
        h_img, w_img = vis.shape[:2]
        image_labels = []

        if corners is not None and R_cur is not None:
            pts = corners.astype(int)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1) % 4]), (0, 255, 0), 2)
            image_labels.append(
                ("Current", pts[0], (8, -8), 0.45, (0, 255, 0))
            )
            ori = draw_axis(vis, K, R_cur, t_cur, length=0.03)
            if ori is not None:
                eu = euler_xyz_from_matrix(R_cur, degrees=True)
                image_labels.append((
                    f"T:[{t_cur[0]:.3f},{t_cur[1]:.3f},{t_cur[2]:.3f}]",
                    ori, (10, 5), 0.35, (0, 255, 0),
                ))
                image_labels.append((
                    f"R:[{eu[0]:.1f},{eu[1]:.1f},{eu[2]:.1f}]deg",
                    ori, (10, 18), 0.35, (0, 255, 0),
                ))

            force_cam = np.asarray(wrench_cam[:3], dtype=float)
            force_norm = float(np.linalg.norm(force_cam))
            force_valid = (
                enable_force
                and force_bias_ready
                and np.all(np.isfinite(force_cam))
            )
            if force_valid and force_norm > 1e-9:
                # Draw a constant-length direction vector. Its label reports
                # the measured magnitude, so large forces do not leave the view.
                force_color = (0, 255, 255)
                arrow_length_m = 0.06
                arrow_start_3d = np.asarray(t_cur, dtype=float).reshape(3)
                arrow_end_3d = (
                    arrow_start_3d
                    + arrow_length_m * force_cam / force_norm
                )
                arrow_3d = np.vstack([arrow_start_3d, arrow_end_3d])
                if np.all(arrow_3d[:, 2] > 0):
                    arrow_2d = project_3d_to_2d(arrow_3d, K).astype(int)
                    arrow_start = tuple(arrow_2d[0])
                    arrow_end = tuple(arrow_2d[1])
                    if np.linalg.norm(arrow_2d[1] - arrow_2d[0]) >= 3.0:
                        cv2.arrowedLine(
                            vis, arrow_start, arrow_end,
                            force_color, 3, cv2.LINE_AA, tipLength=0.25,
                        )
                    else:
                        # A force nearly parallel to the optical axis has
                        # almost no 2-D projection: x means +z (into image),
                        # dot means -z (towards the camera).
                        cv2.circle(vis, arrow_start, 7, force_color, 2, cv2.LINE_AA)
                        if force_cam[2] >= 0.0:
                            cv2.line(vis,
                                     (arrow_start[0] - 4, arrow_start[1] - 4),
                                     (arrow_start[0] + 4, arrow_start[1] + 4),
                                     force_color, 2, cv2.LINE_AA)
                            cv2.line(vis,
                                     (arrow_start[0] - 4, arrow_start[1] + 4),
                                     (arrow_start[0] + 4, arrow_start[1] - 4),
                                     force_color, 2, cv2.LINE_AA)
                        else:
                            cv2.circle(vis, arrow_start, 3, force_color, -1, cv2.LINE_AA)
                    image_labels.append((
                        f"F_cam {force_norm:.1f} N",
                        arrow_2d[1], (7, -7), 0.45, force_color,
                    ))

        if target_name:
            Td = desired_T
            Rd, td = Td[:3, :3], Td[:3, 3]
            c3d = get_tag_3d_corners(tag_size, Td)
            if np.all(c3d[:, 2] > 0):
                c2d = project_3d_to_2d(c3d, K)
                in_v = np.all((c2d[:, 0] >= -50) & (c2d[:, 0] < w_img+50) &
                              (c2d[:, 1] >= -50) & (c2d[:, 1] < h_img+50))
                if in_v:
                    pd2 = c2d.astype(int)
                    for i in range(4):
                        self._dashed_line(vis, tuple(pd2[i]), tuple(pd2[(i+1) % 4]),
                                          (255, 60, 60), 2, 10)
                    image_labels.append(
                        ("Desired", pd2[0], (8, -8), 0.45, (255, 60, 60))
                    )
                    draw_axis(vis, K, Rd, td, length=0.03)

        if np.all(np.isfinite(s_p)):
            p_p, q_p = self._feature_to_pose_parts(s_p)
            T_p = np.eye(4)
            T_p[:3, :3] = matrix_from_quat(q_p)
            T_p[:3, 3] = p_p
            proxy_corners_3d = get_tag_3d_corners(tag_size, T_p)
            if np.all(proxy_corners_3d[:, 2] > 0):
                proxy_corners_2d = project_3d_to_2d(proxy_corners_3d, K)
                in_view = np.all(
                    (proxy_corners_2d[:, 0] >= -50)
                    & (proxy_corners_2d[:, 0] < w_img + 50)
                    & (proxy_corners_2d[:, 1] >= -50)
                    & (proxy_corners_2d[:, 1] < h_img + 50)
                )
                if in_view:
                    proxy_color = (0, 165, 255)
                    proxy_points = proxy_corners_2d.astype(int)
                    for i in range(4):
                        self._dashed_line(
                            vis,
                            tuple(proxy_points[i]),
                            tuple(proxy_points[(i + 1) % 4]),
                            proxy_color,
                            2,
                            6,
                        )
                        cv2.circle(vis, tuple(proxy_points[i]), 3, proxy_color, -1)
                    image_labels.append(
                        ("Proxy", proxy_points[0], (8, -8), 0.45, proxy_color)
                    )

        hud_y = 45
        sat = snapshot["accel_saturated"]

        if not enable_force:
            force_status = "Force-Zero: disabled"
            force_color = (140, 140, 140)
        elif force_bias_ready:
            gravity_status = (
                "Gravity: ON"
                if gravity_enabled
                else "Gravity: OFF"
            )
            force_status = f"Force-Zero: READY [z]  {gravity_status}"
            force_color = (0, 200, 80)
        else:
            force_status = (
                f"Force-Zero: {force_bias_count}/"
                f"{force_bias_samples}"
            )
            force_color = (0, 180, 255)
        force_cam = np.asarray(wrench_cam[:3], dtype=float)
        force_valid = (
            enable_force
            and force_bias_ready
            and np.all(np.isfinite(force_cam))
        )
        if force_valid:
            force_norm = float(np.linalg.norm(force_cam))
            force_text = f"|F_cam|: {force_norm:.2f} N"
            components_text = (
                f"Fx:{force_cam[0]:+.2f}  Fy:{force_cam[1]:+.2f}  "
                f"Fz:{force_cam[2]:+.2f} N"
            )
            measurement_color = (0, 255, 255)
        else:
            force_text = "|F_cam|: unavailable"
            components_text = "Fx:--  Fy:--  Fz:--"
            measurement_color = (140, 140, 140)
        total_hz_text = (
            f"{snapshot['total_hz']:.1f}"
            if np.isfinite(snapshot["total_hz"]) else "--.-"
        )
        vision_hz_text = (
            f"{snapshot['vision_hz']:.1f}"
            if np.isfinite(snapshot["vision_hz"]) else "--.-"
        )
        force_hz_text = (
            f"{snapshot['force_control_hz']:.1f}"
            if np.isfinite(snapshot["force_control_hz"]) else "--.-"
        )
        frequency_text = (
            f"Hz  Total:{total_hz_text}  Vision:{vision_hz_text}  "
            f"Force:{force_hz_text}"
        )
        display = cv2.rotate(vis, cv2.ROTATE_180)

        for text, anchor, offset, scale, color in image_labels:
            anchor = np.asarray(anchor, dtype=int).reshape(2)
            origin = (
                int(w_img - 1 - anchor[0] + offset[0]),
                int(h_img - 1 - anchor[1] + offset[1]),
            )
            cv2.putText(display, text, origin,
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 1, cv2.LINE_AA)

        cv2.putText(display,
                    f"[{controller_name}] {target_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Accel-SAT: {'YES' if sat else 'no '}",
                    (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255) if sat else (0, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(display, force_status, (10, hud_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    force_color, 1, cv2.LINE_AA)
        cv2.putText(display, force_text, (10, hud_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    measurement_color, 1, cv2.LINE_AA)
        cv2.putText(display, components_text, (10, hud_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    measurement_color, 1, cv2.LINE_AA)
        cv2.putText(display, frequency_text, (10, hud_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("SO-PBVS View", display)

    def _dashed_line(self, img, pt1, pt2, color, thickness, dash_len):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        dashes = max(int(dist / dash_len), 1)
        for i in range(dashes):
            s = (int(pt1[0] + (pt2[0]-pt1[0]) * i / dashes),
                 int(pt1[1] + (pt2[1]-pt1[1]) * i / dashes))
            e = (int(pt1[0] + (pt2[0]-pt1[0]) * (i+0.5) / dashes),
                 int(pt1[1] + (pt2[1]-pt1[1]) * (i+0.5) / dashes))
            cv2.line(img, s, e, color, thickness)
