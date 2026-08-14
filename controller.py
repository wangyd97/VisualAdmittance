import csv
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
    compute_b,
    get_tag_3d_corners,
    inv_T,
    project_3d_to_2d,
)
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
        # M*s_p_ddot + D*s_p_dot + K*(s_p-s_d) = -f_s.
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
        self.rtde_freq = 200.0
        self.dt = 1.0 / self.rtde_freq

        self.rtde_c = RTDEControl(robot_ip, self.rtde_freq)
        receive_vars = ["actual_TCP_pose", "actual_TCP_force"]
        self.rtde_r = RTDEReceive(
            robot_ip, self.rtde_freq,
            receive_vars,
            True, False, 60
        )
        self.estimator = AprilTagEstimator(self.cfg, intrinsics)

        self.targets: List[TargetPose] = []
        self.cur_target: Optional[TargetPose] = None
        self.cur_target_idx = 0

        self._last_accel_saturated = False

        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)

        self._feature_admittance = FeatureSpaceAdmittance(self.cfg, self.dt)
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
        self._force_bias = np.zeros(6)
        self._force_bias_ready = self.cfg.rtde_force_bias_samples <= 0
        self._force_bias_count = 0
        self._force_zero_pending = False
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp_raw = np.full(6, float("nan"))
        self._last_wrench_tcp = np.full(6, float("nan"))
        self._last_wrench_cam = np.full(6, float("nan"))
        self._s_lowpass: Optional[np.ndarray] = None
        self._tcp_pose_cmd_est: Optional[np.ndarray] = None
        self._R_base_cam_lowpass: Optional[np.ndarray] = None

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
        self._last_u_c = np.zeros(6)
        self._last_R_base_cam = None
        self._last_detection = None
        self._last_c_q_oc = None
        self._last_s = np.full(6, float("nan"))
        self._last_s_p = np.full(6, float("nan"))
        self._last_s_d = np.full(6, float("nan"))
        self._last_s_dot = np.zeros(6)
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp_raw = np.full(6, float("nan"))
        self._last_wrench_tcp = np.full(6, float("nan"))
        self._last_wrench_cam = np.full(6, float("nan"))
        self._s_lowpass = None
        self._tcp_pose_cmd_est = None
        self._R_base_cam_lowpass = None
        self._last_accel_saturated = False

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

    def _compute_u_dot_c(self, alpha_c: np.ndarray, c_q_oc: np.ndarray,
                  c_p_oc: np.ndarray, L_inv: np.ndarray,
                  u_c: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
        """Camera acceleration for the stationary-object PBVS model."""
        u_o = np.zeros(6)
        b = compute_b(c_p_oc, c_q_oc, u_c, u_o, R_base_cam)
        return L_inv @ (alpha_c - b)

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

    def _compute_control(self, T_current: np.ndarray,
                         R_base_cam: np.ndarray,
                         external_wrench_cam: np.ndarray = None,
                         feature_force: np.ndarray = None):
        s, s_d = self._compute_features(T_current)
        s = self._filter_feature_s(s)
        c_p_oc, c_q_oc = self._feature_to_pose_parts(s)
        L = compute_L(c_p_oc, c_q_oc, R_base_cam)
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.pinv(L)
        
        s_p = self._feature_admittance.compute(
            s_d=s_d,
            L=L,
            external_wrench_cam=external_wrench_cam,
            feature_force=feature_force,
            R_base_cam=R_base_cam,
        )
        s_p_dot = self._feature_admittance.s_p_dot.copy()
        e = s_p - s
        u_c = self._last_u_c.copy()
        
        s_dot_by_interaction_matrix = L @ u_c
        K = np.diag(self.cfg.kp)
        B = np.diag(self.cfg.kd)
        edot = s_p_dot - s_dot_by_interaction_matrix
        self._last_s = s.copy()
        self._last_s_p = s_p.copy()
        self._last_s_d = s_d.copy()
        self._last_s_dot = s_dot_by_interaction_matrix.copy()
        mode = self.cfg.controller_mode.upper()

        if mode == "SOPD":
            alpha_c = K @ e + B @ edot
            u_dot_c = self._compute_u_dot_c(
                alpha_c, c_q_oc, c_p_oc, L_inv, u_c, R_base_cam
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

    def _reset_force_bias(self, manual: bool = False):
        self._force_bias = np.zeros(6)
        self._force_bias_ready = self.cfg.rtde_force_bias_samples <= 0
        self._force_bias_count = 0
        self._force_zero_pending = manual and not self._force_bias_ready
        self._force_lowpass = np.zeros(6)
        self._last_wrench_tcp = np.zeros(6)
        self._last_wrench_cam = np.full(6, float("nan"))
        if manual:
            if self._force_bias_ready:
                print("\nForce sensor bias set to zero (bias sampling disabled).")
            else:
                print(
                    f"\nForce sensor zeroing started: keep the tool unloaded "
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

    def _update_force_bias_and_filter(self, wrench_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if wrench_raw is None:
            self._last_wrench_tcp = np.full(6, float("nan"))
            return None

        wrench_raw = np.asarray(wrench_raw, dtype=float).reshape(6)
        if not self._force_bias_ready:
            self._force_bias += wrench_raw
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

        wrench = self.cfg.rtde_force_scale * (wrench_raw - self._force_bias)
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

    def process_step(self, gray_img, tcp_pose: np.ndarray = None,
                     log_tcp_pose: np.ndarray = None,
                     external_wrench_cam: np.ndarray = None,
                     rtde_wrench_tcp: np.ndarray = None,
                     feature_force: np.ndarray = None):
        T_cur, corners, R_cur, t_cur = self._detect_or_reuse_tag(gray_img)
        
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
        R_base_tcp = matrix_from_rotvec(actual_pose[3:])
        R_base_cam = R_base_tcp @ self.e_R_c
        R_base_cam = self._filter_R_base_cam(R_base_cam)
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
            external_wrench_cam=external_wrench_cam,
            feature_force=feature_force,
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
                row[f"spdot{i}"] = float(self._feature_admittance.s_p_dot[i])
                row[f"spddot{i}"] = float(self._feature_admittance.s_p_ddot[i])
                row[f"sdot{i}"] = float(self._last_s_dot[i])
                row[f"uc{i}"] = float(u_c[i])
                row[f"udotc{i}"] = float(u_dot_c[i])
                row[f"tcp_force_raw{i}"] = float(self._last_wrench_tcp_raw[i])
                row[f"tcp_force{i}"] = float(self._last_wrench_tcp[i])
                row[f"cam_force{i}"] = float(self._last_wrench_cam[i])

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

    def run(self, pipeline, init_pose, move_acc: float = 5.0):
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

        try:
            while True:
                now = time.time()
                if self.cfg.max_runtime > 0 and (now - self._t0) > self.cfg.max_runtime:
                    print("\nMax runtime reached; stopping.")
                    break

                t_start = self.rtde_c.initPeriod()
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                try:
                    tcp_pose_now = np.array(self.rtde_r.getActualTCPPose())
                except Exception:
                    tcp_pose_now = None
                tcp_pose_ctrl = self._control_tcp_pose(tcp_pose_now)
                wrench_raw = self._read_rtde_tcp_force()
                rtde_wrench_tcp = self._update_force_bias_and_filter(wrench_raw)

                v_cmd, errs, converged, corners, R_cur, t_cur = self.process_step(
                    gray,
                    tcp_pose=tcp_pose_ctrl,
                    log_tcp_pose=tcp_pose_now,
                    rtde_wrench_tcp=rtde_wrench_tcp,
                )

                if v_cmd is not None:
                    self.rtde_c.speedL(v_cmd, move_acc, self.dt)
                    self._advance_commanded_tcp_pose(v_cmd)
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        status = "CONVERGED" if converged else "Running"
                        ep, er = errs
                        sat_s = "SAT" if self._last_accel_saturated else "---"
                        print(f"\r[{controller_name}] "
                              f"{self.cur_target.name} | "
                              f"P:{ep*1000:.1f}mm R:{np.rad2deg(er):.1f}deg"
                              f" | [{sat_s}] {status}   ", end="", flush=True)
                else:
                    self.rtde_c.speedStop()
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        print("\rTag lost...                                ", end="", flush=True)

                if (self.cfg.enable_visualization
                        and self._frame_idx % self.cfg.visualization_stride == 0):
                    self._visualize(img, corners, v_cmd, R_cur, t_cur)

                key = cv2.waitKey(1) & 0xFF if self.cfg.enable_visualization else 0
                if key == ord('q'):
                    break
                if key == ord('z'):
                    if self.cfg.enable_rtde_tcp_force:
                        self._reset_force_bias(manual=True)
                    else:
                        print("\nForce sensor zeroing unavailable: RTDE force input is disabled.")

                self.rtde_c.waitPeriod(t_start)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
            print("\nControl stopped.")
            self.save_log_csv()

    def _visualize(self, img, corners, v_cmd, R_cur, t_cur):
        vis = img.copy()
        K = self.estimator.K
        controller_name = self.cfg.controller_name
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

            force_cam = np.asarray(self._last_wrench_cam[:3], dtype=float)
            force_norm = float(np.linalg.norm(force_cam))
            force_valid = (
                self.cfg.enable_rtde_tcp_force
                and self._force_bias_ready
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

        if self.cur_target is not None:
            Td = self._desired_T()
            Rd, td = Td[:3, :3], Td[:3, 3]
            c3d = get_tag_3d_corners(self.cfg.tag_size, Td)
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

        if np.all(np.isfinite(self._last_s_p)):
            p_p, q_p = self._feature_to_pose_parts(self._last_s_p)
            T_p = np.eye(4)
            T_p[:3, :3] = matrix_from_quat(q_p)
            T_p[:3, 3] = p_p
            proxy_corners_3d = get_tag_3d_corners(self.cfg.tag_size, T_p)
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
        sat = self._last_accel_saturated

        if not self.cfg.enable_rtde_tcp_force:
            force_status = "Force-Zero: disabled"
            force_color = (140, 140, 140)
        elif self._force_bias_ready:
            force_status = "Force-Zero: READY [z]"
            force_color = (0, 200, 80)
        else:
            force_status = (
                f"Force-Zero: {self._force_bias_count}/"
                f"{self.cfg.rtde_force_bias_samples}"
            )
            force_color = (0, 180, 255)
        force_cam = np.asarray(self._last_wrench_cam[:3], dtype=float)
        force_valid = (
            self.cfg.enable_rtde_tcp_force
            and self._force_bias_ready
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
                    f"[{controller_name}] {self.cur_target.name if self.cur_target else ''}",
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
