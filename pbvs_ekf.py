"""PBVS EKF for high-rate feature prediction and target-motion estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import compute_L, compute_N


@dataclass
class _Snapshot:
    timestamp: float
    state: np.ndarray
    covariance: np.ndarray
    dt: float
    camera_twist: np.ndarray
    R_base_cam: np.ndarray


class PBVSTargetMotionEKF:
    """EKF with state x=[s, u_o, u_o_dot] in R18.

    It specializes the moving-target idea in Oliva et al. (ICRA 2022) to this
    project's PBVS kinematics:
        s_dot = L(s) u_c + N(s) u_o
        (u_o)_dot = u_o_dot
        (u_o_dot)_dot = 0.
    Both camera twist ``u_c`` and object twist ``u_o`` are expressed in the
    robot base frame.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = np.zeros(18)
        self.covariance = np.eye(18)
        self.initialized = False
        self.timestamp: float | None = None
        self.history: list[_Snapshot] = []
        self.last_innovation_norm = float("nan")
        self.last_nis = float("nan")
        self.last_measurement_accepted = False
        self.accepted_measurement_count = 0

    def reset(self) -> None:
        self.state[:] = 0.0
        self.covariance = np.eye(18)
        self.initialized = False
        self.timestamp = None
        self.history.clear()
        self.last_innovation_norm = float("nan")
        self.last_nis = float("nan")
        self.last_measurement_accepted = False
        self.accepted_measurement_count = 0

    @staticmethod
    def _bounded_quaternion_vector(feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=float).reshape(6).copy()
        norm = float(np.linalg.norm(feature[3:6]))
        if norm >= 1.0:
            feature[3:6] *= (1.0 - 1e-9) / max(norm, 1e-12)
        return feature

    def _clip_motion_states(self, state: np.ndarray) -> None:
        state[:6] = self._bounded_quaternion_vector(state[:6])
        velocity_limit = np.asarray(
            self.cfg.pbvs_ekf_max_object_twist, dtype=float
        ).reshape(6)
        acceleration_limit = np.asarray(
            self.cfg.pbvs_ekf_max_object_acceleration, dtype=float
        ).reshape(6)
        state[6:12] = np.clip(state[6:12], -velocity_limit, velocity_limit)
        state[12:18] = np.clip(
            state[12:18], -acceleration_limit, acceleration_limit
        )

    def initialize(self, measurement: np.ndarray, timestamp: float) -> None:
        measurement = self._bounded_quaternion_vector(measurement)
        self.state[:] = 0.0
        self.state[:6] = measurement
        feature_std = np.asarray(
            self.cfg.pbvs_ekf_measurement_std, dtype=float
        ).reshape(6)
        velocity_std = np.asarray(
            self.cfg.pbvs_ekf_initial_object_twist_std, dtype=float
        ).reshape(6)
        acceleration_std = np.asarray(
            self.cfg.pbvs_ekf_initial_object_acceleration_std, dtype=float
        ).reshape(6)
        self.covariance = np.diag(np.concatenate([
            feature_std ** 2,
            velocity_std ** 2,
            acceleration_std ** 2,
        ]))
        self.initialized = True
        self.timestamp = float(timestamp)
        self.history = [_Snapshot(
            timestamp=float(timestamp),
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            dt=0.0,
            camera_twist=np.zeros(6),
            R_base_cam=np.eye(3),
        )]
        self.last_innovation_norm = 0.0
        self.last_nis = 0.0
        self.last_measurement_accepted = True
        self.accepted_measurement_count = 1

    def _process(self, state: np.ndarray, camera_twist: np.ndarray,
                 R_base_cam: np.ndarray) -> np.ndarray:
        feature = self._bounded_quaternion_vector(state[:6])
        qv = feature[3:6]
        quaternion = np.concatenate(([
            np.sqrt(max(0.0, 1.0 - float(qv @ qv)))
        ], qv))
        interaction = compute_L(feature[:3], quaternion, R_base_cam)
        object_interaction = compute_N(quaternion, R_base_cam)
        derivative = np.zeros(18)
        derivative[:6] = (
            interaction @ camera_twist + object_interaction @ state[6:12]
        )
        derivative[6:12] = state[12:18]
        return derivative

    def _continuous_jacobian(self, state: np.ndarray, camera_twist: np.ndarray,
                             R_base_cam: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((18, 18))
        epsilon = np.asarray(
            self.cfg.pbvs_ekf_jacobian_epsilon, dtype=float
        ).reshape(6)
        for index in range(6):
            step = max(float(epsilon[index]), 1e-9)
            plus = state.copy()
            minus = state.copy()
            plus[index] += step
            minus[index] -= step
            jacobian[:6, index] = (
                self._process(plus, camera_twist, R_base_cam)[:6]
                - self._process(minus, camera_twist, R_base_cam)[:6]
            ) / (2.0 * step)
        feature = self._bounded_quaternion_vector(state[:6])
        qv = feature[3:6]
        quaternion = np.concatenate(([
            np.sqrt(max(0.0, 1.0 - float(qv @ qv)))
        ], qv))
        jacobian[:6, 6:12] = compute_N(quaternion, R_base_cam)
        jacobian[6:12, 12:18] = np.eye(6)
        return jacobian

    def _predict_once(self, state: np.ndarray, covariance: np.ndarray,
                      camera_twist: np.ndarray, R_base_cam: np.ndarray,
                      dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return state.copy(), covariance.copy()
        continuous_jacobian = self._continuous_jacobian(
            state, camera_twist, R_base_cam
        )
        transition = np.eye(18) + continuous_jacobian * dt
        predicted_state = state + self._process(
            state, camera_twist, R_base_cam
        ) * dt
        self._clip_motion_states(predicted_state)

        process_std = np.concatenate([
            np.asarray(self.cfg.pbvs_ekf_process_std_feature, dtype=float),
            np.asarray(self.cfg.pbvs_ekf_process_std_object_twist, dtype=float),
            np.asarray(self.cfg.pbvs_ekf_process_std_object_acceleration, dtype=float),
        ])
        process_covariance = np.diag(process_std ** 2) * dt
        predicted_covariance = (
            transition @ covariance @ transition.T + process_covariance
        )
        predicted_covariance = 0.5 * (
            predicted_covariance + predicted_covariance.T
        )
        return predicted_state, predicted_covariance

    def _predict_interval(self, state: np.ndarray, covariance: np.ndarray,
                          camera_twist: np.ndarray, R_base_cam: np.ndarray,
                          dt: float) -> tuple[np.ndarray, np.ndarray]:
        remaining = max(0.0, float(dt))
        max_step = float(self.cfg.pbvs_ekf_max_prediction_step)
        while remaining > 1e-12:
            step = min(remaining, max_step)
            state, covariance = self._predict_once(
                state, covariance, camera_twist, R_base_cam, step
            )
            remaining -= step
        return state, covariance

    def predict(self, camera_twist: np.ndarray, R_base_cam: np.ndarray,
                timestamp: float) -> bool:
        if not self.initialized or self.timestamp is None:
            return False
        timestamp = float(timestamp)
        dt = timestamp - self.timestamp
        if dt <= 0.0:
            return True
        camera_twist = np.asarray(camera_twist, dtype=float).reshape(6)
        R_base_cam = np.asarray(R_base_cam, dtype=float).reshape(3, 3)
        self.state, self.covariance = self._predict_interval(
            self.state, self.covariance, camera_twist, R_base_cam, dt
        )
        self.timestamp = timestamp
        self.history.append(_Snapshot(
            timestamp=timestamp,
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            dt=dt,
            camera_twist=camera_twist.copy(),
            R_base_cam=R_base_cam.copy(),
        ))
        oldest = timestamp - float(self.cfg.pbvs_ekf_history_duration)
        while len(self.history) > 1 and self.history[1].timestamp < oldest:
            self.history.pop(0)
        return True

    def _measurement_update(self, state: np.ndarray, covariance: np.ndarray,
                            measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        measurement = self._bounded_quaternion_vector(measurement)
        innovation = measurement - state[:6]
        measurement_std = np.asarray(
            self.cfg.pbvs_ekf_measurement_std, dtype=float
        ).reshape(6)
        measurement_covariance = np.diag(measurement_std ** 2)
        innovation_covariance = covariance[:6, :6] + measurement_covariance
        try:
            solved_innovation = np.linalg.solve(
                innovation_covariance, innovation
            )
            gain = np.linalg.solve(
                innovation_covariance, covariance[:6, :]
            ).T
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(innovation_covariance)
            solved_innovation = inverse @ innovation
            gain = covariance[:, :6] @ inverse
        nis = float(innovation @ solved_innovation)
        self.last_innovation_norm = float(np.linalg.norm(innovation))
        self.last_nis = nis
        if nis > float(self.cfg.pbvs_ekf_innovation_gate):
            return state, covariance, False

        updated_state = state + gain @ innovation
        self._clip_motion_states(updated_state)
        observation = np.zeros((6, 18))
        observation[:, :6] = np.eye(6)
        identity_minus = np.eye(18) - gain @ observation
        updated_covariance = (
            identity_minus @ covariance @ identity_minus.T
            + gain @ measurement_covariance @ gain.T
        )
        updated_covariance = 0.5 * (
            updated_covariance + updated_covariance.T
        )
        return updated_state, updated_covariance, True

    def update(self, measurement: np.ndarray, timestamp: float) -> bool:
        """Fuse a possibly delayed pose measurement and replay predictions."""
        measurement = np.asarray(measurement, dtype=float).reshape(6)
        timestamp = float(timestamp)
        if not self.initialized:
            self.initialize(measurement, timestamp)
            return True

        if not self.history:
            state, covariance, accepted = self._measurement_update(
                self.state, self.covariance, measurement
            )
            if accepted:
                self.state, self.covariance = state, covariance
                self.accepted_measurement_count += 1
            self.last_measurement_accepted = accepted
            return accepted

        index = min(
            range(len(self.history)),
            key=lambda i: abs(self.history[i].timestamp - timestamp),
        )
        snapshot = self.history[index]
        state, covariance, accepted = self._measurement_update(
            snapshot.state.copy(), snapshot.covariance.copy(), measurement
        )
        self.last_measurement_accepted = accepted
        if not accepted:
            return False
        self.accepted_measurement_count += 1

        snapshot.state = state.copy()
        snapshot.covariance = covariance.copy()
        for replay_index in range(index + 1, len(self.history)):
            transition_data = self.history[replay_index]
            state, covariance = self._predict_interval(
                state,
                covariance,
                transition_data.camera_twist,
                transition_data.R_base_cam,
                transition_data.dt,
            )
            transition_data.state = state.copy()
            transition_data.covariance = covariance.copy()
        self.state = state.copy()
        self.covariance = covariance.copy()
        return True

    @property
    def feature(self) -> np.ndarray:
        return self.state[:6].copy()

    @property
    def object_twist(self) -> np.ndarray:
        return self.state[6:12].copy()

    @property
    def object_acceleration(self) -> np.ndarray:
        return self.state[12:18].copy()

    @property
    def motion_compensation_scale(self) -> float:
        warmup = int(self.cfg.pbvs_ekf_motion_compensation_warmup_updates)
        ramp = int(self.cfg.pbvs_ekf_motion_compensation_ramp_updates)
        return float(np.clip(
            (self.accepted_measurement_count - warmup) / max(ramp, 1),
            0.0,
            1.0,
        ))
