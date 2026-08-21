"""Offline-identified static wrench compensation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .Mathematic import matrix_from_rotvec


def orientation_features(tcp_pose: np.ndarray) -> np.ndarray:
    """Return affine rotation features [1, vec(R)] for one TCP pose."""
    pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    rotation = matrix_from_rotvec(pose[3:])
    return np.concatenate(([1.0], rotation.reshape(9)))


class StaticGravityCompensator:
    """Predict the no-contact static wrench from the current TCP attitude.

    Every component of a rigid payload's static wrench is affine in the
    entries of the TCP rotation matrix.  This representation also avoids
    relying on an ambiguous RTDE wrench-frame convention.
    """

    MODEL_VERSION = 1

    def __init__(self, coefficients: np.ndarray, metadata: dict | None = None):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (10, 6) or not np.all(np.isfinite(coefficients)):
            raise ValueError("static gravity coefficients must have shape (10, 6)")
        self.coefficients = coefficients.copy()
        self.metadata = dict(metadata or {})

    def predict(self, tcp_pose: np.ndarray) -> np.ndarray:
        return orientation_features(tcp_pose) @ self.coefficients

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "affine_tcp_rotation_static_wrench",
            "version": self.MODEL_VERSION,
            "coefficients": self.coefficients.tolist(),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StaticGravityCompensator":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != "affine_tcp_rotation_static_wrench":
            raise ValueError(f"unsupported static gravity model in {path}")
        if int(payload.get("version", -1)) != cls.MODEL_VERSION:
            raise ValueError(f"unsupported static gravity model version in {path}")
        return cls(payload["coefficients"], payload.get("metadata"))


def fit_static_gravity_model(
        tcp_poses: np.ndarray,
        wrenches: np.ndarray,
        ridge: float = 1e-8,
        robust_iterations: int = 5) -> tuple[StaticGravityCompensator, dict]:
    """Fit an orientation-dependent static wrench model with robust IRLS."""
    tcp_poses = np.asarray(tcp_poses, dtype=float)
    wrenches = np.asarray(wrenches, dtype=float)
    if tcp_poses.ndim != 2 or tcp_poses.shape[1] != 6:
        raise ValueError("tcp_poses must have shape (N, 6)")
    if wrenches.shape != tcp_poses.shape:
        raise ValueError("wrenches must have shape (N, 6)")
    finite = np.all(np.isfinite(tcp_poses), axis=1) & np.all(
        np.isfinite(wrenches), axis=1
    )
    tcp_poses = tcp_poses[finite]
    wrenches = wrenches[finite]
    if len(tcp_poses) < 12:
        raise ValueError("at least 12 distinct static poses are required")

    design = np.vstack([orientation_features(pose) for pose in tcp_poses])
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            f"calibration attitudes are insufficiently diverse (rank {rank}/10)"
        )

    weights = np.ones(len(design))
    regularizer = np.eye(design.shape[1]) * max(float(ridge), 0.0)
    regularizer[0, 0] = 0.0
    coefficients = np.zeros((10, 6))
    for _ in range(max(1, int(robust_iterations))):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_wrench = wrenches * np.sqrt(weights)[:, None]
        coefficients = np.linalg.solve(
            weighted_design.T @ weighted_design + regularizer,
            weighted_design.T @ weighted_wrench,
        )
        residual_norm = np.linalg.norm(wrenches - design @ coefficients, axis=1)
        median = float(np.median(residual_norm))
        scale = 1.4826 * float(np.median(np.abs(residual_norm - median))) + 1e-12
        huber_limit = median + 1.5 * scale
        weights = np.minimum(1.0, huber_limit / np.maximum(residual_norm, 1e-12))

    residual = wrenches - design @ coefficients
    rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    metrics = {
        "pose_count": int(len(tcp_poses)),
        "design_rank": rank,
        "rmse": rmse.tolist(),
        "force_rmse_norm": float(np.linalg.norm(rmse[:3])),
        "torque_rmse_norm": float(np.linalg.norm(rmse[3:])),
    }
    return StaticGravityCompensator(coefficients, metrics), metrics
