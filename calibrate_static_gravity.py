"""Collect static RTDE wrench samples and fit a compensation model."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e2.gravity_compensation import fit_static_gravity_model
else:
    from .gravity_compensation import fit_static_gravity_model


POSE_COLUMNS = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
WRENCH_COLUMNS = [f"tcp_force_raw{i}" for i in range(6)]


def save_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["pose_id", "sample_id", *POSE_COLUMNS, *WRENCH_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(rows)


def load_pose_averages(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row_number, row in enumerate(csv.DictReader(stream)):
                try:
                    pose = np.array([float(row[name]) for name in POSE_COLUMNS])
                    wrench = np.array([float(row[name]) for name in WRENCH_COLUMNS])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid calibration row {row_number + 2} in {path}") from exc
                pose_id = row.get("pose_id", str(row_number))
                groups.setdefault((str(path.resolve()), pose_id), []).append((pose, wrench))

    poses, wrenches = [], []
    for samples in groups.values():
        poses.append(np.median(np.vstack([sample[0] for sample in samples]), axis=0))
        wrenches.append(np.median(np.vstack([sample[1] for sample in samples]), axis=0))
    return np.asarray(poses), np.asarray(wrenches)


def fit_files(inputs: list[Path], model_path: Path) -> None:
    poses, wrenches = load_pose_averages(inputs)
    model, metrics = fit_static_gravity_model(poses, wrenches)
    model.metadata.update({"source_files": [str(path) for path in inputs]})
    model.save(model_path)
    rmse = np.asarray(metrics["rmse"])
    print(f"Model saved: {model_path}")
    print(f"Static poses: {metrics['pose_count']}, design rank: {metrics['design_rank']}/10")
    print("RMSE [Fx Fy Fz Tx Ty Tz]: " + np.array2string(rmse, precision=4))


def collect(args) -> None:
    try:
        from rtde_receive import RTDEReceiveInterface as RTDEReceive
    except ImportError as exc:
        raise RuntimeError("ur_rtde is required for live calibration collection") from exc

    receiver = RTDEReceive(
        args.robot_ip,
        args.frequency,
        ["actual_TCP_pose", "actual_TCP_force"],
        True,
        False,
        int(args.frequency),
    )
    rows: list[dict] = []
    pose_id = 0
    print("Use pendant freedrive to choose diverse, collision-free attitudes.")
    print("Release the robot and let it settle before each capture.")
    try:
        while True:
            command = input("Enter=capture, f=fit and finish, q=save and quit: ").strip().lower()
            if command == "f":
                break
            if command == "q":
                save_samples(args.samples, rows)
                print(f"Samples saved without fitting: {args.samples}")
                return
            if command:
                continue

            time.sleep(args.settle_time)
            pose_samples, wrench_samples = [], []
            sample_count = max(1, int(round(args.duration * args.frequency)))
            for sample_id in range(sample_count):
                pose = np.asarray(receiver.getActualTCPPose(), dtype=float).reshape(6)
                wrench = np.asarray(receiver.getActualTCPForce(), dtype=float).reshape(6)
                pose_samples.append(pose)
                wrench_samples.append(wrench)
                row = {"pose_id": pose_id, "sample_id": sample_id}
                row.update(dict(zip(POSE_COLUMNS, pose)))
                row.update(dict(zip(WRENCH_COLUMNS, wrench)))
                rows.append(row)
                time.sleep(1.0 / args.frequency)

            pose_samples = np.vstack(pose_samples)
            position_span = np.ptp(pose_samples[:, :3], axis=0)
            rotation_span = np.ptp(pose_samples[:, 3:], axis=0)
            if np.max(position_span) > args.max_position_span or np.max(rotation_span) > args.max_rotation_span:
                del rows[-sample_count:]
                print("Rejected: the TCP moved during capture; release it and try again.")
                continue
            median_wrench = np.median(np.vstack(wrench_samples), axis=0)
            print(f"Captured pose {pose_id}: wrench {np.array2string(median_wrench, precision=3)}")
            pose_id += 1
            save_samples(args.samples, rows)
    finally:
        if hasattr(receiver, "disconnect"):
            receiver.disconnect()

    save_samples(args.samples, rows)
    fit_files([args.samples], args.model)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Offline static gravity calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="collect poses from the robot and fit")
    collect_parser.add_argument("--robot-ip", default="10.31.17.47")
    collect_parser.add_argument("--frequency", type=float, default=100.0)
    collect_parser.add_argument("--duration", type=float, default=1.0)
    collect_parser.add_argument("--settle-time", type=float, default=0.5)
    collect_parser.add_argument("--max-position-span", type=float, default=0.001)
    collect_parser.add_argument("--max-rotation-span", type=float, default=0.005)
    collect_parser.add_argument(
        "--samples", type=Path, default=project_dir / "data" / "static_gravity_samples.csv"
    )
    collect_parser.add_argument(
        "--model", type=Path, default=project_dir / "data" / "static_gravity_model.json"
    )

    fit_parser = subparsers.add_parser("fit", help="fit one or more existing calibration CSVs")
    fit_parser.add_argument("inputs", nargs="+", type=Path)
    fit_parser.add_argument(
        "--model", type=Path, default=project_dir / "data" / "static_gravity_model.json"
    )
    args = parser.parse_args()

    if args.command == "collect":
        collect(args)
    else:
        fit_files(args.inputs, args.model)


if __name__ == "__main__":
    main()
