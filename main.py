#visual-admittance controller, notice that here we use position-based visual servoing (PBVS) but not image-based visual servoing(IBVS).
import numpy as np
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e2.config import PBVSConfig, TargetPose
    from e2.Mathematic import matrix_from_quat
else:
    from .config import PBVSConfig, TargetPose
    from .Mathematic import matrix_from_quat


ENABLE_VISUALIZATION = True
ENABLE_MEMORY_LOG = True
STATUS_PRINT_INTERVAL = 30
APRILTAG_NTHREADS = 2
APRILTAG_QUAD_DECIMATE = 2.0

ROBOT_IP = "10.31.17.47"
  

def hand_eye_matrix() -> np.ndarray:
    return np.array([
        [0.0074, -0.9994, -0.0329, -0.0715],
        [1.0000,  0.0074, -0.0010, -0.0328],
        [0.0012, -0.0329,  0.9995,  0.0499],
        [0.0000,  0.0000,  0.0000,  1.0000],
    ])


def controller_params() -> dict:
    return dict(
        controller_mode="SOPD",
        kp=500 * np.ones(6),
        kd=150 * np.ones(6),
        max_linear_vel=float("inf"),
        max_angular_vel=float("inf"),
        feature_admittance_mass= 1.0,
        feature_admittance_damping= 20.0,
        feature_admittance_stiffness= 500.0,
        feature_admittance_max_offset=(float("inf"),) * 6,
        cartesian_admittance_mass=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        cartesian_admittance_damping=(20.0, 20.0, 20.0, 20.0, 20.0, 20.0),
        cartesian_admittance_stiffness=(500.0, 500.0, 500.0, 500.0, 500.0, 500.0),
        cartesian_admittance_max_offset=(float("inf"),) * 6,
    )


def build_config(args) -> PBVSConfig:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    gravity_model_path = Path(args.gravity_model)
    gravity_compensation_enabled = (
        not args.no_gravity_compensation and gravity_model_path.is_file()
    )
    if not args.no_gravity_compensation and not gravity_compensation_enabled:
        print(
            f"Static gravity model not found: {gravity_model_path}. "
            "Running with the legacy single-pose force zero."
        )
    controller_name = args.controller
    enable_feature_admittance = controller_name == "cA"
    enable_cartesian_admittance = controller_name == "cB"
    params = controller_params()
    if enable_feature_admittance:
        # Conservative hardware envelope for force-driven motion.
        params.update(
            accel_limit_pos=float("inf"),
            accel_limit_rot=float("inf"),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        )
    elif enable_cartesian_admittance:
        # cB directly converts physical wrench into a Cartesian pose offset.
        # Keep finite command limits while validating it on hardware.
        params.update(
            accel_limit_pos=float("inf"),
            accel_limit_rot=float("inf"),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        )

    return PBVSConfig(
        tag_size=0.08,
        detect_stride=1,
        apriltag_nthreads=APRILTAG_NTHREADS,
        apriltag_quad_decimate=APRILTAG_QUAD_DECIMATE,
        enable_visualization=ENABLE_VISUALIZATION,
        visualization_stride=1,
        pos_threshold=0.002,
        rot_threshold=0.01,
        slow_after_convergence=False,
        max_runtime=args.runtime,
        log_save_path=str(data_dir / f"log_{controller_name}.csv"),
        enable_memory_log=ENABLE_MEMORY_LOG,
        status_print_interval=STATUS_PRINT_INTERVAL,
        enable_static_gravity_compensation=gravity_compensation_enabled,
        static_gravity_model_path=str(gravity_model_path),
        controller_name=controller_name,
        enable_feature_admittance=enable_feature_admittance,
        enable_cartesian_admittance=enable_cartesian_admittance,
        **params,
    )


def build_targets():
    desired_quaternion = np.array([1, 0, 0, 0])
    base_rotation = matrix_from_quat(desired_quaternion)
    base_translation = np.array([0.00, 0.00, 0.25])
    return [
        TargetPose(
            name="Large_error_start",
            desired_rotation=base_rotation,
            desired_translation=base_translation,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run cO visual servoing, cA feature-space admittance, "
            "or cB Cartesian-space admittance."
        )
    )
    parser.add_argument(
        "--controller",
        choices=["cO", "cA", "cB"],
        default="cA",
        help=(
            "cO: visual servoing only; cA: feature-space admittance; "
            "cB: Cartesian-space admittance."
        ),
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=8.0,
        help="Recording duration in seconds. Use 0 for manual stop.",
    )
    parser.add_argument(
        "--gravity-model",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "static_gravity_model.json",
        help="Offline-identified static wrench compensation model.",
    )
    parser.add_argument(
        "--no-gravity-compensation",
        action="store_true",
        help="Disable the static gravity model even if the model file exists.",
    )
    args = parser.parse_args()

    if __package__ in (None, ""):
        from e2.vision import init_realsense
        from e2.controller import PBVSController
    else:
        from .vision import init_realsense
        from .controller import PBVSController

    pipeline, intr = init_realsense()
    intrinsics_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)

    controller = PBVSController(
        robot_ip=ROBOT_IP,
        intrinsics=intrinsics_params,
        hand_eye_calib=hand_eye_matrix(),
        config=build_config(args),
    )
    controller.set_targets(build_targets())

    init_pose = np.array([
        -0.25588217, -0.15717437,  0.250001008,
        3.10924706, -0.43501290,  0.01221673,
    ])
    # init_pose = np.array([
    #         -0.20588217, -0.05717437,  0.420001008,
    #         -2.50455863, -1.8897531,  -0.01089382,
    #     ])
    controller.run(pipeline, init_pose)


if __name__ == "__main__":
    main()
