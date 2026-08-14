import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


# controller.py imports the RTDE modules at module load time.  The tests exercise
# only the numerical controller and must never attempt a robot connection.
rtde_control = types.ModuleType("rtde_control")
rtde_receive = types.ModuleType("rtde_receive")
rtde_control.RTDEControlInterface = object
rtde_receive.RTDEReceiveInterface = object
sys.modules.setdefault("rtde_control", rtde_control)
sys.modules.setdefault("rtde_receive", rtde_receive)

from e2.config import PBVSConfig
from e2.controller import FeatureSpaceAdmittance, PBVSController
from e2.geometry import compute_L
from e2.main import build_config


class FeatureSpaceAdmittanceTests(unittest.TestCase):
    def test_controller_name_selects_admittance_and_log(self):
        visual = build_config(Namespace(runtime=8.0, controller="cO"))
        admittance = build_config(Namespace(runtime=8.0, controller="cA"))

        self.assertEqual(visual.controller_name, "cO")
        self.assertFalse(visual.enable_feature_admittance)
        self.assertEqual(Path(visual.log_save_path).name, "log_cO.csv")
        self.assertEqual(admittance.controller_name, "cA")
        self.assertTrue(admittance.enable_feature_admittance)
        self.assertEqual(Path(admittance.log_save_path).name, "log_cA.csv")

    def test_default_parameters_remain_bounded_and_converge(self):
        cfg = PBVSConfig(enable_feature_admittance=True)
        model = FeatureSpaceAdmittance(cfg, 1.0 / 60.0)
        desired = np.zeros(6)
        interaction = np.eye(6)
        wrench = np.array([1.6, -1.2, 0.8, 0.2, -0.1, 0.05])

        history = []
        for _ in range(6000):
            history.append(
                model.compute(
                    desired,
                    interaction,
                    wrench,
                    R_base_cam=np.eye(3),
                ).copy()
            )

        history = np.asarray(history)
        self.assertTrue(np.all(np.isfinite(history)))
        self.assertLessEqual(
            np.max(np.abs(history)),
            np.max(cfg.feature_admittance_max_offset) + 1e-12,
        )
        np.testing.assert_allclose(
            model.s_p[:3], wrench[:3] / cfg.feature_admittance_stiffness[:3],
            atol=1e-8,
        )
        self.assertLess(np.max(np.abs(model.s_p_dot)), 1e-8)

    def test_feature_state_limits_are_enforced(self):
        cfg = PBVSConfig(enable_feature_admittance=True)
        model = FeatureSpaceAdmittance(cfg, 1.0 / 60.0)

        for _ in range(1000):
            model.compute(
                np.zeros(6),
                np.eye(6),
                np.full(6, 1e9),
                R_base_cam=np.eye(3),
            )

        self.assertTrue(np.all(np.isfinite(model.s_p)))
        self.assertTrue(
            np.all(np.abs(model.s_p) <= cfg.feature_admittance_max_offset + 1e-12)
        )
        self.assertTrue(
            np.all(np.abs(model.s_p_dot) <= cfg.feature_admittance_max_velocity + 1e-12)
        )
        self.assertTrue(
            np.all(
                np.abs(model.s_p_ddot)
                <= cfg.feature_admittance_max_acceleration + 1e-12
            )
        )

    def test_tcp_twist_norm_limits_are_enforced(self):
        controller = object.__new__(PBVSController)
        controller.cfg = PBVSConfig(max_linear_vel=0.1, max_angular_vel=0.3)
        clipped, saturated = controller._clip_tcp_twist(
            np.array([3.0, 4.0, 0.0, 0.0, 0.0, -2.0])
        )

        self.assertTrue(saturated)
        self.assertAlmostEqual(np.linalg.norm(clipped[:3]), 0.1)
        self.assertAlmostEqual(np.linalg.norm(clipped[3:]), 0.3)

    def test_manual_force_zeroing_recomputes_bias(self):
        controller = object.__new__(PBVSController)
        controller.cfg = PBVSConfig(
            rtde_force_bias_samples=3,
            rtde_force_lowpass_tau=0.0,
            rtde_force_deadband=np.zeros(6),
        )
        controller.dt = 1.0 / 60.0
        controller._force_bias = np.ones(6)
        controller._force_lowpass = np.ones(6)
        controller._last_wrench_tcp = np.ones(6)
        controller._last_wrench_cam = np.ones(6)

        controller._reset_force_bias(manual=True)
        sample = np.arange(1.0, 7.0)
        for _ in range(3):
            self.assertIsNone(controller._update_force_bias_and_filter(sample))

        self.assertTrue(controller._force_bias_ready)
        self.assertFalse(controller._force_zero_pending)
        np.testing.assert_allclose(controller._force_bias, sample)
        np.testing.assert_allclose(
            controller._update_force_bias_and_filter(sample), np.zeros(6)
        )

    def test_camera_wrench_rotation_is_not_applied_twice(self):
        cfg = PBVSConfig(enable_feature_admittance=True)
        model = FeatureSpaceAdmittance(cfg, 1.0 / 60.0)
        R_base_cam = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        L_base = compute_L(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            R_base_cam,
        )

        model.compute(
            np.zeros(6),
            L_base,
            external_wrench_cam=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            R_base_cam=R_base_cam,
        )

        # Camera motion changes the observed feature in the opposite direction,
        # so test the resulting base-frame camera command rather than the sign
        # of the feature displacement itself.
        self.assertLess(model.s_p_dot[0], 0.0)
        self.assertAlmostEqual(model.s_p_dot[1], 0.0, places=12)
        self.assertAlmostEqual(model.s_p_dot[2], 0.0, places=12)
        command_base = np.linalg.solve(L_base, model.s_p)
        applied_force_base = R_base_cam @ np.array([2.0, 0.0, 0.0])
        self.assertGreater(command_base[:3] @ applied_force_base, 0.0)
        self.assertAlmostEqual(
            np.linalg.norm(np.cross(command_base[:3], applied_force_base)),
            0.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
