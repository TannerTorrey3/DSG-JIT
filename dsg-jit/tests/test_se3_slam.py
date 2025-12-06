# tests/test_se3_slam.py
from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual
from dsg_jit.optimization.solvers import gauss_newton, GNConfig


def test_se3_odom_two_poses():
    """
    SE(3) SLAM-style test with true SE(3) residual.

    Variables:
      pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]

    Factors:
      - prior on pose0: wants pose0 == identity (0 translation, 0 rotation)
      - odom_se3 between pose0 and pose1: wants relative transform:
            translation: [1, 0, 0]
            rotation:    0 (for now, to keep things numerically mild)

    Optimum:
      pose0 ~ [0, 0, 0, 0, 0, 0]
      pose1 ~ [1, 0, 0, 0, 0, 0]
    """

    wm = WorldModel()

    # Initial guesses
    pose0_val = jnp.array([0.1, -0.2, 0.05, 0.01, -0.02, 0.005])
    pose1_val = jnp.array([0.2, 0.1, -0.1, 0.1, 0.05, -0.02])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    pose1_id = wm.add_variable(var_type="pose_se3", value=pose1_val)

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
    )

    # SE(3) odometry measurement: 1m along x, zero rotation
    measurement = jnp.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])

    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": measurement},
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)

    # Build residual function and run Gauss-Newton
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    p0_opt = values[pose0_id]
    p1_opt = values[pose1_id]

    # pose0 should be near identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # pose1 should be close to [1, 0, 0, 0, 0, 0]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[5]) == pytest.approx(0.0, abs=5e-2)