from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual


def test_world_model_se3_two_pose_chain():
    """
    WorldModel + SE(3) check.

    Variables:
      - pose0, pose1 (R^6: [tx, ty, tz, wx, wy, wz])

    Factors:
      - prior on pose0: identity (0 translation, 0 rotation)
      - odom_se3 between pose0 and pose1: 1m along x, 90 deg about z

    Expected:
      - pose0 ~ [0, 0, 0, 0, 0, 0]
      - pose1 ~ [1, 0, 0, 0, 0, pi/2]
    """

    wm = WorldModel()

    # Register residuals
    wm.fg.register_residual("prior", prior_residual)
    wm.fg.register_residual("odom_se3", odom_se3_residual)

    # Initial guesses (intentionally off)
    pose0 = wm.add_variable(
        "pose_se3",
        jnp.array([0.2, -0.1, 0.05, 0.02, -0.01, 0.01]),
    )
    pose1 = wm.add_variable(
        "pose_se3",
        jnp.array([0.8, 0.2, -0.1, 0.2, 0.1, -0.05]),
    )

    # Prior on pose0: identity
    wm.add_factor(
        "prior",
        (pose0,),
        {"target": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
    )

    # SE(3) odom: 1m along x, 90 deg around z
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    wm.add_factor(
        "odom_se3",
        (pose0, pose1),
        {"measurement": meas},
    )

    # Optimize
    wm.optimize(iters=20, method="gn")

    p0v = wm.fg.variables[pose0].value
    p1v = wm.fg.variables[pose1].value

    # pose0 ~ identity
    for i in range(6):
        assert float(p0v[i]) == pytest.approx(0.0, abs=1e-2)

    # pose1 ~ [1, 0, 0, 0, 0, pi/2]
    assert float(p1v[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1v[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1v[2]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1v[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1v[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1v[5]) == pytest.approx(0.0, abs=5e-2)