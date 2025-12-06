# tests/test_pose_place_attachment.py
from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual
from dsg_jit.scene_graph.relations import pose_place_attachment_residual
from dsg_jit.optimization.solvers import gradient_descent, GDConfig, GNConfig, gauss_newton


def test_pose_place_attachment_1d_along_x():
    """
    Test that a place node is pulled towards the pose's x-translation.

    Setup:
      - pose0, pose1 in R^6 (SE(3)-like, but additive)
      - place0 scalar

    Factors:
      - prior on pose0: identity (0,...)
      - odom_se3 from pose0 -> pose1: +1m in x
      - attachment between pose1 and place0: place0 ~ pose1.tx

    Expect:
      - pose0.tx ~ 0
      - pose1.tx ~ 1
      - place0   ~ 1
    """

    wm = WorldModel()

    # Variables
    pose0_val = jnp.array([0.2, -0.1, 0.0, 0.0, 0.0, 0.0])
    pose1_val = jnp.array([0.7, 0.1, 0.0, 0.0, 0.0, 0.0])
    place0_val = jnp.array([3.0])  # far away

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    pose1_id = wm.add_variable(var_type="pose_se3", value=pose1_val)
    place0_id = wm.add_variable(var_type="place1d", value=place0_val)

    # Prior on pose0
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Odom: +1m along x
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": meas},
    )

    # Attachment: place0 ~ pose1.tx
    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose1_id, place0_id),  # pose1, place0
        params={
            "pose_dim": jnp.array(6),
            "place_dim": jnp.array(1),
            "pose_coord_index": jnp.array(0),  # tx
        },
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # Optimize using Gauss-Newton on residuals via WorldModel
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    p0 = values[pose0_id]
    p1 = values[pose1_id]
    pl = values[place0_id]

    # pose0 ~ identity
    assert float(p0[0]) == pytest.approx(0.0, abs=1e-2)
    # pose1.x ~ 1.0
    assert float(p1[0]) == pytest.approx(1.0, abs=5e-2)
    # place0 ~ pose1.x
    assert float(pl[0]) == pytest.approx(1.0, abs=5e-2)