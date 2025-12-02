# tests/test_pose_place_attachment.py
from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
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

    fg = FactorGraph()

    # Variables
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.2, -0.1, 0.0, 0.0, 0.0, 0.0]),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([0.7, 0.1, 0.0, 0.0, 0.0, 0.0]),
    )
    place0 = Variable(
        id=NodeId(2),
        type="place1d",
        value=jnp.array([3.0]),  # far away
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(place0)

    # Prior on pose0
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Odom: +1m along x
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    f_odom = Factor(
        id=FactorId(1),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": meas},
    )

    # Attachment: place0 ~ pose1.tx
    f_attach = Factor(
        id=FactorId(2),
        type="pose_place_attachment",
        var_ids=(NodeId(1), NodeId(2)),  # pose1, place0
        params={
            "pose_dim": jnp.array(6),
            "place_dim": jnp.array(1),
            "pose_coord_index": jnp.array(0),  # tx
        },
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom)
    fg.add_factor(f_attach)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # Optimize using Gauss-Newton on residuals
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = fg.unpack_state(x_opt, index)
    p0 = values[NodeId(0)]
    p1 = values[NodeId(1)]
    pl = values[NodeId(2)]

    # pose0 ~ identity
    assert float(p0[0]) == pytest.approx(0.0, abs=1e-2)
    # pose1.x ~ 1.0
    assert float(p1[0]) == pytest.approx(1.0, abs=5e-2)
    # place0 ~ pose1.x
    assert float(pl[0]) == pytest.approx(1.0, abs=5e-2)