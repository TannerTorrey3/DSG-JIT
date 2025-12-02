import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,
    pose_landmark_bearing_residual,
)
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_pose_landmark_bearing_two_poses():
    """
    Two poses observing the same landmark with bearing-only measurements.

    Ground truth:
      pose0: identity
      pose1: +1m along x, no rotation
      landmark_world ~ [1.0, 1.0, 0.0]

    From pose0:
      bearing0 = normalize([1, 1, 0])
    From pose1:
      bearing1 = normalize([0, 1, 0])  # landmark at x=1, y=1

    Factors:
      - prior on pose0
      - odom_se3 between pose0 and pose1: [1, 0, 0, 0, 0, 0]
      - bearing from pose0 to landmark
      - bearing from pose1 to landmark

    We perturb initial guesses and check we recover near ground truth.
    """
    fg = FactorGraph()

    # Initial guesses
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.005]),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([1.2, 0.2, 0.0, 0.02, 0.01, -0.01]),
    )
    landmark = Variable(
        id=NodeId(2),
        type="landmark3d",
        value=jnp.array([0.8, 1.3, -0.1]),
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(landmark)

    # Prior on pose0: identity
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Odom: pose0 -> pose1 is +1m in x, no rotation
    odom_meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    f_odom = Factor(
        id=FactorId(1),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": odom_meas},
    )

    # True landmark
    l_true = jnp.array([1.0, 1.0, 0.0])

    # Bearing measurements
    b0 = l_true - jnp.array([0.0, 0.0, 0.0])
    b1 = l_true - jnp.array([1.0, 0.0, 0.0])
    b0 = b0 / (jnp.linalg.norm(b0) + 1e-8)
    b1 = b1 / (jnp.linalg.norm(b1) + 1e-8)

    f_b0 = Factor(
        id=FactorId(2),
        type="pose_landmark_bearing",
        var_ids=(NodeId(0), NodeId(2)),
        params={"bearing_meas": b0},
    )
    f_b1 = Factor(
        id=FactorId(3),
        type="pose_landmark_bearing",
        var_ids=(NodeId(1), NodeId(2)),
        params={"bearing_meas": b1},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom)
    fg.add_factor(f_b0)
    fg.add_factor(f_b1)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("pose_landmark_bearing", pose_landmark_bearing_residual)

    # Solve with conservative GN
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = fg.unpack_state(x_opt, index)
    p0_opt = values[NodeId(0)]
    p1_opt = values[NodeId(1)]
    l_opt = values[NodeId(2)]

    # pose0 ~ identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # pose1 ~ [1, 0, 0, 0, 0, 0]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=1e-2)
    for i in range(3, 6):
        assert float(p1_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # landmark ~ [1, 1, 0]
    assert float(l_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(l_opt[1]) == pytest.approx(1.0, abs=5e-2)
    assert float(l_opt[2]) == pytest.approx(0.0, abs=5e-2)