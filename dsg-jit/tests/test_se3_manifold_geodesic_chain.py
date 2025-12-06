from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_geodesic_residual
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig


def test_se3_manifold_geodesic_three_pose_chain():
    """
    Manifold GN + true SE(3) geodesic residual on a 3-pose chain.

    Ground truth (in se(3)-param form):
      pose0: [0, 0, 0, 0, 0, 0]
      pose1: [1, 0, 0, 0, 0, theta]
      pose2: [2, 0, 0, 0, 0, 2*theta]

    Factors:
      - prior on pose0: identity
      - odom_geodesic between pose0 and pose1: [1, 0, 0, 0, 0, theta]
      - odom_geodesic between pose1 and pose2: [1, 0, 0, 0, 0, theta]

    We check that manifold GN recovers this configuration from a perturbed
    initial guess.
    """

    wm = WorldModel()

    theta = 0.1  # small yaw

    # Initial guesses (intentionally a bit off)
    pose0_val = jnp.array([0.10, -0.05, 0.02, 0.02, -0.01, -0.01])
    pose1_val = jnp.array([0.9, 0.05, -0.02, 0.01, 0.01, theta + 0.03])
    pose2_val = jnp.array([2.1, -0.02, 0.01, -0.02, 0.02, 2 * theta - 0.04])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    pose1_id = wm.add_variable(var_type="pose_se3", value=pose1_val)
    pose2_id = wm.add_variable(var_type="pose_se3", value=pose2_val)

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Geodesic SE(3) odom: 1m along x, yaw = theta
    meas01 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas12 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])

    wm.add_factor(
        f_type="odom_se3_geodesic",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": meas01},
    )
    wm.add_factor(
        f_type="odom_se3_geodesic",
        var_ids=(pose1_id, pose2_id),
        params={"measurement": meas12},
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    # Build residual fn and manifold metadata using WorldModel
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()
    packed_state = (x_init, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )

    # Manifold-aware Gauss-Newton (slightly conservative damping/step)
    cfg = GNConfig(max_iters=20, damping=5e-3, max_step_norm=0.5)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = wm.unpack_state(x_opt, index)
    p0_opt = values[pose0_id]
    p1_opt = values[pose1_id]
    p2_opt = values[pose2_id]

    # pose0 ~ identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=5e-3)

    # pose1 ~ [1, 0, 0, 0, 0, theta]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[5]) == pytest.approx(theta, abs=5e-2)

    # pose2 ~ [2, 0, 0, 0, 0, 2*theta]
    """
    Ground truth (world frame):
    pose0: t=[0, 0, 0], yaw=0
    pose1: t=[1, 0, 0], yaw=theta
    pose2: t=[1+cos(theta), sin(theta), 0], yaw=2*theta
    """
    expected_t2x = 1.0 + jnp.cos(theta)
    expected_t2y = jnp.sin(theta)

    assert float(p2_opt[0]) == pytest.approx(float(expected_t2x), abs=5e-2)
    assert float(p2_opt[1]) == pytest.approx(float(expected_t2y), abs=5e-2)
    assert float(p2_opt[2]) == pytest.approx(0.0, abs=5e-2)

    assert float(p2_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p2_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p2_opt[5]) == pytest.approx(2 * theta, abs=5e-2)