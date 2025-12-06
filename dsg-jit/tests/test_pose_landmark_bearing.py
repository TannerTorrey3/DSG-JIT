import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
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
    wm = WorldModel()

    # Initial guesses
    pose0_val = jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.005])
    pose1_val = jnp.array([1.2, 0.2, 0.0, 0.02, 0.01, -0.01])
    landmark_val = jnp.array([0.8, 1.3, -0.1])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    pose1_id = wm.add_variable(var_type="pose_se3", value=pose1_val)
    landmark_id = wm.add_variable(var_type="landmark3d", value=landmark_val)

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Odom: pose0 -> pose1 is +1m in x, no rotation
    odom_meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": odom_meas},
    )

    # True landmark
    l_true = jnp.array([1.0, 1.0, 0.0])

    # Bearing measurements
    b0 = l_true - jnp.array([0.0, 0.0, 0.0])
    b1 = l_true - jnp.array([1.0, 0.0, 0.0])
    b0 = b0 / (jnp.linalg.norm(b0) + 1e-8)
    b1 = b1 / (jnp.linalg.norm(b1) + 1e-8)

    wm.add_factor(
        f_type="pose_landmark_bearing",
        var_ids=(pose0_id, landmark_id),
        params={"bearing_meas": b0},
    )
    wm.add_factor(
        f_type="pose_landmark_bearing",
        var_ids=(pose1_id, landmark_id),
        params={"bearing_meas": b1},
    )

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("pose_landmark_bearing", pose_landmark_bearing_residual)

    # Solve with conservative GN using WorldModel residuals/packing
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    p0_opt = values[pose0_id]
    p1_opt = values[pose1_id]
    l_opt = values[landmark_id]

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