import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    pose_voxel_point_residual,
)
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_pose_voxel_point_alignment():
    """
    One pose, one voxel, one 3D point measurement.

    Ground truth:
      - pose0: identity (world = pose frame)
      - point_meas (pose frame): [1, 0.5, 0.0]
      - voxel_world should converge to [1, 0.5, 0.0]

    Setup:
      Variables:
        - pose0: pose_se3 (R^6)
        - voxel0: voxel_cell (R^3)

      Factors:
        - prior on pose0 -> identity
        - weak prior on voxel0 around [1, 0.5, 0]
        - pose_voxel_point: enforces voxel_center ≈ T(pose0) * point_meas
    """
    wm = WorldModel()

    # Initial guesses (intentionally a bit off)
    pose0_val = jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.0])
    voxel0_val = jnp.array([0.5, 0.0, 0.1])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    voxel0_id = wm.add_variable(var_type="voxel_cell", value=voxel0_val)

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Weak prior on voxel0 near the expected solution
    wm.add_factor(
        f_type="prior",
        var_ids=(voxel0_id,),
        params={"target": jnp.array([1.0, 0.5, 0.0])},
    )

    # 3D point measurement in the pose frame
    point_meas = jnp.array([1.0, 0.5, 0.0])
    wm.add_factor(
        f_type="pose_voxel_point",
        var_ids=(pose0_id, voxel0_id),
        params={"point_meas": point_meas},
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("pose_voxel_point", pose_voxel_point_residual)

    # Optimize with conservative Gauss-Newton (to avoid SE(3) blowups)
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(
        max_iters=40,
        damping=1e-2,       # stronger damping for stability
        max_step_norm=0.05  # smaller steps in SE(3)
    )
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    p_opt = values[pose0_id]
    v_opt = values[voxel0_id]

    # pose0 should be near identity
    for i in range(6):
        assert float(p_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # voxel0 should be near the world point [1, 0.5, 0.0]
    assert float(v_opt[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(v_opt[1]) == pytest.approx(0.5, abs=1e-2)
    assert float(v_opt[2]) == pytest.approx(0.0, abs=1e-2)