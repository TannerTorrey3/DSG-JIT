import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, pose_landmark_relative_residual
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_pose_landmark_relative_single_pose():
    """
    Simple SE(3) + 3D landmark test.

    Ground truth:
      - pose: identity (0 translation, 0 rotation)
      - landmark_world: [1, 2, 3]

    Measurement:
      - landmark in pose frame: [1, 2, 3] (since pose is identity)

    We start from a slightly perturbed pose + landmark and check that
    Gauss–Newton recovers the correct configuration.
    """

    wm = WorldModel()

    # Slightly perturbed initial guesses
    pose0_val = jnp.array([0.1, -0.1, 0.05, 0.01, -0.02, 0.005])
    landmark0_val = jnp.array([1.1, 2.2, 2.9])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    landmark0_id = wm.add_variable(var_type="landmark3d", value=landmark0_val)

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Relative measurement: landmark in pose frame should be [1, 2, 3]
    meas = jnp.array([1.0, 2.0, 3.0])
    wm.add_factor(
        f_type="pose_landmark_relative",
        var_ids=(pose0_id, landmark0_id),
        params={"measurement": meas},
    )
    wm.add_factor(
        f_type="prior",
        var_ids=(landmark0_id,),
        params={"target": jnp.array([1.0, 2.0, 3.0])},
    )

    wm.register_residual("prior", prior_residual)
    wm.register_residual("pose_landmark_relative", pose_landmark_relative_residual)

    # Build residual fn and run Gauss–Newton using WorldModel
    x_init, index = wm.pack_state()
    x0 = x_init

    # Quick sanity check: residual at initial guess
    residual_fn_debug = wm.build_residual()
    r0 = residual_fn_debug(x0)
    print("initial residual:", r0)
    assert jnp.all(jnp.isfinite(r0))
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    p_opt = values[pose0_id]
    l_opt = values[landmark0_id]

    # pose ~ identity
    for i in range(6):
        assert float(p_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # landmark ~ [1, 2, 3]
    assert float(l_opt[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(l_opt[1]) == pytest.approx(2.0, abs=1e-2)
    assert float(l_opt[2]) == pytest.approx(3.0, abs=1e-2)