import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
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

    fg = FactorGraph()

    # Slightly perturbed initial guesses
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.1, -0.1, 0.05, 0.01, -0.02, 0.005]),
    )
    landmark0 = Variable(
        id=NodeId(1),
        type="landmark3d",
        value=jnp.array([1.1, 2.2, 2.9]),
    )

    fg.add_variable(pose0)
    fg.add_variable(landmark0)

    # Prior on pose0: identity
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Relative measurement: landmark in pose frame should be [1, 2, 3]
    meas = jnp.array([1.0, 2.0, 3.0])
    f_rel = Factor(
        id=FactorId(1),
        type="pose_landmark_relative",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": meas},
    )
    f_prior_lmk = Factor(
    id=FactorId(2),
    type="prior",
    var_ids=(NodeId(1),),
    params={"target": jnp.array([1.0, 2.0, 3.0])},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_prior_lmk)
    fg.add_factor(f_rel)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("pose_landmark_relative", pose_landmark_relative_residual)

    # Build residual fn and run Gauss–Newton
    x_init, index = fg.pack_state()

    x0, index = fg.pack_state()

# Quick sanity check: residual at initial guess
    residual_fn_debug = fg.build_residual_function()
    r0 = residual_fn_debug(x0)
    print("initial residual:", r0)
    assert jnp.all(jnp.isfinite(r0))
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = fg.unpack_state(x_opt, index)
    p_opt = values[NodeId(0)]
    l_opt = values[NodeId(1)]

    # pose ~ identity
    for i in range(6):
        assert float(p_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # landmark ~ [1, 2, 3]
    assert float(l_opt[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(l_opt[1]) == pytest.approx(2.0, abs=1e-2)
    assert float(l_opt[2]) == pytest.approx(3.0, abs=1e-2)