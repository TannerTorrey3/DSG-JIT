
from __future__ import annotations

import pytest
import jax.numpy as jnp

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import prior_residual, odom_residual
from optimization.solvers import gradient_descent, GDConfig


def test_single_variable_prior():
    """
    One variable x, one prior factor:
        residual = x - target
    The optimum should be x ~= target.
    """
    fg = FactorGraph()

    # Variable: scalar x initialized at 0
    x0 = Variable(id=NodeId(0), type="scalar", value=jnp.array([0.0]))
    fg.add_variable(x0)

    # Factor: prior that wants x = 2.0
    f0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([2.0])},
    )
    fg.add_factor(f0)

    # Register residual
    fg.register_residual("prior", prior_residual)

    # Build objective and initial state
    x_init, index = fg.pack_state()
    objective = fg.build_objective()

    # Run simple GD
    cfg = GDConfig(learning_rate=0.2, max_iters=100)
    x_opt = gradient_descent(objective, x_init, cfg)

    # Expect x_opt ~ 2.0
    assert x_opt.shape == (1,)
    assert float(x_opt[0]) == pytest.approx(2.0, rel=1e-3, abs=1e-3)


def test_tiny_slam_prior_plus_odom():
    """
    Two variables: p0, p1 (1D each for simplicity).

    Factors:
      - prior on p0: wants p0 = 0
      - odom between p0 and p1: wants (p1 - p0) = 1

    Optimum: p0 = 0, p1 = 1
    """
    fg = FactorGraph()

    p0 = Variable(id=NodeId(0), type="pose1d", value=jnp.array([0.5]))
    p1 = Variable(id=NodeId(1), type="pose1d", value=jnp.array([0.5]))

    fg.add_variable(p0)
    fg.add_variable(p1)

    # Prior on p0
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([0.0])},
    )

    # Odometry factor between p0 and p1: expects delta = 1
    f_odom = Factor(
        id=FactorId(1),
        type="odom",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": jnp.array([1.0])},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom", odom_residual)

    x_init, index = fg.pack_state()
    objective = fg.build_objective()

    cfg = GDConfig(learning_rate=0.1, max_iters=300)
    x_opt = gradient_descent(objective, x_init, cfg)

    # Unpack to variables
    var_values = fg.unpack_state(x_opt, index)
    p0_opt = float(var_values[NodeId(0)][0])
    p1_opt = float(var_values[NodeId(1)][0])

    assert p0_opt == pytest.approx(0.0, rel=1e-3, abs=1e-3)
    assert p1_opt == pytest.approx(1.0, rel=1e-3, abs=1e-3)