# tests/test_scene_graph.py
from __future__ import annotations

import jax.numpy as jnp
import pytest

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import prior_residual
from optimization.solvers import gradient_descent, GDConfig
from scene_graph.relations import room_centroid_residual


def test_room_centroid_with_fixed_places():
    """
    Scene-graph-style test:

      - Three place nodes at positions: 0, 2, 4  (1D for simplicity)
      - One room node with unknown position.

    Factors:
      - Priors on each place to keep them near [0, 2, 4].
      - A room_centroid factor tying the room to the mean of the places.

    Optimum:
      - place0 ≈ 0
      - place1 ≈ 2
      - place2 ≈ 4
      - room ≈ (0 + 2 + 4) / 3 = 2
    """

    fg = FactorGraph()

    # --- Variables: 1D positions for simplicity ---
    place0 = Variable(id=NodeId(0), type="place1d", value=jnp.array([0.5]))
    place1 = Variable(id=NodeId(1), type="place1d", value=jnp.array([2.5]))
    place2 = Variable(id=NodeId(2), type="place1d", value=jnp.array([3.5]))
    room   = Variable(id=NodeId(3), type="room1d",  value=jnp.array([5.0]))

    fg.add_variable(place0)
    fg.add_variable(place1)
    fg.add_variable(place2)
    fg.add_variable(room)

    # --- Priors on places to fix them near 0, 2, 4 ---
    f_prior_p0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([0.0])},
    )
    f_prior_p1 = Factor(
        id=FactorId(1),
        type="prior",
        var_ids=(NodeId(1),),
        params={"target": jnp.array([2.0])},
    )
    f_prior_p2 = Factor(
        id=FactorId(2),
        type="prior",
        var_ids=(NodeId(2),),
        params={"target": jnp.array([4.0])},
    )

    # --- Room centroid factor: room tied to mean of places ---
    f_room = Factor(
        id=FactorId(3),
        type="room_centroid",
        var_ids=(NodeId(3), NodeId(0), NodeId(1), NodeId(2)),
        params={"dim": jnp.array(1)},  # 1D positions
    )

    fg.add_factor(f_prior_p0)
    fg.add_factor(f_prior_p1)
    fg.add_factor(f_prior_p2)
    fg.add_factor(f_room)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("room_centroid", room_centroid_residual)

    # Optimize
    x_init, index = fg.pack_state()
    objective = fg.build_objective()

    cfg = GDConfig(learning_rate=0.1, max_iters=400)
    x_opt = gradient_descent(objective, x_init, cfg)

    values = fg.unpack_state(x_opt, index)

    p0_opt = float(values[NodeId(0)][0])
    p1_opt = float(values[NodeId(1)][0])
    p2_opt = float(values[NodeId(2)][0])
    r_opt  = float(values[NodeId(3)][0])

    # Places close to desired priors
    assert p0_opt == pytest.approx(0.0, rel=1e-3, abs=1e-3)
    assert p1_opt == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert p2_opt == pytest.approx(4.0, rel=1e-3, abs=1e-3)

    # Room at centroid (2.0)
    assert r_opt == pytest.approx(2.0, rel=1e-3, abs=1e-3)