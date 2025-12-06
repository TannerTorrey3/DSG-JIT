from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual
from dsg_jit.optimization.solvers import gradient_descent, GDConfig
from dsg_jit.scene_graph.relations import room_centroid_residual


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

    wm = WorldModel()

    # --- Variables: 1D positions for simplicity ---
    place0_id = wm.add_variable(var_type="place1d", value=jnp.array([0.5]))
    place1_id = wm.add_variable(var_type="place1d", value=jnp.array([2.5]))
    place2_id = wm.add_variable(var_type="place1d", value=jnp.array([3.5]))
    room_id   = wm.add_variable(var_type="room1d",  value=jnp.array([5.0]))

    # --- Priors on places to fix them near 0, 2, 4 ---
    wm.add_factor(
        f_type="prior",
        var_ids=(place0_id,),
        params={"target": jnp.array([0.0])},
    )
    wm.add_factor(
        f_type="prior",
        var_ids=(place1_id,),
        params={"target": jnp.array([2.0])},
    )
    wm.add_factor(
        f_type="prior",
        var_ids=(place2_id,),
        params={"target": jnp.array([4.0])},
    )

    # --- Room centroid factor: room tied to mean of places ---
    wm.add_factor(
        f_type="room_centroid",
        var_ids=(room_id, place0_id, place1_id, place2_id),
        params={"dim": jnp.array(1)},  # 1D positions
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("room_centroid", room_centroid_residual)

    # Optimize using WorldModel
    x_init, index = wm.pack_state()
    objective = wm.build_objective()

    cfg = GDConfig(learning_rate=0.1, max_iters=400)
    x_opt = gradient_descent(objective, x_init, cfg)

    values = wm.unpack_state(x_opt, index)

    p0_opt = float(values[place0_id][0])
    p1_opt = float(values[place1_id][0])
    p2_opt = float(values[place2_id][0])
    r_opt  = float(values[room_id][0])

    # Places close to desired priors
    assert p0_opt == pytest.approx(0.0, rel=1e-3, abs=1e-3)
    assert p1_opt == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert p2_opt == pytest.approx(4.0, rel=1e-3, abs=1e-3)

    # Room at centroid (2.0)
    assert r_opt == pytest.approx(2.0, rel=1e-3, abs=1e-3)