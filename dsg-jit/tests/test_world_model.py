from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_residual
from dsg_jit.scene_graph.relations import room_centroid_residual


def test_world_model_joint_optimization():
    """
    Same scenario as prior tests, but using the WorldModel API.

    Variables:
      - place0 = ~0
      - place1 = ~2
      - place2 = ~4
      - room   = unknown

    Factors:
      - priors on each place
      - room_centroid factor tying room to mean(place)
    """

    wm = WorldModel()

    # Register residuals
    wm.fg.register_residual("prior", prior_residual)
    wm.fg.register_residual("odom", odom_residual)
    wm.fg.register_residual("room_centroid", room_centroid_residual)

    # Variables (1D)
    p0 = wm.add_variable("place1d", jnp.array([0.5]))
    p1 = wm.add_variable("place1d", jnp.array([2.5]))
    p2 = wm.add_variable("place1d", jnp.array([3.5]))
       # room starts far away
    room = wm.add_variable("room1d", jnp.array([10.0]))

    # Priors
    wm.add_factor("prior", (p0,), {"target": jnp.array([0.0])})
    wm.add_factor("prior", (p1,), {"target": jnp.array([2.0])})
    wm.add_factor("prior", (p2,), {"target": jnp.array([4.0])})

    # Room centroid
    wm.add_factor("room_centroid", (room, p0, p1, p2), {"dim": jnp.array(1)})

    # Optimize
    wm.optimize(lr=0.1, iters=400)

    # Check
    p0v = float(wm.fg.variables[p0].value[0])
    p1v = float(wm.fg.variables[p1].value[0])
    p2v = float(wm.fg.variables[p2].value[0])
    rv  = float(wm.fg.variables[room].value[0])

    assert p0v == pytest.approx(0.0, abs=1e-3)
    assert p1v == pytest.approx(2.0, abs=1e-3)
    assert p2v == pytest.approx(4.0, abs=1e-3)
    assert rv  == pytest.approx(2.0, abs=1e-3)