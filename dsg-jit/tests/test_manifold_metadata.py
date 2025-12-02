
from __future__ import annotations

import jax.numpy as jnp

from dsg_jit.core.types import NodeId, Variable
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.manifold import build_manifold_metadata


def test_build_manifold_metadata_basic():
    fg = FactorGraph()

    # Two poses + one place
    pose0 = Variable(id=NodeId(0), type="pose_se3", value=jnp.zeros(6))
    pose1 = Variable(id=NodeId(1), type="pose_se3", value=jnp.ones(6))
    place0 = Variable(id=NodeId(2), type="place1d", value=jnp.array([0.5]))

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(place0)

    x, index = fg.pack_state()

    block_slices, manifold_types = build_manifold_metadata(fg)

    # Check slices align with index (start, length)
    for nid in [NodeId(0), NodeId(1), NodeId(2)]:
        sl = block_slices[nid]
        idx = index[nid]  # usually (start, length)

        if isinstance(idx, tuple):
            start, length = idx
            assert sl.start == start
            assert sl.stop == start + length
        elif isinstance(idx, slice):
            # If FactorGraph already gave slices, just compare directly
            assert sl == idx
        else:
            raise AssertionError(f"Unexpected index type for {nid}: {type(idx)}")

    # Check manifold tags
    assert manifold_types[NodeId(0)] == "se3"
    assert manifold_types[NodeId(1)] == "se3"
    assert manifold_types[NodeId(2)] == "euclidean"