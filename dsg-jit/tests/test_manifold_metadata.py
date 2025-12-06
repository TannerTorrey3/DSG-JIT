from __future__ import annotations

import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.manifold import build_manifold_metadata


def test_build_manifold_metadata_basic():
    wm = WorldModel()

    # Two poses + one place, created via WorldModel so we use the actual
    # NodeIds assigned by the underlying FactorGraph.
    pose0_nid = wm.add_variable(var_type="pose_se3", value=jnp.zeros(6))
    pose1_nid = wm.add_variable(var_type="pose_se3", value=jnp.ones(6))
    place0_nid = wm.add_variable(var_type="place1d", value=jnp.array([0.5]))

    packed_state = wm.pack_state()
    x, index = packed_state
    block_slices, manifold_types = build_manifold_metadata(packed_state=packed_state, fg=wm.fg)

    # Check slices align with index (start, length)
    for nid in [pose0_nid, pose1_nid, place0_nid]:
        sl = block_slices[nid]
        idx = index[nid]  # usually (start, length)

        if isinstance(idx, tuple):
            start, length = idx
            assert sl.start == start
            assert sl.stop == start + length
        elif isinstance(idx, slice):
            # If the index is already a slice, just compare directly
            assert sl == idx
        else:
            raise AssertionError(f"Unexpected index type for {nid}: {type(idx)}")

    # Check manifold tags
    assert manifold_types[pose0_nid] == "se3"
    assert manifold_types[pose1_nid] == "se3"
    assert manifold_types[place0_nid] == "euclidean"