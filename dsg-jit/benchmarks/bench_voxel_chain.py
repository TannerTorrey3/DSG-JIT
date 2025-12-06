# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

import time
import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    voxel_smoothness_residual,
)
from dsg_jit.optimization.solvers import (
    gauss_newton_manifold,
    GNConfig,
)
from dsg_jit.slam.manifold import build_manifold_metadata


def build_voxel_chain_world_model(num_voxels: int = 100):
    """
    Simple 1D-ish voxel chain in R^3 backed by a WorldModel:

        v0 --smooth--> v1 --smooth--> ... --smooth--> v_{N-1}

    - Each voxel is a 3D point (type 'voxel').
    - Prior on v0 near [0, 0, 0].
    - Smoothness factors encourage neighboring voxels to be similar
      (r ~ v_{i+1} - v_i).

    This is a pure Euclidean problem but still flows through the same
    world-model + manifold GN pipeline as SE(3).
    """
    wm = WorldModel()
    voxel_ids = []

    # Initial guesses: voxels roughly along +x with some noise
    for i in range(num_voxels):
        init_val = jnp.array(
            [
                float(i) + 0.1 * jnp.sin(0.2 * i),
                0.05 * jnp.cos(0.3 * i),
                0.0,
            ]
        )
        vid = wm.add_variable(var_type="voxel", value=init_val)
        voxel_ids.append(vid)

    # Prior on v0 around the origin
    wm.add_factor(
        f_type="prior",
        var_ids=(voxel_ids[0],),
        params={
            "target": jnp.zeros(3),
            "weight": 1.0,
        },
    )

    # Smoothness factors between neighboring voxels
    smooth_weight = 1.0
    offset = jnp.zeros(3)  # no preferred offset between neighbors
    for i in range(num_voxels - 1):
        wm.add_factor(
            f_type="voxel_smoothness",
            var_ids=(voxel_ids[i], voxel_ids[i + 1]),
            params={
                "weight": smooth_weight,
                "offset": offset,
            },
        )

    # Register residuals on the world model
    wm.register_residual("prior", prior_residual)
    wm.register_residual("voxel_smoothness", voxel_smoothness_residual)

    return wm, voxel_ids


def run_benchmark(num_voxels: int = 500, max_iters: int = 20, use_jit: bool = True):
    print("=== Voxel Chain Gauss-Newton Benchmark (WorldModel) ===")
    print(f"num_voxels = {num_voxels}, max_iters = {max_iters}, use_jit = {use_jit}")

    wm, voxel_ids = build_voxel_chain_world_model(num_voxels)
    x0, index = wm.pack_state()
    residual_fn = wm.build_residual()

    # Build manifold metadata from the world model's factor graph and packed state
    packed_state = (x0, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )

    cfg = GNConfig(
        max_iters=max_iters,
        damping=1e-3,
        max_step_norm=1.0,
    )

    def solve_once(x_init):
        return gauss_newton_manifold(
            residual_fn,
            x_init,
            block_slices,
            manifold_types,
            cfg,
        )

    if use_jit:
        solve_once = jax.jit(solve_once)

    # Warmup (forces compilation when use_jit=True)
    x_warm = solve_once(x0)
    if hasattr(x_warm, "block_until_ready"):
        x_warm.block_until_ready()

    # Benchmark
    t0 = time.time()
    x_opt = solve_once(x0)
    if hasattr(x_opt, "block_until_ready"):
        x_opt.block_until_ready()
    t1 = time.time()

    elapsed = (t1 - t0) * 1000.0
    print(f"Elapsed time: {elapsed:.3f} ms")

    # Quick sanity check: first and last voxel
    values = wm.unpack_state(x_opt, index)
    v0 = values[voxel_ids[0]]
    vlast = values[voxel_ids[-1]]

    print(f"voxel0 (opt):   {v0}")
    print(f"voxelN-1 (opt): {vlast}")


if __name__ == "__main__":
    # Example:
    #   PYTHONPATH=src python3 benchmarks/bench_voxel_chain.py
    run_benchmark(num_voxels=500, max_iters=20, use_jit=True)
    run_benchmark(num_voxels=500, max_iters=20, use_jit=False)