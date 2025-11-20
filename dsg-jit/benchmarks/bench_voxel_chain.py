# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

import time
import jax
import jax.numpy as jnp

from core.types import NodeId, Variable, FactorId, Factor
from core.factor_graph import FactorGraph
from slam.measurements import (
    prior_residual,
    voxel_smoothness_residual,
)
from optimization.solvers import (
    gauss_newton_manifold,
    GNConfig,
)
from slam.manifold import build_manifold_metadata


def build_voxel_chain_graph(num_voxels: int = 100):
    """
    Simple 1D-ish voxel chain in R^3:

        v0 --smooth--> v1 --smooth--> ... --smooth--> v_{N-1}

    - Each voxel is a 3D point (type 'voxel3d').
    - Prior on v0 near [0, 0, 0].
    - Smoothness factors encourage neighboring voxels to be similar
      (r ~ v_{i+1} - v_i).

    This is a pure Euclidean problem but still flows through the same
    FactorGraph + manifold GN pipeline as SE(3).
    """
    fg = FactorGraph()

    # Initial guesses: voxels roughly along +x with some noise
    for i in range(num_voxels):
        init_val = jnp.array([
            float(i) + 0.1 * jnp.sin(0.2 * i),
            0.05 * jnp.cos(0.3 * i),
            0.0,
        ])
        v = Variable(id=NodeId(i), type="voxel3d", value=init_val)
        fg.add_variable(v)

    # Prior on v0 around the origin
    fg.add_factor(
        Factor(
            id=FactorId(0),
            type="prior",
            var_ids=(NodeId(0),),
            params={
                "target": jnp.zeros(3),
                "weight": 1.0,
            },
        )
    )

    # Smoothness factors between neighboring voxels
    smooth_weight = 1.0
    offset = jnp.zeros(3)  # no preferred offset between neighbors
    for i in range(num_voxels - 1):
        fg.add_factor(
            Factor(
                id=FactorId(i + 1),
                type="voxel_smoothness",
                var_ids=(NodeId(i), NodeId(i + 1)),
                params={
                    "weight": smooth_weight,
                    "offset": offset,
                },
            )
        )

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("voxel_smoothness", voxel_smoothness_residual)

    return fg


def run_benchmark(num_voxels: int = 500, max_iters: int = 20, use_jit: bool = True):
    print("=== Voxel Chain Gauss-Newton Benchmark ===")
    print(f"num_voxels = {num_voxels}, max_iters = {max_iters}, use_jit = {use_jit}")

    fg = build_voxel_chain_graph(num_voxels)
    x0, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

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
    values = fg.unpack_state(x_opt, index)
    v0 = values[NodeId(0)]
    vlast = values[NodeId(num_voxels - 1)]

    print(f"voxel0 (opt):   {v0}")
    print(f"voxelN-1 (opt): {vlast}")


if __name__ == "__main__":
    # Example:
    #   PYTHONPATH=src python3 benchmarks/bench_voxel_chain.py
    run_benchmark(num_voxels=500, max_iters=20, use_jit=True)
    run_benchmark(num_voxels=500, max_iters=20, use_jit=False)