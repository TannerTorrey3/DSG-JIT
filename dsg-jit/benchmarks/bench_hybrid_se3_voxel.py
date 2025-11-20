# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

import time
import jax
import jax.numpy as jnp

from core.types import NodeId, Variable, FactorId, Factor
from core.factor_graph import FactorGraph
from slam.measurements import (
    prior_residual,
    odom_se3_residual,          # additive SE3 odometry
    voxel_smoothness_residual,  # voxel chain smoothness
)
from optimization.solvers import (
    gauss_newton_manifold,
    GNConfig,
)
from slam.manifold import build_manifold_metadata


def build_hybrid_graph(num_poses: int = 50, num_voxels: int = 500) -> FactorGraph:
    """
    Hybrid SE3 + Voxel chain in one factor graph.

      - SE3 pose chain:
          pose0 --odom--> pose1 -- ... --> pose_{N-1}
        with a prior on pose0 at the origin.

      - Voxel chain:
          voxel0 --smooth--> voxel1 -- ... --> voxel_{M-1}
        with a prior on voxel0 near the origin and smoothness offset ~[1,0,0].

    This exercises:
      - Manifold SE3 blocks (pose_se3)
      - Euclidean voxel blocks
      - Mixed factor types through a single Gauss-Newton manifold solve.
    """
    fg = FactorGraph()

    # --- Add SE3 poses ---
    for i in range(num_poses):
        # Slightly perturbed around [i, 0, 0, 0, 0, 0]
        init_val = jnp.array(
            [
                i + 0.05 * jnp.sin(0.1 * i),  # tx
                0.05 * jnp.cos(0.15 * i),     # ty
                0.0,                          # tz
                0.0, 0.0, 0.0,                # rotation (small)
            ]
        )
        v = Variable(id=NodeId(i), type="pose_se3", value=init_val)
        fg.add_variable(v)

    # Prior on pose0 at the SE3 origin
    fg.add_factor(
        Factor(
            id=FactorId(0),
            type="prior",
            var_ids=(NodeId(0),),
            params={"target": jnp.zeros(6), "weight": 1.0},
        )
    )

    # Odom factors between poses: +1m in x, no rotation
    meas_se3 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(num_poses - 1):
        fg.add_factor(
            Factor(
                id=FactorId(1 + i),
                type="odom_se3",
                var_ids=(NodeId(i), NodeId(i + 1)),
                params={"measurement": meas_se3, "weight": 1.0},
            )
        )

    # --- Add Voxel chain ---
    voxel_offset_id = num_poses
    voxel_dim = 3

    for j in range(num_voxels):
        # Slightly noisy around [j, 0, 0]
        init_val = jnp.array(
            [
                j + 0.05 * jnp.sin(0.05 * j),
                0.05 * jnp.cos(0.07 * j),
                0.0,
            ]
        )
        v = Variable(
            id=NodeId(voxel_offset_id + j),
            type="voxel",     # treated as Euclidean in manifold metadata
            value=init_val,
        )
        fg.add_variable(v)

    # Prior on voxel0 near the origin
    fg.add_factor(
        Factor(
            id=FactorId(1 + (num_poses - 1)),  # continue factor IDs
            type="prior",
            var_ids=(NodeId(voxel_offset_id),),
            params={"target": jnp.zeros(voxel_dim), "weight": 1.0},
        )
    )

    # Smoothness factors between neighboring voxels:
    #   residual ~ (v_{j+1} - v_j) - offset
    # with offset ~ [1, 0, 0]
    offset = jnp.array([1.0, 0.0, 0.0])
    base_factor_id = 1 + (num_poses - 1) + 1
    for j in range(num_voxels - 1):
        f_id = FactorId(base_factor_id + j)
        v_id_curr = NodeId(voxel_offset_id + j)
        v_id_next = NodeId(voxel_offset_id + j + 1)

        fg.add_factor(
            Factor(
                id=f_id,
                type="voxel_smoothness",
                var_ids=(v_id_curr, v_id_next),
                params={"offset": offset, "weight": 1.0},
            )
        )

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("voxel_smoothness", voxel_smoothness_residual)

    return fg


def run_benchmark(
    num_poses: int = 50,
    num_voxels: int = 500,
    max_iters: int = 20,
    use_jit: bool = True,
):
    print("=== Hybrid SE3 + Voxel Gauss-Newton Benchmark ===")
    print(
        f"num_poses = {num_poses}, num_voxels = {num_voxels}, "
        f"max_iters = {max_iters}, use_jit = {use_jit}"
    )

    fg = build_hybrid_graph(num_poses=num_poses, num_voxels=num_voxels)
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

    # Warmup (JIT compile if enabled)
    x_warm = solve_once(x0)
    if hasattr(x_warm, "block_until_ready"):
        x_warm.block_until_ready()

    # Benchmark
    t0 = time.time()
    x_opt = solve_once(x0)
    if hasattr(x_opt, "block_until_ready"):
        x_opt.block_until_ready()
    t1 = time.time()

    elapsed = t1 - t0
    print(f"Elapsed time: {elapsed * 1000:.3f} ms")

    # Quick sanity checks:
    values = fg.unpack_state(x_opt, index)

    pose0 = values[NodeId(0)]
    pose_last = values[NodeId(num_poses - 1)]

    voxel0 = values[NodeId(num_poses)]
    voxel_last = values[NodeId(num_poses + num_voxels - 1)]

    print(f"pose0 (opt):     {pose0}")
    print(f"poseN-1 (opt):   {pose_last}")
    print(f"voxel0 (opt):    {voxel0}")
    print(f"voxelM-1 (opt):  {voxel_last}")
    print()


if __name__ == "__main__":
    # JIT benchmark
    run_benchmark(num_poses=50, num_voxels=500, max_iters=20, use_jit=True)
    # Non-JIT benchmark
    run_benchmark(num_poses=50, num_voxels=500, max_iters=20, use_jit=False)