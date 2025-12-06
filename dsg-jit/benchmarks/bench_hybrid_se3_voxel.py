# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

import time
import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,          # additive SE3 odometry
    voxel_smoothness_residual,  # voxel chain smoothness
)
from dsg_jit.optimization.solvers import (
    gauss_newton_manifold,
    GNConfig,
)
from dsg_jit.slam.manifold import build_manifold_metadata


def build_hybrid_world_model(
    num_poses: int = 50, num_voxels: int = 500
) -> tuple[WorldModel, list[int]]:
    """
    Hybrid SE3 + Voxel chain in one world model.

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
    wm = WorldModel()

    pose_ids: list[int] = []
    voxel_ids: list[int] = []

    # --- Add SE3 poses ---
    for i in range(num_poses):
        # Slightly perturbed around [i, 0, 0, 0, 0, 0]
        init_val = jnp.array(
            [
                i + 0.05 * jnp.sin(0.1 * i),  # tx
                0.05 * jnp.cos(0.15 * i),     # ty
                0.0,                          # tz
                0.0,
                0.0,
                0.0,                          # rotation (small)
            ]
        )
        pose_id = wm.add_variable(var_type="pose_se3", value=init_val)
        pose_ids.append(pose_id)

    # Prior on pose0 at the SE3 origin
    wm.add_factor(
        f_type="prior",
        var_ids=(pose_ids[0],),
        params={"target": jnp.zeros(6), "weight": 1.0},
    )

    # Odom factors between poses: +1m in x, no rotation
    meas_se3 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(num_poses - 1):
        wm.add_factor(
            f_type="odom_se3",
            var_ids=(pose_ids[i], pose_ids[i + 1]),
            params={"measurement": meas_se3, "weight": 1.0},
        )

    # --- Add Voxel chain ---
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
        v_id = wm.add_variable(
            var_type="voxel",  # treated as Euclidean in manifold metadata
            value=init_val,
        )
        voxel_ids.append(v_id)

    # Prior on voxel0 near the origin
    wm.add_factor(
        f_type="prior",
        var_ids=(voxel_ids[0],),
        params={"target": jnp.zeros(voxel_dim), "weight": 1.0},
    )

    # Smoothness factors between neighboring voxels:
    #   residual ~ (v_{j+1} - v_j) - offset
    # with offset ~ [1, 0, 0]
    offset = jnp.array([1.0, 0.0, 0.0])
    for j in range(num_voxels - 1):
        wm.add_factor(
            f_type="voxel_smoothness",
            var_ids=(voxel_ids[j], voxel_ids[j + 1]),
            params={"offset": offset, "weight": 1.0},
        )

    # Register residuals on the world model
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("voxel_smoothness", voxel_smoothness_residual)

    # We return the world model and the pose / voxel ID lists so the caller
    # can interpret the optimized state.
    return wm, pose_ids + voxel_ids


def run_benchmark(
    num_poses: int = 50,
    num_voxels: int = 500,
    max_iters: int = 20,
    use_jit: bool = True,
):
    print("=== Hybrid SE3 + Voxel Gauss-Newton Benchmark (WorldModel) ===")
    print(
        f"num_poses = {num_poses}, num_voxels = {num_voxels}, "
        f"max_iters = {max_iters}, use_jit = {use_jit}"
    )

    wm, ids = build_hybrid_world_model(num_poses=num_poses, num_voxels=num_voxels)

    # Pack initial state and build the vmap-optimized residual
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
    values = wm.unpack_state(x_opt, index)

    pose0 = values[ids[0]]
    pose_last = values[ids[num_poses - 1]]

    voxel0 = values[ids[num_poses]]
    voxel_last = values[ids[num_poses + num_voxels - 1]]

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