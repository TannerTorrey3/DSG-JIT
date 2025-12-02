# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

import time
import jax
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, Variable, FactorId, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,  # or odom_se3_geodesic_residual if that's your “default”
)
from dsg_jit.optimization.solvers import (
    gauss_newton_manifold,
    GNConfig,
)
from dsg_jit.slam.manifold import build_manifold_metadata


def build_se3_chain_graph(num_poses: int = 10):
    """
    Simple SE3 pose chain:
        pose0 --odom--> pose1 --odom--> ... --odom--> pose_{N-1}
    Prior on pose0, odom edges of +1m in x, no rotation.
    """
    fg = FactorGraph()

    # Initial guesses: slightly perturbed around ground truth [i, 0, 0, 0, 0, 0]
    for i in range(num_poses):
        init_val = jnp.array(
            [i + 0.1 * jnp.sin(0.3 * i),   # tx
             0.05 * jnp.cos(0.2 * i),     # ty
             0.0,                         # tz
             0.0, 0.0, 0.0]               # rotation
        )
        v = Variable(id=NodeId(i), type="pose_se3", value=init_val)
        fg.add_variable(v)

    # Prior on pose0 at identity
    fg.add_factor(
        Factor(
            id=FactorId(0),
            type="prior",
            var_ids=(NodeId(0),),
            params={"target": jnp.zeros(6), "weight": 1.0},
        )
    )

    # Odom factors: +1m in x
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(num_poses - 1):
        fg.add_factor(
            Factor(
                id=FactorId(i + 1),
                type="odom_se3",  # or "odom_se3_geodesic" depending on your residual
                var_ids=(NodeId(i), NodeId(i + 1)),
                params={"measurement": meas, "weight": 1.0},
            )
        )

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)

    return fg


def run_benchmark(num_poses: int = 50, max_iters: int = 20, use_jit: bool = True):
    print(f"=== SE3 Gauss-Newton Benchmark ===")
    print(f"num_poses = {num_poses}, max_iters = {max_iters}, use_jit = {use_jit}")

    fg = build_se3_chain_graph(num_poses)
    x0, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    cfg = GNConfig(
        max_iters=max_iters,
        damping=1e-3,
        max_step_norm=1.0,
    )

    # Wrap solver so we can JIT it if desired
    def solve_once(x_init):
        return gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    if use_jit:
        solve_once = jax.jit(solve_once)

    # Warmup: run once and, if JIT-compiled, force compilation + execution
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

    # Quick sanity: print pose0 / pose_last
    values = fg.unpack_state(x_opt, index)
    p0 = values[NodeId(0)]
    plast = values[NodeId(num_poses - 1)]
    print(f"pose0 (opt):   {p0}")
    print(f"poseN-1 (opt): {plast}")


if __name__ == "__main__":
    run_benchmark(num_poses=50, max_iters=20, use_jit=True)