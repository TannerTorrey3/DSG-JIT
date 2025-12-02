
import jax.numpy as jnp

from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.core.types import NodeId, Variable, Factor
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.slam.measurements import se3_chain_residual  
from dsg_jit.world.visualization import plot_factor_graph_2d, plot_factor_graph_3d
from dsg_jit.slam.manifold import build_manifold_metadata


def build_demo_graph(num_poses: int = 5) -> FactorGraph:
    fg = FactorGraph()

    # Register residuals
    fg.register_residual("odom_se3", se3_chain_residual) 

    # Variables: simple SE3 poses along x
    for i in range(num_poses):
        nid = NodeId(i)
        fg.add_variable(
            Variable(
                id=nid,
                type="pose_se3",
                value=jnp.zeros(6),  # initial guess
            )
        )

    # Factors: odometry between consecutive poses
    fid = 0
    for i in range(num_poses - 1):
        meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fg.add_factor(
            Factor(
                id=fid,
                type="odom_se3",
                var_ids=(NodeId(i), NodeId(i + 1)),
                params={"measurement": meas, "weight": 1.0},
            )
        )
        fid += 1

    return fg


def main():
    fg = build_demo_graph(num_poses=5)

    block_slices, manifold_types = build_manifold_metadata(fg)
    x0, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    cfg = GNConfig(max_iters=10, damping=1e-3, max_step_norm=1.0)

    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )
    values = fg.unpack_state(x_opt, index)

    print("Optimized poses:")
    for nid, v in values.items():
        print(nid, v)
        fg.variables[nid].value = v

    # Visualize
    plot_factor_graph_3d(fg)


if __name__ == "__main__":
    main()