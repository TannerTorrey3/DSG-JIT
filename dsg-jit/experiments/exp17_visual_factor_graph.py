import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.slam.measurements import se3_chain_residual  
from dsg_jit.world.visualization import plot_factor_graph_2d, plot_factor_graph_3d
from dsg_jit.slam.manifold import build_manifold_metadata


def build_demo_graph(num_poses: int = 5) -> WorldModel:
    wm = WorldModel()

    # Register residuals at the WorldModel level
    wm.register_residual("odom_se3", se3_chain_residual)

    # Variables: simple SE3 poses along x (initialized at zero)
    pose_ids = []
    for _ in range(num_poses):
        nid = wm.add_variable(
            var_type="pose_se3",
            value=jnp.zeros(6),  # initial guess
        )
        pose_ids.append(nid)

    # Factors: odometry between consecutive poses
    for i in range(num_poses - 1):
        meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wm.add_factor(
            f_type="odom_se3",
            var_ids=(pose_ids[i], pose_ids[i + 1]),
            params={"measurement": meas, "weight": 1.0},
        )

    return wm


def main():
    wm = build_demo_graph(num_poses=5)

    # Build manifold metadata using the WorldModel's factor graph and packed state
    x0, index = wm.pack_state()
    packed_state = (x0, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )
    residual_fn = wm.build_residual()
    cfg = GNConfig(max_iters=10, damping=1e-3, max_step_norm=1.0)

    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )
    values = wm.unpack_state(x_opt, index)

    print("Optimized poses:")
    for nid, v in values.items():
        print(nid, v)
        # Update the stored values in the underlying factor graph for visualization
        wm.fg.variables[nid].value = v

    # Visualize the underlying factor graph structure
    plot_factor_graph_3d(wm.fg)


if __name__ == "__main__":
    main()