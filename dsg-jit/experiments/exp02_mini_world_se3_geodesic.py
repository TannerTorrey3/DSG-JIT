from __future__ import annotations
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    pose_place_attachment_residual,
    odom_se3_residual,
)
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def run_experiment():
    wm = WorldModel()

    # --- Create 3 SE(3) robot poses ---
    pose0_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]),
    )
    pose1_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([0.8, 0.1, -0.05, 0.05, 0.03, -0.01]),
    )
    pose2_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([1.9, -0.2, 0.10, -0.02, 0.01, 0.02]),
    )

    # --- Scene graph: places (1D) ---
    place0_id = wm.add_variable(var_type="place1d", value=jnp.array([-0.2]))
    place1_id = wm.add_variable(var_type="place1d", value=jnp.array([1.4]))
    place2_id = wm.add_variable(var_type="place1d", value=jnp.array([2.1]))

    # --- Scene graph: room centroid (1D) ---
    room_id = wm.add_variable(var_type="place1d", value=jnp.array([5.0]))

    # --- Factors ---
    # Prior on pose0
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Odom (SE(3) geodesic measurement along +x)
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": meas},
    )
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose1_id, pose2_id),
        params={"measurement": meas},
    )

    # Place attachments (1D) for each pose-place pair
    attach_params = {
        "pose_dim": jnp.array(6),
        "place_dim": jnp.array(1),
        "pose_coord_index": jnp.array(0),
    }

    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose0_id, place0_id),
        params=dict(attach_params),
    )
    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose1_id, place1_id),
        params=dict(attach_params),
    )
    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose2_id, place2_id),
        params=dict(attach_params),
    )

    # Room centroid attachment from pose1 to room
    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose1_id, room_id),
        params=dict(attach_params),
    )

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # Capture initial values before optimization for printing
    initial_values = {nid: var.value for nid, var in wm.fg.variables.items()}

    # --- Optimize ---
    x0, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=40, damping=1e-3, max_step_norm=0.5)
    x_opt = gauss_newton(residual_fn, x0, cfg)

    values = wm.unpack_state(x_opt, index)

    print("\n=== INITIAL STATE ===")
    for nid, val in initial_values.items():
        print(f"{nid}: {val}")

    print("\n=== OPTIMIZED STATE (SE3 Manifold) ===")
    for nid, val in values.items():
        print(f"{nid}: {val}")

    return values

if __name__ == "__main__":
    run_experiment()