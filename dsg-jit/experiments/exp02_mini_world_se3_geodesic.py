from __future__ import annotations
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    pose_place_attachment_residual,
    odom_se3_residual
)
from dsg_jit.optimization.solvers import  GNConfig, gauss_newton


def run_experiment():

    fg = FactorGraph()

    # --- Create 3 SE(3) robot poses ---
    pose0 = Variable(NodeId(0), "pose_se3", jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]))
    pose1 = Variable(NodeId(1), "pose_se3", jnp.array([0.8,  0.1, -0.05, 0.05, 0.03, -0.01]))
    pose2 = Variable(NodeId(2), "pose_se3", jnp.array([1.9, -0.2, 0.10, -0.02,0.01, 0.02]))

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(pose2)

    # --- Scene graph: places ---
    place0 = Variable(NodeId(3), "place1d", jnp.array([-0.2]))
    place1 = Variable(NodeId(4), "place1d", jnp.array([ 1.4]))
    place2 = Variable(NodeId(5), "place1d", jnp.array([ 2.1]))

    fg.add_variable(place0)
    fg.add_variable(place1)
    fg.add_variable(place2)

    # --- Scene graph: room centroid ---
    room = Variable(NodeId(6), "place1d", jnp.array([5.0]))
    fg.add_variable(room)

    # --- Factors ---
    # Prior
    fg.add_factor(Factor(0, "prior", (NodeId(0),), {"target": jnp.zeros(6)}))

    # Odom (true SE(3) geodesic)
    fg.add_factor(Factor(1, "odom_se3", (NodeId(0), NodeId(1)), {
        "measurement": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }))
    fg.add_factor(Factor(2, "odom_se3", (NodeId(1), NodeId(2)), {
        "measurement": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }))

    # Place attachments (1D)
    fg.add_factor(Factor(3, "pose_place_attachment", (NodeId(0), NodeId(3)), {
        "pose_dim": jnp.array(6),
        "place_dim": jnp.array(1),
        "pose_coord_index": jnp.array(0),
    }))
    fg.add_factor(Factor(4, "pose_place_attachment", (NodeId(1), NodeId(4)), {
        "pose_dim": jnp.array(6),
        "place_dim": jnp.array(1),
        "pose_coord_index": jnp.array(0),
    }))
    fg.add_factor(Factor(5, "pose_place_attachment", (NodeId(2), NodeId(5)), {
        "pose_dim": jnp.array(6),
        "place_dim": jnp.array(1),
        "pose_coord_index": jnp.array(0),
    }))

    # Room centroid attachment
    fg.add_factor(Factor(6, "pose_place_attachment", (NodeId(1), NodeId(6)), {
        "pose_dim": jnp.array(6),
        "place_dim": jnp.array(1),
        "pose_coord_index": jnp.array(0),
    }))

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # --- Optimize ---
    x0, index = fg.pack_state()
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=40, damping=1e-3, max_step_norm=0.5)
    x_opt = gauss_newton(residual_fn, x0, cfg)

    values = fg.unpack_state(x_opt, index)

    print("\n=== INITIAL STATE ===")
    for nid, var in fg.variables.items():
        print(f"{nid}: {var.value}")

    print("\n=== OPTIMIZED STATE (SE3 Manifold) ===")
    for nid, val in values.items():
        print(f"{nid}: {val}")

    return values

if __name__ == "__main__":
    run_experiment()