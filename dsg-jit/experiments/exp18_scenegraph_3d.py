"""
exp18_scenegraph_3d.py

Hero-style 3D Scene Graph visualization for DSG-JIT.

- Builds a small SE(3) odometry chain in a FactorGraph
- Solves with manifold Gauss-Newton
- Exports pose nodes as VisNode / VisEdge
- Adds semantic nodes (rooms, places, objects)
- Renders a layered 3D scene graph with metric + semantic edges.

Usage (from repo root):

    export PYTHONPATH=src
    python3 experiments/exp18_scenegraph_3d.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.optimization.solvers import GNConfig, gauss_newton_manifold
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.slam.measurements import prior_residual, odom_se3_geodesic_residual
from dsg_jit.world.visualization import (
    VisNode,
    VisEdge,
    export_factor_graph_for_vis,
)


# ---------------------------------------------------------------------------
# Helper: build & solve a simple SE3 odom chain
# ---------------------------------------------------------------------------


def build_pose_chain_factor_graph(num_poses: int = 5) -> Tuple[FactorGraph, jnp.ndarray]:
    """
    Build a 1D SE(3) pose chain factor graph on the x-axis.

    pose0 --odom--> pose1 --odom--> ... --odom--> pose_{N-1}

    We add:
      - Variables: pose_i (type "pose_se3", 6D se(3) vector)
      - Factors:
          * prior on pose0 at the origin
          * odom_se3 factors between consecutive poses with dx = [1, 0, ..., 0]

    Returns:
        fg: constructed FactorGraph
        x0: initial packed state
    """
    fg = FactorGraph()

    # Register residual functions
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_geodesic_residual)

    # Add pose variables (initialized at zero)
    for i in range(num_poses):
        nid = NodeId(i)
        v = Variable(
            id=nid,
            type="pose_se3",
            value=jnp.zeros(6, dtype=jnp.float32),
        )
        fg.add_variable(v)

    # Prior on pose0 at the origin
    prior_factor = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={
            "target": jnp.zeros(6, dtype=jnp.float32),
            "weight": 100.0,
        },
    )
    fg.add_factor(prior_factor)

    # Odom factors between consecutive poses
    for i in range(1, num_poses):
        fid = FactorId(i)
        meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        odom_factor = Factor(
            id=fid,
            type="odom_se3",
            var_ids=(NodeId(i - 1), NodeId(i)),
            params={
                "measurement": meas,
                "weight": 1.0,
            },
        )
        fg.add_factor(odom_factor)

    # Initial state
    x0, _ = fg.pack_state()
    return fg, x0


def solve_pose_chain(fg: FactorGraph, x0: jnp.ndarray) -> jnp.ndarray:
    """
    Solve the SE3 pose chain using the manifold-aware Gauss-Newton solver.
    """
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )
    return x_opt


# ---------------------------------------------------------------------------
# Scene-graph 3D visualization from VisNode / VisEdge
# ---------------------------------------------------------------------------


def plot_scene_graph_3d_from_nodes(
    nodes: List[VisNode],
    metric_edges: List[VisEdge] | None = None,
    semantic_edges: List[VisEdge] | None = None,
    z_by_type: Dict[str, float] | None = None,
    show_labels: bool = True,
) -> None:
    """
    3D renderer for a generic scene graph defined over VisNode / VisEdge.

    Args:
        nodes: List of VisNode (pose, voxel, room, place, object, ...)
        metric_edges: Edges representing metric constraints (SLAM factors).
        semantic_edges: Edges representing semantic relations (room-place, place-object, ...).
        z_by_type: Optional map of node.type -> z-height override for layered visualization.
        show_labels: Whether to draw labels next to nodes.
    """
    if metric_edges is None:
        metric_edges = []
    if semantic_edges is None:
        semantic_edges = []

    # Color map per node type
    type_to_color: Dict[str, str] = {
        "pose": "C0",
        "voxel": "C1",
        "place": "C2",
        "room": "C3",
        "object": "C4",
        "other": "C5",
    }

    # Build lookup for positions
    node_pos: Dict[NodeId, jnp.ndarray] = {}

    for n in nodes:
        p = n.position
        # Optionally lift based on type
        if z_by_type is not None and n.type in z_by_type:
            p = p.at[2].set(z_by_type[n.type])
        node_pos[n.id] = p

    # Prepare for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Metric edges: solid black
    for e in metric_edges:
        var_ids = list(e.var_ids)
        if len(var_ids) < 2:
            continue
        for i in range(len(var_ids) - 1):
            a = node_pos.get(var_ids[i])
            b = node_pos.get(var_ids[i + 1])
            if a is None or b is None:
                continue
            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[1]), float(b[1])],
                [float(a[2]), float(b[2])],
                linewidth=0.8,
                alpha=0.4,
                color="k",
            )

    # Semantic edges: dashed, colored
    for e in semantic_edges:
        var_ids = list(e.var_ids)
        if len(var_ids) < 2:
            continue
        for i in range(len(var_ids) - 1):
            a = node_pos.get(var_ids[i])
            b = node_pos.get(var_ids[i + 1])
            if a is None or b is None:
                continue
            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[1]), float(b[1])],
                [float(a[2]), float(b[2])],
                linewidth=0.9,
                alpha=0.6,
                linestyle="--",
                color="C6",
            )

    # Draw nodes
    for n in nodes:
        c = type_to_color.get(n.type, "k")
        p = node_pos[n.id]
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        ax.scatter(x, y, z, s=35, c=c)
        if show_labels:
            ax.text(x, y, z, n.label, fontsize=7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("DSG-JIT Scene Graph (3D)")

    # Equal aspect ratio in 3D
    xs = [float(p[0]) for p in node_pos.values()]
    ys = [float(p[1]) for p in node_pos.values()]
    zs = [float(p[2]) for p in node_pos.values()]

    if xs:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        if max_range <= 0.0:
            max_range = 1.0

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)
        cz = 0.5 * (z_max + z_min)

        pad = 0.1 * max_range
        half = 0.5 * max_range + pad

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Build a "hero" scene graph: rooms, places, objects, poses
# ---------------------------------------------------------------------------


def build_semantic_scene(nodes_fg: List[VisNode]) -> Tuple[List[VisNode], List[VisEdge]]:
    """
    Given the pose nodes exported from the factor graph, create a semantic
    layer with rooms, places, and objects.

    Layout (conceptual):
      - Two rooms: Room A (left), Room B (right)
      - Three places along the trajectory
      - Three objects attached to places
      - Places & poses also connected (pose_at_place)
    """
    # Pose nodes
    pose_nodes = [n for n in nodes_fg if n.type == "pose"]
    pose_nodes_sorted = sorted(pose_nodes, key=lambda n: float(n.position[0]))

    # Convenience: x positions of some key poses
    p0 = pose_nodes_sorted[0]
    p2 = pose_nodes_sorted[len(pose_nodes_sorted) // 2]
    p_last = pose_nodes_sorted[-1]

    # Start new NodeIds after existing ones
    max_existing = max(int(n.id) for n in nodes_fg) if nodes_fg else 0
    next_id = max_existing + 1

    vis_nodes: List[VisNode] = []

    # --- Rooms ---
    room0_id = NodeId(next_id)
    room1_id = NodeId(next_id + 1)
    next_id += 2

    room0_pos = jnp.array([float(p0.position[0]) - 0.5, 2.0, 0.0])
    room1_pos = jnp.array([float(p_last.position[0]) + 0.5, 2.0, 0.0])

    room0 = VisNode(id=room0_id, type="room", position=room0_pos, label="room:A")
    room1 = VisNode(id=room1_id, type="room", position=room1_pos, label="room:B")

    vis_nodes.extend([room0, room1])

    # --- Places ---
    place_nodes: List[VisNode] = []
    place_ids: List[NodeId] = []

    def make_place(base_pose: VisNode, dy: float, label: str) -> VisNode:
        nonlocal next_id
        nid = NodeId(next_id)
        next_id += 1
        pos = base_pose.position + jnp.array([0.0, dy, 0.0])
        node = VisNode(id=nid, type="place", position=pos, label=label)
        place_ids.append(nid)
        place_nodes.append(node)
        return node

    place0 = make_place(p0, dy=0.5, label="place:0")
    place1 = make_place(p2, dy=0.5, label="place:1")
    place2 = make_place(p_last, dy=0.5, label="place:2")

    vis_nodes.extend(place_nodes)

    # --- Objects ---
    object_nodes: List[VisNode] = []
    object_ids: List[NodeId] = []

    def make_object(base_place: VisNode, offset, label: str) -> VisNode:
        nonlocal next_id
        nid = NodeId(next_id)
        next_id += 1
        pos = base_place.position + jnp.array(offset)
        node = VisNode(id=nid, type="object", position=pos, label=label)
        object_ids.append(nid)
        object_nodes.append(node)
        return node

    obj0 = make_object(place0, offset=[0.3, 0.2, 0.2], label="obj:chair")
    obj1 = make_object(place1, offset=[-0.2, 0.1, 0.3], label="obj:table")
    obj2 = make_object(place2, offset=[0.1, -0.1, 0.25], label="obj:plant")

    vis_nodes.extend(object_nodes)

    # --- Semantic edges ---
    semantic_edges: List[VisEdge] = []

    # room-place relations
    semantic_edges.append(VisEdge(var_ids=(room0_id, place0.id), factor_type="room_place"))
    semantic_edges.append(VisEdge(var_ids=(room0_id, place1.id), factor_type="room_place"))
    semantic_edges.append(VisEdge(var_ids=(room1_id, place2.id), factor_type="room_place"))

    # place-object relations
    semantic_edges.append(VisEdge(var_ids=(place0.id, obj0.id), factor_type="place_object"))
    semantic_edges.append(VisEdge(var_ids=(place1.id, obj1.id), factor_type="place_object"))
    semantic_edges.append(VisEdge(var_ids=(place2.id, obj2.id), factor_type="place_object"))

    # pose-place relations (robot at place)
    semantic_edges.append(VisEdge(var_ids=(p0.id, place0.id), factor_type="pose_at_place"))
    semantic_edges.append(VisEdge(var_ids=(p2.id, place1.id), factor_type="pose_at_place"))
    semantic_edges.append(VisEdge(var_ids=(p_last.id, place2.id), factor_type="pose_at_place"))

    return vis_nodes, semantic_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1) Build and solve pose chain
    fg, x0 = build_pose_chain_factor_graph(num_poses=5)
    x_opt = solve_pose_chain(fg, x0)

    # Inspect optimized poses
    _, index = fg.pack_state()
    values = fg.unpack_state(x_opt, index)

    print("=== Optimized poses (se(3) vector) ===")
    for i in range(len(values)):
        v = values[NodeId(i)]
        print(f"pose{i}: {v}")

    # 2) Export factor graph to VisNode/VisEdge
    nodes_fg, edges_fg = export_factor_graph_for_vis(fg)

    # 3) Build semantic layer (rooms, places, objects)
    semantic_nodes, semantic_edges = build_semantic_scene(nodes_fg)

    # 4) Combine nodes and visualize
    all_nodes = nodes_fg + semantic_nodes
    metric_edges = list(edges_fg)

    z_by_type = {
        "room": 0.0,
        "place": 0.5,
        "pose": 1.0,
        "object": 1.5,
    }

    plot_scene_graph_3d_from_nodes(
        all_nodes,
        metric_edges=metric_edges,
        semantic_edges=semantic_edges,
        z_by_type=z_by_type,
        show_labels=True,
    )


if __name__ == "__main__":
    main()