# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Visualization utilities for DSG-JIT.

This module provides lightweight 2D and 3D rendering tools for visualizing
factor graphs, scene graphs, and mixed-level semantic structures. It is
designed to support both debugging and demonstration of DSG-JIT’s hierarchical
representations, including robot poses, voxel cells, places, rooms, and
arbitrary semantic objects.

The visualization pipeline follows three main steps:

1. **Exporting graph data**  
   `export_factor_graph_for_vis()` converts an internal `FactorGraph` into
   color-coded `VisNode` and `VisEdge` lists. Variable types such as
   `pose_se3`, `voxel_cell`, `place1d`, and `room1d` are mapped to coarse
   visualization categories, and heuristic 3D positions are extracted for
   rendering.

2. **2D top-down rendering**  
   `plot_factor_graph_2d()` produces a Matplotlib top-down view (x–y plane)
   with automatically computed bounds, node type coloring, and optional label
   rendering. This is especially useful for SE(3) SLAM chains, grid-based
   voxel fields, and planar semantic graphs.

3. **Full 3D scene graph rendering**  
   `plot_factor_graph_3d()` draws a complete 3D view of poses, voxels, places,
   rooms, and objects. Edges between nodes represent geometric or semantic
   relationships. Aspect ratios are normalized so spatial structure remains
   visually meaningful regardless of scale.

These visualizers are intentionally decoupled from the high-level world model
(`SceneGraphWorld`) so they can be used directly on raw factor graphs produced
by optimization procedures or experiment scripts.

Example usage is provided in:
- `experiments/exp17_visual_factor_graph.py` (basic 2D + 3D factor graph)
- `experiments/exp18_scenegraph_3d.py` (HYDRA-style multi-level scene graph)
- `experiments/exp18_scenegraph_demo.py` (HYDRA-style 2D + 3D scene graph)

Module contents:
    - `VisNode`: Lightweight typed node container for visualization.
    - `VisEdge`: Lightweight edge container (factor connections).
    - `_infer_node_type()`: Maps variable types → canonical visualization types.
    - `_extract_position()`: Extracts a 3D coordinate from variable states.
    - `export_factor_graph_for_vis()`: Converts a FactorGraph → vis nodes & edges.
    - `plot_factor_graph_2d()`: Renders a 2D top-down view of the graph.
    - `plot_factor_graph_3d()`: Renders a full 3D scene graph with semantic layers.

This module is designed to be extendable—for example:
- Additional node types can be added via `_infer_node_type`.
- SceneGraphWorld can later provide richer semantic annotations.
- Future versions may support interactive or WebGL visualizations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

from core.types import NodeId
from core.factor_graph import FactorGraph

NodeType = Literal["pose", "voxel", "place", "room", "other"]


@dataclass
class VisNode:
    """Lightweight node representation for visualization."""
    id: NodeId
    type: NodeType
    position: jnp.ndarray  # shape (3,)
    label: str


@dataclass
class VisEdge:
    """Lightweight edge representation for visualization."""
    var_ids: Tuple[NodeId, ...]
    factor_type: str


def _infer_node_type(var_type: str) -> NodeType:
    """Map internal variable.type strings to coarse visualization types."""
    if var_type == "pose_se3":
        return "pose"
    if var_type == "voxel_cell":
        return "voxel"
    if var_type == "place1d":
        return "place"
    if var_type == "room1d":
        return "room"
    return "other"


def _extract_position(var_type: str, value: jnp.ndarray) -> jnp.ndarray:
    """
    Heuristic: try to get a 3D position for visualization.

    - pose_se3: take translation (first 3 elements)
    - voxel_cell: assume first 3 are position
    - place1d/room1d:
        * if len(value) >= 3: treat first 3 entries as 3D position
        * else: embed scalar along x-axis, with y offset for rooms
    - fallback: origin
    """
    v = jnp.asarray(value)

    if var_type == "pose_se3":
        if v.shape[0] >= 3:
            return v[:3]
        return jnp.zeros(3)

    if var_type == "voxel_cell":
        if v.shape[0] >= 3:
            return v[:3]
        return jnp.zeros(3)

    if var_type in ("place1d", "room1d"):
        # If user provided full 3D, use it directly
        if v.shape[0] >= 3:
            return v[:3]
        # Otherwise embed 1D along x with small y-offset for rooms
        x = float(v[0]) if v.shape[0] >= 1 else 0.0
        y = 0.0 if var_type == "place1d" else 1.0
        return jnp.array([x, y, 0.0])

    # fallback
    return jnp.zeros(3)


def export_factor_graph_for_vis(fg: FactorGraph) -> Tuple[List[VisNode], List[VisEdge]]:
    """
    Export a FactorGraph into a visualization-friendly node/edge list.

    This does *not* require any SceneGraphWorld; it just uses variables/factors.

    :param fg: The factor graph to visualize.
    :return: (nodes, edges) where nodes is a list of VisNode and edges is a list of VisEdge.
    """
    nodes: List[VisNode] = []
    edges: List[VisEdge] = []

    # Nodes
    for nid, var in fg.variables.items():
        ntype = _infer_node_type(var.type)
        pos = _extract_position(var.type, var.value)
        nodes.append(
            VisNode(
                id=nid,
                type=ntype,
                position=pos,
                label=f"{ntype}:{int(nid)}",
            )
        )

    # Edges (one edge per factor, between all its variables)
    for f in fg.factors.values():
        edges.append(VisEdge(var_ids=tuple(f.var_ids), factor_type=f.type))

    return nodes, edges


def _classify_edge_kind(a_type: NodeType, b_type: NodeType) -> str:
    """
    Classify an edge based on endpoint node types.

    Returns one of:
        - "room-place"
        - "place-object"
        - "pose-edge"
        - "other"
    """
    types = {a_type, b_type}

    if "room" in types and "place" in types:
        return "room-place"
    if "place" in types and ("voxel" in types or "other" in types):
        return "place-object"
    if "pose" in types:
        return "pose-edge"
    return "other"


def plot_factor_graph_2d(fg: FactorGraph, show_labels: bool = True) -> None:
    """
    Simple top-down 2D visualization of the factor graph.

    - nodes colored by type
    - edges drawn between connected variable nodes (projected to x–y)
    - dynamic aspect ratio and bounds based on node extents

    :param fg: The factor graph to visualize.
    :param show_labels: Whether to draw node labels.
    """
    nodes, edges = export_factor_graph_for_vis(fg)

    # color palette per node type
    type_to_color: Dict[NodeType, str] = {
        "pose": "C0",
        "voxel": "C1",
        "place": "C2",
        "room": "C3",
        "other": "C4",
    }

    # Build quick lookup for positions and types
    node_pos: Dict[NodeId, jnp.ndarray] = {n.id: n.position for n in nodes}
    node_type: Dict[NodeId, NodeType] = {n.id: n.type for n in nodes}

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    # Draw edges (as lines between all pairs in each factor)
    for e in edges:
        var_ids = list(e.var_ids)
        if len(var_ids) < 2:
            continue
        for i in range(len(var_ids) - 1):
            ida = var_ids[i]
            idb = var_ids[i + 1]
            a = node_pos.get(ida)
            b = node_pos.get(idb)
            if a is None or b is None:
                continue

            kind = _classify_edge_kind(node_type.get(ida, "other"),
                                       node_type.get(idb, "other"))

            if kind == "room-place":
                color, ls, lw, alpha = "magenta", "-", 1.5, 0.6
            elif kind == "place-object":
                color, ls, lw, alpha = "magenta", ":", 1.2, 0.6
            elif kind == "pose-edge":
                color, ls, lw, alpha = "gray", "--", 0.8, 0.4
            else:
                color, ls, lw, alpha = "k", ":", 0.5, 0.2

            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[1]), float(b[1])],
                linewidth=lw,
                alpha=alpha,
                linestyle=ls,
                color=color,
            )

    # Draw nodes
    xs, ys = [], []
    for n in nodes:
        c = type_to_color.get(n.type, "k")
        x, y = float(n.position[0]), float(n.position[1])
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=25, c=c)
        if show_labels:
            ax.text(x + 0.05, y + 0.05, n.label, fontsize=6)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("DSG-JIT Factor Graph (2D / top-down)")

    # Dynamic bounds with equal aspect
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        max_range = max(max_x - min_x, max_y - min_y) / 2.0
        if max_range < 1e-3:
            max_range = 1.0
        mid_x = 0.5 * (max_x + min_x)
        mid_y = 0.5 * (max_y + min_y)
        ax.set_xlim(mid_x - max_range * 1.1, mid_x + max_range * 1.1)
        ax.set_ylim(mid_y - max_range * 1.1, mid_y + max_range * 1.1)

    fig.tight_layout()
    plt.show()


def plot_factor_graph_3d(fg: FactorGraph, show_labels: bool = True) -> None:
    """
    3D visualization of the factor graph.

    - Nodes plotted as (x, y, z)
    - Edges drawn as 3D line segments
    - Colors by node type

    :param fg: The factor graph to visualize.
    :param show_labels: Whether to draw node labels in 3D.
    """
    nodes, edges = export_factor_graph_for_vis(fg)

    type_to_color: Dict[NodeType, str] = {
        "pose": "C0",
        "voxel": "C1",
        "place": "C2",
        "room": "C3",
        "other": "C4",
    }

    node_pos: Dict[NodeId, jnp.ndarray] = {n.id: n.position for n in nodes}
    node_type: Dict[NodeId, NodeType] = {n.id: n.type for n in nodes}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw edges
    for e in edges:
        var_ids = list(e.var_ids)
        if len(var_ids) < 2:
            continue
        for i in range(len(var_ids) - 1):
            ida = var_ids[i]
            idb = var_ids[i + 1]
            a = node_pos.get(ida)
            b = node_pos.get(idb)
            if a is None or b is None:
                continue

            kind = _classify_edge_kind(node_type.get(ida, "other"),
                                       node_type.get(idb, "other"))

            if kind == "room-place":
                color, ls, lw, alpha = "magenta", "-", 1.5, 0.6
            elif kind == "place-object":
                color, ls, lw, alpha = "magenta", ":", 1.2, 0.6
            elif kind == "pose-edge":
                color, ls, lw, alpha = "gray", "--", 0.8, 0.4
            else:
                color, ls, lw, alpha = "k", ":", 0.5, 0.2

            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[1]), float(b[1])],
                [float(a[2]), float(b[2])],
                linewidth=lw,
                alpha=alpha,
                linestyle=ls,
                color=color,
            )

    # Draw nodes
    xs, ys, zs = [], [], []
    for n in nodes:
        c = type_to_color.get(n.type, "k")
        x, y, z = map(float, n.position[:3])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(x, y, z, s=30, c=c)
        if show_labels:
            ax.text(x, y, z, n.label, fontsize=6)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("DSG-JIT Factor Graph (3D)")

    # Make aspect ratio equal in 3D
    if xs and ys and zs:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
        if max_range < 1e-3:
            max_range = 1.0
        mid_x = 0.5 * (max_x + min_x)
        mid_y = 0.5 * (max_y + min_y)
        mid_z = 0.5 * (max_z + min_z)
        ax.set_xlim(mid_x - max_range * 1.1, mid_x + max_range * 1.1)
        ax.set_ylim(mid_y - max_range * 1.1, mid_y + max_range * 1.1)
        ax.set_zlim(mid_z - max_range * 1.1, mid_z + max_range * 1.1)

    plt.show()