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
- `experiments/exp19_dynamic_scene_graph_demo.py` (dynamic agent trajectories)

Module contents:
    - `VisNode`: Lightweight typed node container for visualization.
    - `VisEdge`: Lightweight edge container (factor connections).
    - `_infer_node_type()`: Maps variable types → canonical visualization types.
    - `_extract_position()`: Extracts a 3D coordinate from variable states.
    - `export_factor_graph_for_vis()`: Converts a FactorGraph → vis nodes & edges.
    - `plot_factor_graph_2d()`: Renders a 2D top-down view of the graph.
    - `plot_factor_graph_3d()`: Renders a full 3D scene graph with semantic layers.
    - `plot_scenegraph_3d()`: Renders a scene graph with semantic layers and (optionally) agent trajectories.
    - `plot_dynamic_trajectories_3d()`: Renders 3D agent trajectories with time-encoded color.

This module is designed to be extendable—for example:
- Additional node types can be added via `_infer_node_type`.
- SceneGraphWorld can later provide richer semantic annotations.
- Future versions may support interactive or WebGL visualizations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Union, Optional

import numpy as np

import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dsg_jit.core.types import NodeId
from dsg_jit.core.factor_graph import FactorGraph

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
    """
    Map internal ``Variable.type`` strings to coarse visualization node types.

    :param var_type: The low-level variable type string (e.g. ``"pose_se3"``,
        ``"voxel_cell"``, ``"place1d"``, ``"room1d"``).
    :return: A normalized node type label used by the visualization layer
        (``"pose"``, ``"voxel"``, ``"place"``, ``"room"``, or ``"other"``).
    """
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

    - ``pose_se3``: take translation (first 3 elements)
    - ``voxel_cell``: assume first 3 entries are the position
    - ``place1d`` / ``room1d``:
      * if ``len(value) >= 3``: treat first 3 entries as a 3D position
      * else: embed the scalar along the x-axis, with a y-offset for rooms
    - fallback: origin ``[0, 0, 0]``

    :param var_type: Low-level variable type string indicating how ``value``
        should be interpreted.
    :param value: State vector for the variable; may be 1D (e.g. a scalar)
        or higher dimensional.
    :return: A length-3 JAX array representing the 3D position used for
        plotting.
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
    Classify an edge based on the node types at its endpoints.

    The returned label is used to select line style and color in the
    visualization.

    :param a_type: Node type label for the first endpoint (e.g. ``"pose"``,
        ``"voxel"``, ``"place"``, ``"room"``, or ``"other"``).
    :param b_type: Node type label for the second endpoint.
    :return: A string identifying the edge category, one of
        ``"room-place"``, ``"place-object"``, ``"pose-edge"``, or ``"other"``.
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

def plot_scenegraph_3d(
    sg: Any,
    x_opt: Any = None,
    index: Optional[Dict[Any, Union[slice, tuple]]] = None,
    title: str = "Scene Graph 3D",
    dsg: Optional[Any] = None,
) -> None:
    """
    Render a 3D scene graph with rooms, places, objects, place attachments,
    and optional agent trajectories.

    This function supports two modes:
    - If ``sg`` exposes a ``_memory`` attribute (the SceneGraph memory layer introduced in ``SceneGraphWorld``),
      node positions are read from this memory and ``x_opt`` and ``index`` are ignored.
    - If no memory is present, the function falls back to the previous behavior using ``x_opt`` and ``index``
      to decode node states.

    :param sg: Scene-graph world instance. It is expected to expose
        attributes such as ``rooms``, ``places``, ``objects``,
        ``place_parents``, ``object_parents``, and ``place_attachments``,
        following the conventions used by :class:`SceneGraphWorld`.
    :param x_opt: (Optional) Optimized flat state vector (e.g. from
        :meth:`WorldModel.pack_state`), containing the current estimates
        of all node states. Not required if ``sg`` exposes a ``_memory`` layer.
    :param index: (Optional) Mapping from node identifier to either a slice or
        ``(start, dim)`` tuple describing where that node’s state lives
        inside ``x_opt``. Not required if ``sg`` exposes a ``_memory`` layer.
    :param title: Optional figure title for the Matplotlib 3D axes.
    :param dsg: Optional dynamic scene graph used to overlay agent
        trajectories. It should expose an iterable ``agents`` attribute
        and a ``get_agent_trajectory(agent, x_opt, index)`` method that
        returns an array of shape ``(T, 6)`` or ``(T, 3)``.
    :return: None. The function creates and displays a Matplotlib 3D figure.
    """
    has_memory = hasattr(sg, "_memory")
    mem = getattr(sg, "_memory", None)

    def _partition_memory_by_type() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Derive rooms / places / objects mappings from the SceneGraphWorld
        memory layer when explicit dictionaries (sg.rooms, sg.places,
        sg.objects) are not available or are empty.

        This assumes each memory entry is a small dataclass-like object
        exposing ``node_id`` and ``var_type`` attributes, where
        ``var_type`` starts with e.g. ``"room"``, ``"place"``, or
        ``"object"`` / ``"voxel"``.
        """
        rooms_m: Dict[str, Any] = {}
        places_m: Dict[str, Any] = {}
        objects_m: Dict[str, Any] = {}
        if not has_memory or mem is None:
            return rooms_m, places_m, objects_m

        # Iterate over stored node states and group by var_type prefix.
        for state in getattr(mem, "values", lambda: [])():
            vt = getattr(state, "var_type", "")
            nid = getattr(state, "node_id", None)
            if nid is None:
                continue
            # Construct simple human-readable names when no explicit names exist.
            if vt.startswith("room"):
                key = f"room_{nid}"
                rooms_m[key] = nid
            elif vt.startswith("place"):
                key = f"place_{nid}"
                places_m[key] = nid
            elif vt.startswith("object") or vt.startswith("voxel"):
                key = f"obj_{nid}"
                objects_m[key] = nid
        return rooms_m, places_m, objects_m

    def _has_state(nid: Any) -> bool:
        """
        Check whether we have a stored state for the given node id.

        When using SceneGraphWorld memory, we support both integer keys
        and arbitrary NodeId-like keys by trying ``nid`` directly first
        and then falling back to ``int(nid)`` if conversion is possible.
        """
        if has_memory:
            # Try raw key as-is
            try:
                if nid in mem:
                    return True
            except TypeError:
                # Some key types may not support `in` with this nid
                pass
            # Fallback: try integer-cast key
            try:
                nid_int = int(nid)
            except (TypeError, ValueError):
                return False
            return nid_int in mem
        if index is None:
            return False
        return nid in index

    def _vec(nid: Any) -> np.ndarray:
        if has_memory:
            # Support both direct nid keys and integer-cast keys.
            state = None
            # Try raw nid first
            try:
                state = mem.get(nid)  # type: ignore[call-arg]
            except AttributeError:
                # If _memory is not a Mapping, fall back to direct indexing
                try:
                    state = mem[nid]  # type: ignore[index]
                except Exception:
                    state = None
            if state is None:
                # Fallback: try integer-cast key
                try:
                    nid_int = int(nid)
                except (TypeError, ValueError):
                    raise KeyError(f"No state in SceneGraph memory for node id={nid!r}")
                try:
                    state = mem.get(nid_int)  # type: ignore[call-arg]
                except AttributeError:
                    state = mem[nid_int]  # type: ignore[index]
            if state is None:
                raise KeyError(f"No state in SceneGraph memory for node id={nid!r}")
            v = np.asarray(state.value).reshape(-1)
            return v
        if index is None or x_opt is None:
            raise ValueError("x_opt and index must be provided when SceneGraph memory is not available")
        idx = index[nid]
        if isinstance(idx, slice):
            sl = idx
        else:
            start, length = idx
            sl = slice(start, start + length)
        v = np.asarray(x_opt[sl]).reshape(-1)
        return v

    # Safely grab scene-graph structures (with defaults if missing).
    rooms = getattr(sg, "rooms", {}) or {}
    places = getattr(sg, "places", {}) or {}
    objects = getattr(sg, "objects", {}) or {}

    # If we have a memory layer but no explicit named dicts, derive them from memory.
    if has_memory and mem is not None:
        if not rooms or not isinstance(rooms, dict):
            mem_rooms, mem_places, mem_objects = _partition_memory_by_type()
            # Only fill in from memory when each layer is empty; this way,
            # user-provided names (if any) take precedence.
            if not rooms:
                rooms = mem_rooms
            if not places:
                places = mem_places
            if not objects:
                objects = mem_objects

    place_parents = getattr(sg, "place_parents", {}) or {}
    object_parents = getattr(sg, "object_parents", {}) or {}
    attachments = getattr(sg, "place_attachments", []) or []

    # -------------------------------------------------
    # Collect pose node ids (for rendering trajectories / agent poses).
    # We look in both the memory layer and the place-attachment edges.
    pose_ids: set[Any] = set()

    # From attachments: first element of each tuple is assumed to be a pose node id.
    for pose_nid, _ in attachments:
        pose_ids.add(pose_nid)

    # From memory: any node whose var_type starts with "pose" is treated as a pose.
    if has_memory and mem is not None:
        for state in getattr(mem, "values", lambda: [])():
            vt = getattr(state, "var_type", "")
            if vt.startswith("pose"):
                nid = getattr(state, "node_id", None)
                if nid is not None:
                    pose_ids.add(nid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    all_pts = []

    # ------------------------------
    # Rooms: large semi-transparent markers
    # ------------------------------
    first_room = next(iter(rooms), None)
    for name, nid in rooms.items():
        if not _has_state(nid):
            continue
        p = _vec(nid)
        if p.shape[0] < 3:
            # If we only have 1D, pad to 3D for visualization
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        all_pts.append(p[:3])
        label = "room" if name == first_room else ""
        ax.scatter(
            p[0],
            p[1],
            p[2],
            s=200,
            marker="s",
            alpha=0.3,
            edgecolor="k",
            label=label,
        )

    # ------------------------------
    # Places: medium spheres
    # ------------------------------
    first_place = next(iter(places), None)
    for name, nid in places.items():
        if not _has_state(nid):
            continue
        p = _vec(nid)
        if p.shape[0] < 3:
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        all_pts.append(p[:3])
        label = "place" if name == first_place else ""
        ax.scatter(
            p[0],
            p[1],
            p[2],
            s=60,
            marker="o",
            alpha=0.8,
            label=label,
        )

    # ------------------------------
    # Objects: small pyramids/triangles
    # ------------------------------
    first_obj = next(iter(objects), None)
    for name, nid in objects.items():
        if not _has_state(nid):
            continue
        p = _vec(nid)
        if p.shape[0] < 3:
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        all_pts.append(p[:3])
        label = "object" if name == first_obj else ""
        ax.scatter(
            p[0],
            p[1],
            p[2],
            s=40,
            marker="^",
            alpha=0.9,
            label=label,
        )

    # ------------------------------
    # Poses: agent pose nodes (small spheres)
    # ------------------------------
    first_pose = next(iter(pose_ids), None)
    for nid in pose_ids:
        if not _has_state(nid):
            continue
        p = _vec(nid)
        if p.shape[0] < 3:
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        all_pts.append(p[:3])
        label = "pose" if nid == first_pose else ""
        ax.scatter(
            p[0],
            p[1],
            p[2],
            s=30,
            marker="o",
            alpha=1.0,
            label=label,
        )

    # ------------------------------
    # Hierarchical edges: room -> place, place -> object
    # ------------------------------
    for place_nid, room_nid in place_parents.items():
        if not (_has_state(place_nid) and _has_state(room_nid)):
            continue
        p = _vec(place_nid)
        r = _vec(room_nid)
        if p.shape[0] < 3:
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        if r.shape[0] < 3:
            r = np.pad(r, (0, 3 - r.shape[0]), mode="constant")
        ax.plot(
            [p[0], r[0]],
            [p[1], r[1]],
            [p[2], r[2]],
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    for obj_nid, place_nid in object_parents.items():
        if not (_has_state(obj_nid) and _has_state(place_nid)):
            continue
        o = _vec(obj_nid)
        p = _vec(place_nid)
        if o.shape[0] < 3:
            o = np.pad(o, (0, 3 - o.shape[0]), mode="constant")
        if p.shape[0] < 3:
            p = np.pad(p, (0, 3 - p.shape[0]), mode="constant")
        ax.plot(
            [o[0], p[0]],
            [o[1], p[1]],
            [o[2], p[2]],
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
        )

    # ------------------------------
    # Place attachments: pose -> place (dashed)
    # ------------------------------
    for pose_nid, place_nid in attachments:
        if not (_has_state(pose_nid) and _has_state(place_nid)):
            continue
        pose = _vec(pose_nid)
        plc = _vec(place_nid)
        if pose.shape[0] < 3:
            pose = np.pad(pose, (0, 3 - pose.shape[0]), mode="constant")
        if plc.shape[0] < 3:
            plc = np.pad(plc, (0, 3 - plc.shape[0]), mode="constant")
        ax.plot(
            [pose[0], plc[0]],
            [pose[1], plc[1]],
            [pose[2], plc[2]],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    # ------------------------------
    # Optional: agent trajectories from DynamicSceneGraph
    # ------------------------------
    if dsg is not None and hasattr(dsg, "agents"):
        for agent in dsg.agents:
            traj = dsg.get_agent_trajectory(agent, x_opt, index)
            traj = np.asarray(traj)
            if traj.ndim != 2 or traj.shape[1] < 3:
                continue
            xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
            all_pts.extend(traj[:, :3])
            ax.plot(xs, ys, zs, linewidth=2.0, alpha=0.9, label=f"{agent}_traj")

    # ------------------------------
    # Autoscale axes to fit everything
    # ------------------------------
    if all_pts:
        all_pts_arr = np.vstack(all_pts)
        mins = all_pts_arr.min(axis=0)
        maxs = all_pts_arr.max(axis=0)
        center = 0.5 * (mins + maxs)
        extent = float((maxs - mins).max())
        if extent <= 0.0:
            extent = 1.0
        scale = 0.6 * extent
        ax.set_xlim(center[0] - scale, center[0] + scale)
        ax.set_ylim(center[1] - scale, center[1] + scale)
        ax.set_zlim(center[2] - scale, center[2] + scale)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc="best")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# Dynamic Scene Graph: 4D-style agent trajectory visualization
# ------------------------------------------------------
def plot_dynamic_trajectories_3d(
    dsg: Any,
    x_opt: Any,
    index: Dict[Any, Union[slice, tuple]],
    title: str = "Dynamic 3D Scene Graph",
    color_by_time: bool = True,
) -> None:
    """
    Render 3D agent trajectories with time encoded as color.

    This helper is intended for ``DynamicSceneGraph``-style structures where
    agents move through time. It treats time as an implicit fourth
    dimension and visualizes it via either a color gradient or a solid
    color per agent.

    :param dsg: Dynamic scene graph object exposing an iterable
        ``agents`` attribute and a
        ``get_agent_trajectory(agent, x_opt, index)`` method that returns
        an array of shape ``(T, 6)`` or ``(T, 3)``. Only the translational
        components ``(x, y, z)`` are visualized.
    :param x_opt: Optimized flat state vector used to decode agent poses.
    :param index: Mapping from node identifier to slice or ``(start, dim)``
        describing how to extract each node’s state from ``x_opt``. This is
        passed through to ``dsg.get_agent_trajectory``.
    :param title: Optional figure title for the 3D plot.
    :param color_by_time: If ``True``, encode time as a colormap gradient
        along each trajectory; if ``False``, use a single solid color per
        agent.
    :return: None. The function creates and displays a Matplotlib 3D figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    all_pts: List[np.ndarray] = []

    if not hasattr(dsg, "agents"):
        raise ValueError("Dynamic scene graph object must expose an 'agents' attribute")

    # Use a base list of colors when not color-coding by time.
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])  # type: ignore[index]

    for i, agent in enumerate(dsg.agents):
        traj = dsg.get_agent_trajectory(agent, x_opt, index)
        traj = np.asarray(traj)
        if traj.ndim != 2 or traj.shape[1] < 3:
            continue

        xyz = traj[:, :3]
        all_pts.append(xyz)

        if color_by_time and xyz.shape[0] > 1:
            # Encode time as a gradient along the trajectory
            t = np.linspace(0.0, 1.0, xyz.shape[0])
            cmap = plt.get_cmap("viridis")
            for j in range(xyz.shape[0] - 1):
                c = cmap(t[j])
                ax.plot(
                    xyz[j : j + 2, 0],
                    xyz[j : j + 2, 1],
                    xyz[j : j + 2, 2],
                    color=c,
                    linewidth=2.0,
                )
        else:
            color = base_colors[i % len(base_colors)]
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                linewidth=2.0,
                label=f"{agent}_traj",
                color=color,
            )

    # Autoscale axes to include all trajectories
    if all_pts:
        stacked = np.vstack(all_pts)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        center = 0.5 * (mins + maxs)
        extent = float((maxs - mins).max())
        if extent <= 0.0:
            extent = 1.0
        scale = 0.6 * extent
        ax.set_xlim(center[0] - scale, center[0] + scale)
        ax.set_ylim(center[1] - scale, center[1] + scale)
        ax.set_zlim(center[2] - scale, center[2] + scale)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Only show legend entries when not using per-segment colors.
    if not color_by_time:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            uniq = {}
            for h, l in zip(handles, labels):
                if l and l not in uniq:
                    uniq[l] = h
            ax.legend(uniq.values(), uniq.keys(), loc="best")

    plt.tight_layout()
    plt.show()