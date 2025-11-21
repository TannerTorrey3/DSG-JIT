"""
exp18_scenegraph_demo.py

"Hero" DSG-JIT scene graph visualization:

- Two rooms (A, B)
- One shared place (hallway)
- A small robot trajectory (5 poses) near the place
- Three objects anchored near the place

We construct this as a bare FactorGraph with variables + structural
factors used *only* for visualization. No optimization is run here:
all positions are specified directly to keep the demo robust and
purely focused on rendering.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from world.visualization import plot_factor_graph_3d, plot_factor_graph_2d


def build_scenegraph_factor_graph() -> FactorGraph:
    fg = FactorGraph()

    # --- Helpers ------------------------------------------------------------
    def add_var(idx: int, vtype: str, value) -> NodeId:
        nid = NodeId(idx)
        fg.add_variable(
            Variable(
                id=nid,
                type=vtype,
                value=jnp.asarray(value, dtype=jnp.float32),
            )
        )
        return nid

    def add_edge(fid: int, var_indices, ftype: str = "scene_edge") -> None:
        fg.add_factor(
            Factor(
                id=FactorId(fid),
                type=ftype,
                var_ids=tuple(NodeId(i) for i in var_indices),
                params={},  # purely structural, no residuals used
            )
        )

    # --- Rooms (high-level) -------------------------------------------------
    # Positions are (x, y, z) in "world" coordinates
    room_a = add_var(100, "room1d", [2.0, 2.0, 1.2])   # Room A up/right
    room_b = add_var(101, "room1d", [4.5, 2.0, 1.0])   # Room B further right

    # --- Shared place (hallway / doorway) -----------------------------------
    place_corridor = add_var(200, "place1d", [1.0, 0.0, 0.0])

    # Connect rooms to shared place
    eid = 0
    add_edge(eid, [room_a, place_corridor], ftype="room_place")
    eid += 1
    add_edge(eid, [room_b, place_corridor], ftype="room_place")
    eid += 1

    # --- Robot trajectory (poses near the place) ----------------------------
    pose_ids = []
    for i, x in enumerate([-1.0, -0.2, 0.6, 1.4, 2.2]):
        # pose_se3: [tx, ty, tz, roll, pitch, yaw]
        p = add_var(10 + i, "pose_se3", [x, 0.0, 0.5, 0.0, 0.0, 0.0])
        pose_ids.append(p)

    # Link poses in a chain (visual odometry edges)
    for i in range(len(pose_ids) - 1):
        add_edge(eid, [pose_ids[i], pose_ids[i + 1]], ftype="pose_chain")
        eid += 1

    # Attach all poses to the corridor place (like a localization prior)
    for pid in pose_ids:
        add_edge(eid, [pid, place_corridor], ftype="pose_place_attachment")
        eid += 1

    # --- Objects near the place ---------------------------------------------
    obj_chair = add_var(300, "voxel_cell", [0.7, 0.4, 0.6])
    obj_table = add_var(301, "voxel_cell", [1.3, 0.5, 0.7])
    obj_plant = add_var(302, "voxel_cell", [0.9, 0.9, 0.9])

    # Attach objects to the place (semantic containment)
    add_edge(eid, [place_corridor, obj_chair], ftype="place_object")
    eid += 1
    add_edge(eid, [place_corridor, obj_table], ftype="place_object")
    eid += 1
    add_edge(eid, [place_corridor, obj_plant], ftype="place_object")
    eid += 1

    return fg


def main() -> None:
    fg = build_scenegraph_factor_graph()

    # Just visualize – no optimization in this hero scene graph demo.
    print("=== DSG-JIT Scene Graph Demo (exp18) ===")
    print(f"Num variables: {len(fg.variables)}")
    print(f"Num factors:   {len(fg.factors)}")

    # 2D top-down
    plot_factor_graph_2d(fg, show_labels=True)

    # 3D view
    plot_factor_graph_3d(fg, show_labels=True)


if __name__ == "__main__":
    main()