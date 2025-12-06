from __future__ import annotations

import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.world.visualization import plot_factor_graph_3d, plot_factor_graph_2d


def build_scenegraph_world_model() -> WorldModel:
    wm = WorldModel()

    # --- Helpers ------------------------------------------------------------
    def add_var(vtype: str, value):
        """
        Add a variable of the given type with the given value.
        Returns the variable id assigned by the WorldModel.
        """
        return wm.add_variable(
            var_type=vtype,
            value=jnp.asarray(value, dtype=jnp.float32),
        )

    def add_edge(var_ids, ftype: str = "scene_edge") -> None:
        """
        Add a purely structural factor (no residual) connecting var_ids.
        """
        wm.add_factor(
            f_type=ftype,
            var_ids=tuple(var_ids),
            params={},  # purely structural, no residuals used
        )

    # --- Rooms (high-level) -------------------------------------------------
    # Positions are (x, y, z) in "world" coordinates
    room_a = add_var("room1d", [2.0, 2.0, 1.2])   # Room A up/right
    room_b = add_var("room1d", [4.5, 2.0, 1.0])   # Room B further right

    # --- Shared place (hallway / doorway) -----------------------------------
    place_corridor = add_var("place1d", [1.0, 0.0, 0.0])

    # Connect rooms to shared place
    add_edge([room_a, place_corridor], ftype="room_place")
    add_edge([room_b, place_corridor], ftype="room_place")

    # --- Robot trajectory (poses near the place) ----------------------------
    pose_ids = []
    for x in [-1.0, -0.2, 0.6, 1.4, 2.2]:
        # pose_se3: [tx, ty, tz, roll, pitch, yaw]
        p = add_var("pose_se3", [x, 0.0, 0.5, 0.0, 0.0, 0.0])
        pose_ids.append(p)

    # Link poses in a chain (visual odometry edges)
    for i in range(len(pose_ids) - 1):
        add_edge([pose_ids[i], pose_ids[i + 1]], ftype="pose_chain")

    # Attach all poses to the corridor place (like a localization prior)
    for pid in pose_ids:
        add_edge([pid, place_corridor], ftype="pose_place_attachment")

    # --- Objects near the place ---------------------------------------------
    obj_chair = add_var("voxel_cell", [0.7, 0.4, 0.6])
    obj_table = add_var("voxel_cell", [1.3, 0.5, 0.7])
    obj_plant = add_var("voxel_cell", [0.9, 0.9, 0.9])

    # Attach objects to the place (semantic containment)
    add_edge([place_corridor, obj_chair], ftype="place_object")
    add_edge([place_corridor, obj_table], ftype="place_object")
    add_edge([place_corridor, obj_plant], ftype="place_object")

    return wm


def main() -> None:
    wm = build_scenegraph_world_model()
    fg = wm.fg  # underlying FactorGraph for visualization

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