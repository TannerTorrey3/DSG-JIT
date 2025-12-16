"""exp23_sliding_window_sg

Experiment 23: Sliding-window optimization with persistent SceneGraph memory.

This experiment stress-tests the interaction between:
  (1) a bounded *active* FactorGraph (optimized online), and
  (2) a persistent SceneGraph memory layer (which retains the full history).

We construct:
  * rooms (place layer), objects (object layer), and a pose trajectory (agent layer)
  * semantic edges (room↔place, room↔object) stored in SceneGraph memory
  * metric factors (prior/odom, and pose↔place attachments via SceneGraphWorld)

The online optimizer uses an ActiveWindowTemplate of size W (default 20).
At each time step:
  * a new pose is added to the SceneGraph
  * poses are assigned to the closest room via a transient place node
  * the active template is populated with the last W poses
  * prior + odom factor slots are configured for those active poses
  * a small number of GN iterations are run on the active window

Finally, we launch the Three.js web visualizer to render the full persistent
SceneGraph memory.

NOTE:
  SceneGraphWorld may introduce pose↔place attachment factors whose default
  residual implementation reads `pose_dim = int(params['pose_dim'])`. Under
  `jax.vmap`, that scalar becomes a tracer and JAX raises a ConcretizationTypeError.
  To keep this experiment working without modifying engine code, we override the
  residual for factor type `pose_place_attachment` via the public
  `register_residual` API, using overwrite semantics when supported.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.world.model import ActiveWindowTemplate, WorldModel
from dsg_jit.world.scene_graph import SceneGraphWorld


def _register_residuals_for_active_template(wm: WorldModel) -> None:
    """Register residual functions required by this experiment.

    Active-template GN requires residuals to be registered at the WorldModel.

    Note:
        SceneGraphWorld may add pose↔place attachment factors whose reference
        residual implementation reads `pose_dim = int(params['pose_dim'])`.
        Under `jax.vmap`, that scalar becomes a tracer and JAX raises a
        ConcretizationTypeError. We override the attachment residual via the
        public `register_residual` API, using overwrite semantics when supported.
    """

    from dsg_jit.slam.measurements import odom_se3_residual, prior_residual

    def _register(name: str, fn) -> None:
        """Register a residual with overwrite semantics if supported by WorldModel.

        Some WorldModel versions may expose an overwrite/replace flag; older ones
        simply overwrite by default. We try the flagged form first and fall back.
        """
        try:
            wm.register_residual(factor_type=name, fn=fn)  # type: ignore[call-arg]
        except TypeError:
            # Older signature: (name, fn)
            wm.register_residual(name, fn)

    _register("prior", prior_residual)
    _register("odom_se3", odom_se3_residual)

    # Tracer-safe residual for pose↔place attachments.
    # Assumes stacked ordering: [pose_se3(6), place3d(3)] -> total 9.
    def pose_place_attachment_residual_safe(stacked: jnp.ndarray, params: dict) -> jnp.ndarray:
        pose_xyz = stacked[0:3]
        place_xyz = stacked[6:9]
        r = pose_xyz - place_xyz
        w = params.get("weight", 1.0)
        return jnp.sqrt(w) * r

    _register("pose_place_attachment", pose_place_attachment_residual_safe)


def _optimize_active_window(wm: WorldModel, iters: int) -> None:
    """Run optimization using the active-template pathway when available."""

    # Prefer explicit active-template optimize methods if implemented.
    if hasattr(wm, "optimize_active_template"):
        wm.optimize_active_template(iters=iters)
        return
    if hasattr(wm, "optimize_active"):
        wm.optimize_active(iters=iters)
        return

    # Fallback to the generic optimize.
    wm.optimize(method="gn", iters=iters)


def build_sliding_window_scene(
    num_poses: int = 100,
    num_rooms: int = 5,
    objects_per_room: int = 20,
    window_size: int = 20,
    gn_iters_per_step: int = 1,
) -> SceneGraphWorld:
    """Build a complex SceneGraph and run online batch optimization with Active Templates."""

    assert window_size > 0

    sg = SceneGraphWorld()
    wm: WorldModel = sg.wm

    # Ensure residuals exist for factors we will create.
    _register_residuals_for_active_template(wm)

    # ---------------------------------------------------------------------
    # 1) Create rooms in a simple linear layout.
    # ---------------------------------------------------------------------
    rooms: List[int] = []
    room_centers: List[np.ndarray] = []
    for i in range(num_rooms):
        center = np.array([5.0 * i, 0.0, 0.0], dtype=np.float32)
        room_id = sg.add_room(f"room_{i}", center)
        rooms.append(room_id)
        room_centers.append(center)

    # ---------------------------------------------------------------------
    # 2) Populate each room with objects (semantic only; persistent memory).
    # ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    for i, room_id in enumerate(rooms):
        center = room_centers[i]
        for j in range(objects_per_room):
            offset = rng.normal(scale=0.75, size=3).astype(np.float32)
            obj_pos = center + offset
            obj_id = sg.add_named_object3d(f"room{i}_obj{j}", obj_pos)
            # Ensure hierarchy exists in memory:
            if hasattr(sg, "add_object_room_edge"):
                sg.add_object_room_edge(obj_id, room_id)

    # ---------------------------------------------------------------------
    # 3) Initialize the active template once.
    # ---------------------------------------------------------------------
    POSE_DIM = 6
    W = window_size

    # Variable slots: window pose slots pose_se3[0..W-1]
    variable_slots = [("pose_se3", i, POSE_DIM) for i in range(W)]

    # Factor slots:
    #  - slot 0: prior on pose_se3[0]
    #  - slots 1..W-1: odom factors between consecutive pose slots
    factor_slots: List[Tuple[str, int, Tuple[Tuple[str, int], ...]]] = []
    factor_slots.append(("prior", 0, (("pose_se3", 0),)))
    for k in range(1, W):
        factor_slots.append(("odom_se3", k, (("pose_se3", k - 1), ("pose_se3", k))))

    template = ActiveWindowTemplate(variable_slots=variable_slots, factor_slots=factor_slots)
    wm.init_active_template(template)

    # Persistent list of all pose node-ids (SceneGraph memory ids).
    pose_ids: List[int] = []

    # Cache last known optimized values for poses currently in the window.
    pose_state_cache: Dict[int, jnp.ndarray] = {}

    step_dx = 0.5

    # ---------------------------------------------------------------------
    # 4) Online loop: add data, update window slots, optimize a batch.
    # ---------------------------------------------------------------------
    for t in range(num_poses):
        x = step_dx * t
        y = 0.5 * math.sin(0.1 * t)
        z = 0.0

        pose_vec = jnp.array([x, y, z, 0.0, 0.0, 0.0], dtype=jnp.float32)
        pose_id = sg.add_agent_pose_se3("agent0", t, pose_vec)
        pose_ids.append(pose_id)

        # Attach pose to closest room via a place node (for richer hierarchy).
        pose_pos = np.array([x, y, z], dtype=np.float32)
        dists = [np.linalg.norm(pose_pos - rc) for rc in room_centers]
        room_idx = int(np.argmin(dists))

        place_name = f"place_room{room_idx}_t{t}"
        place_id = sg.add_place3d(place_name, pose_pos)
        # These calls may also create factors in the WM via SceneGraphWorld.
        sg.add_place_attachment(pose_id, place_id)
        if hasattr(sg, "add_room_place_edge"):
            sg.add_room_place_edge(rooms[room_idx], place_id)

        # Odometry measurement relative to synthetic GT.
        if t > 0:
            prev_y = 0.5 * math.sin(0.1 * (t - 1))
            dy = y - prev_y
            odom_meas = jnp.array([step_dx, dy, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        else:
            odom_meas = None

        # -----------------------------------------------------------------
        # Active window: last W poses.
        # -----------------------------------------------------------------
        if len(pose_ids) < W:
            pose_state_cache[pose_id] = pose_vec
            continue

        active_pose_ids = pose_ids[-W:]

        # Populate variable slots.
        slot_nids: List[int] = []
        for i, pid in enumerate(active_pose_ids):
            init_val = pose_state_cache.get(pid)

            # Guard against corrupted/empty cached values (can happen if a slice is missing)
            # Slot expects shape (6,).
            if init_val is None or getattr(init_val, "shape", ()) != (6,):
                init_val = pose_vec if pid == pose_id else jnp.zeros((6,), dtype=jnp.float32)

            nid = int(wm.set_variable_slot("pose_se3", i, init_val))
            slot_nids.append(nid)

        # Prior slot on first pose in window.
        anchor_pid = active_pose_ids[0]
        anchor_target = pose_state_cache.get(anchor_pid, jnp.zeros((6,), dtype=jnp.float32))
        wm.configure_factor_slot(
            factor_type="prior",
            slot_idx=0,
            var_ids=(slot_nids[0],),
            params={"target": anchor_target, "weight": 1.0},
            active=True,
        )

        # Odom slots.
        for k in range(1, W):
            pid_i = active_pose_ids[k - 1]
            pid_j = active_pose_ids[k]

            xi = pose_state_cache.get(pid_i)
            xj = pose_state_cache.get(pid_j)
            if xi is not None and xj is not None:
                meas = xj - xi
            else:
                meas = jnp.array([step_dx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

            if k == W - 1 and odom_meas is not None:
                meas = odom_meas

            wm.configure_factor_slot(
                factor_type="odom_se3",
                slot_idx=k,
                var_ids=(slot_nids[k - 1], slot_nids[k]),
                params={"measurement": meas, "weight": 1.0},
                active=True,
            )

        # Optimize the active window.
        _optimize_active_window(wm, iters=gn_iters_per_step)

        # Update cache from optimized packed state.
        x_opt, index = wm.pack_state()
        for i, pid in enumerate(active_pose_ids):
            nid = slot_nids[i]
            if nid in index:
                st, en = index[nid]
                est = jnp.array(x_opt[st:en])
                if est.shape == (6,):
                    pose_state_cache[pid] = est

    return sg


def run_sliding_window_sg_experiment(
    num_poses: int = 100,
    num_rooms: int = 5,
    objects_per_room: int = 20,
    window_size: int = 20,
    gn_iters_per_step: int = 1,
    web_port: int = 8000,
) -> None:
    """Run experiment then visualize the persistent SceneGraph memory."""

    sg = build_sliding_window_scene(
        num_poses=num_poses,
        num_rooms=num_rooms,
        objects_per_room=objects_per_room,
        window_size=window_size,
        gn_iters_per_step=gn_iters_per_step,
    )

    wm: WorldModel = sg.wm
    print("=== Experiment 23 summary ===")
    print(f"Total SG memory entries: {len(getattr(sg, '_memory', {}))}")
    print(f"Active FG variables: {len(wm.fg.variables)}")
    print(f"Active FG factors: {len(wm.fg.factors)}")
    print("Launching web visualizer...")

    sg.visualize_web(port=web_port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 23: Sliding-window DSG with persistent SceneGraph memory."
    )
    parser.add_argument("--num-poses", type=int, default=100)
    parser.add_argument("--num-rooms", type=int, default=5)
    parser.add_argument("--objects-per-room", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--gn-iters-per-step", type=int, default=1)
    parser.add_argument("--web-port", type=int, default=8000)

    args = parser.parse_args()

    run_sliding_window_sg_experiment(
        num_poses=args.num_poses,
        num_rooms=args.num_rooms,
        objects_per_room=args.objects_per_room,
        window_size=args.window_size,
        gn_iters_per_step=args.gn_iters_per_step,
        web_port=args.web_port,
    )


if __name__ == "__main__":
    main()
