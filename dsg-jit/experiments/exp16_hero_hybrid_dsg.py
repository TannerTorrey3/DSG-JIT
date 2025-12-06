import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel

from dsg_jit.optimization.solvers import gradient_descent, GDConfig
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,                  # additive SE3 odom residual
    voxel_point_observation_residual,   # voxel_point_obs residual
)


def build_hybrid_graph():
    """
    Hero hybrid experiment:
      - 6 SE(3) poses along x (ground truth ~ 0..5)
      - 6 voxel cells roughly at x = 0..5
      - odom_se3 factors between consecutive poses
      - voxel_point_obs factors from poses to voxels
      - priors on pose0 and voxel0

    Returns:
        wm: WorldModel
        pose_ids: list of pose variable ids, length 6
        voxel_ids: list of voxel variable ids, length 6
    """
    wm = WorldModel()

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # ---- Variables ----
    n_poses = 6
    n_voxels = 6

    pose_ids = []
    voxel_ids = []

    # Poses: small noise around [i,0,0,0,0,0]
    for i in range(n_poses):
        tx_gt = float(i)
        init = jnp.array(
            [
                tx_gt + 0.1 * (0.5 - i / (n_poses - 1.0)),  # small bias in x
                0.05 * ((-1.0) ** i),                      # tiny zig-zag in y
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=jnp.float32,
        )
        vid = wm.add_variable(
            var_type="pose_se3",
            value=init,
        )
        pose_ids.append(vid)

    # Voxels: initial guesses near ground-truth centers x=0..5
    for i in range(n_voxels):
        x_gt = float(i)
        init = jnp.array(
            [
                x_gt + 0.2 * (0.5 - i / (n_voxels - 1.0)),  # mild bias
                0.1 * ((-1.0) ** i),                        # off in y a bit
                0.0,
            ],
            dtype=jnp.float32,
        )
        vid = wm.add_variable(
            var_type="voxel_cell3d",  # treated as Euclidean by manifold code
            value=init,
        )
        voxel_ids.append(vid)

    # ---- Factors ----

    # Prior on pose0 at identity (tx=0, all else 0)
    pose0_target = jnp.zeros((6,), dtype=jnp.float32)
    wm.add_factor(
        f_type="prior",
        var_ids=(pose_ids[0],),
        params={"target": pose0_target},
    )

    # Weak prior on voxel0 near [0,0,0]
    voxel0_target = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    wm.add_factor(
        f_type="prior",
        var_ids=(voxel_ids[0],),
        params={"target": voxel0_target},
    )

    # Odom factors between consecutive poses: we will override "measurement"
    # in the experiment, but we still need a placeholder here.
    # measurement in se(3)-vector form: [dx, dy, dz, wx, wy, wz]
    for i in range(n_poses - 1):
        meas_placeholder = jnp.zeros((6,), dtype=jnp.float32)
        wm.add_factor(
            f_type="odom_se3",
            var_ids=(pose_ids[i], pose_ids[i + 1]),
            params={"measurement": meas_placeholder},
        )

    # Voxel observation factors: one obs per voxel, each attached to
    # the nearest pose along the chain for simplicity.
    #
    # factor type: "voxel_point_obs"
    # var_ids: (pose_j, voxel_i)
    # params: {"point_world": placeholder}
    for i, vid in enumerate(voxel_ids):
        # Attach each voxel to the nearest pose in x-index
        pose_index = min(i, n_poses - 1)
        pid = pose_ids[pose_index]

        point_placeholder = jnp.zeros((3,), dtype=jnp.float32)
        wm.add_factor(
            f_type="voxel_point_obs",
            var_ids=(pid, vid),
            params={"point_world": point_placeholder},
        )

    return wm, pose_ids, voxel_ids


def build_param_residual(wm: WorldModel):
    """
    Build a residual function that treats both:
      - ALL odom_se3 measurements (per-factor se(3))
      - ALL voxel_point_obs point_world (per-factor 3D point)
    as explicit parameters in a PyTree:

        theta = {
            "odom": jnp.ndarray[K_odom, 6],
            "obs":  jnp.ndarray[K_obs, 3],
        }

    Returns:
        residual_fn(x, theta), index, n_odom, n_obs
    """
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    _, index = wm.pack_state()

    n_odom = sum(1 for f in factors if f.type == "odom_se3")
    n_obs = sum(1 for f in factors if f.type == "voxel_point_obs")

    def residual(x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        """
        x: flat state
        theta["odom"]: shape (n_odom, 6)
        theta["obs"]:  shape (n_obs, 3)
        """
        var_values = wm.unpack_state(x, index)
        res_list = []
        odom_idx = 0
        obs_idx = 0

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn registered for factor type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

            if f.type == "odom_se3":
                meas = theta["odom"][odom_idx]
                odom_idx += 1
                params = dict(f.params)
                params["measurement"] = meas
            elif f.type == "voxel_point_obs":
                point_world = theta["obs"][obs_idx]
                obs_idx += 1
                params = dict(f.params)
                params["point_world"] = point_world
            else:
                params = f.params

            r = res_fn(stacked, params)  # (k,)
            res_list.append(r)

        return jnp.concatenate(res_list)

    return residual, index, n_odom, n_obs


def main():
    print("=== 4.d – HERO Hybrid SE3 + Voxel joint param learning (exp16, GD inner) ===\n")

    wm, pose_ids, voxel_ids = build_hybrid_graph()
    residual_param, index, n_odom, n_obs = build_param_residual(wm)

    # Initial parameters theta
    # Ground-truth odom should be ~[1,0,0,0,0,0] between poses; start biased.
    theta_odom0 = jnp.stack(
        [
            jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([1.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ],
        axis=0,
    )
    assert theta_odom0.shape == (n_odom, 6)

    # Voxel observations: noisy points near GT [i,0,0]
    obs_list = []
    for i in range(n_obs):
        x_gt = float(i)
        obs_list.append(
            jnp.array(
                [
                    x_gt + 0.3 * (0.5 - i / max(1.0, n_obs - 1.0)),
                    0.15 * ((-1.0) ** i),
                    0.0,
                ],
                dtype=jnp.float32,
            )
        )
    theta_obs0 = jnp.stack(obs_list, axis=0)

    theta0 = {"odom": theta_odom0, "obs": theta_obs0}

    # Pack initial state once
    x0, _ = wm.pack_state()

    # Inner GD config (a bit conservative; we differentiate through this)
    gd_cfg = GDConfig(learning_rate=0.05, max_iters=80)

    def inner_solve(theta: dict) -> jnp.ndarray:
        """Gradient-descent inner solver over poses + voxels, given theta."""
        def objective(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_param(x, theta)
            return 0.5 * jnp.sum(r * r)

        return gradient_descent(objective, x0, gd_cfg)

    # Supervised loss: encourage
    #   - final pose tx -> 5.0
    #   - voxel centers x -> [0,1,2,3,4,5] and y -> 0
    def supervised_loss(theta: dict) -> jnp.ndarray:
        x_opt = inner_solve(theta)
        values = wm.unpack_state(x_opt, index)

        # Pose loss on last pose
        pose_last = values[pose_ids[-1]]
        tx_last = pose_last[0]
        loss_pose = (tx_last - 5.0) ** 2

        # Voxel losses
        loss_vox = 0.0
        for i, vid in enumerate(voxel_ids):
            v = values[vid]
            x_gt = float(i)
            loss_vox = loss_vox + (v[0] - x_gt) ** 2 + 0.1 * (v[1] ** 2)

        loss_vox = loss_vox / float(len(voxel_ids))

        return loss_pose + loss_vox

    loss_fn = supervised_loss
    grad_fn = jax.grad(supervised_loss)

    # --- Initial loss / grad ---
    loss0 = float(loss_fn(theta0))
    g0 = grad_fn(theta0)

    def grad_norm(g: dict) -> float:
        return float(
            jnp.sqrt(
                jnp.sum(g["odom"] ** 2) + jnp.sum(g["obs"] ** 2)
            )
        )

    print("Initial theta['odom']:\n", theta0["odom"])
    print("Initial theta['obs']:\n", theta0["obs"])
    print(f"Initial supervised loss: {loss0:.6f}")
    print(f"Initial grad norm: {grad_norm(g0):.6f}\n")

    # --- Outer gradient descent on theta ---
    lr = 0.2
    steps = 30

    theta = theta0
    for it in range(steps):
        g = grad_fn(theta)

        # Simple PyTree update: theta <- theta - lr * g
        theta = {
            "odom": theta["odom"] - lr * g["odom"],
            "obs": theta["obs"] - lr * g["obs"],
        }

        if it % 2 == 0 or it == steps - 1:
            loss_t = float(loss_fn(theta))
            print(
                f"iter {it:02d}: loss={loss_t:.6f}, ||g||={grad_norm(g):.6f}"
            )

    # --- Final results ---
    print("\n=== Final learned parameters ===")
    print("theta['odom']:\n", theta["odom"])
    print("theta['obs']:\n", theta["obs"])

    # Final optimized state
    x_final = inner_solve(theta)
    values = wm.unpack_state(x_final, index)

    print("\n=== Final optimized state ===")
    for i, pid in enumerate(pose_ids):
        print(f"pose{i}:", values[pid])
    for i, vid in enumerate(voxel_ids):
        print(f"voxel{i}:", values[vid])

    print("\nGround truth voxel centers:")
    print(jnp.stack([jnp.array([float(i), 0.0, 0.0]) for i in range(len(voxel_ids))]))


if __name__ == "__main__":
    main()