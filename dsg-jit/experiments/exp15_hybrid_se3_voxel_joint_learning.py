import jax
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.optimization.solvers import gradient_descent, GDConfig
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,                 # additive SE3 odom residual (R^6)
    voxel_point_observation_residual,  # your voxel_point_obs residual name
)

# -------------------------------------------------------------------
# 1. Build a hybrid SE3 + voxel factor graph
# -------------------------------------------------------------------

def build_hybrid_factor_graph():
    """
    Hybrid factor graph with:
      - 3 SE3 poses (pose_se3) in R^6
      - 3 voxel cells (voxel3d) in R^3

    Ground-truth we conceptually aim for:
      pose0: [0, 0, 0, 0, 0, 0]
      pose1: [1, 0, 0, 0, 0, 0]
      pose2: [2, 0, 0, 0, 0, 0]

      voxel0: [0, 0, 0]
      voxel1: [1, 0, 0]
      voxel2: [2, 0, 0]

    But we will use *biased* odom and noisy voxel observations as learnable params.
    """

    fg = FactorGraph()

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # --- Variables ---

    # SE3 poses (tx, ty, tz, wx, wy, wz)
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    pose2 = Variable(
        id=NodeId(2),
        type="pose_se3",
        value=jnp.array([1.7, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    # Voxel centers (3D Euclidean)
    voxel0 = Variable(
        id=NodeId(3),
        type="voxel3d",
        value=jnp.array([-0.3, 0.1, 0.0], dtype=jnp.float32),
    )
    voxel1 = Variable(
        id=NodeId(4),
        type="voxel3d",
        value=jnp.array([0.9, -0.2, 0.0], dtype=jnp.float32),
    )
    voxel2 = Variable(
        id=NodeId(5),
        type="voxel3d",
        value=jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32),
    )

    for v in [pose0, pose1, pose2, voxel0, voxel1, voxel2]:
        fg.add_variable(v)

    # --- Factors ---

    # Prior on pose0 ~ identity
    f_prior_p0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6, dtype=jnp.float32), "weight": 1.0},
    )

    # (Optional) weak priors on voxels near [0,1,2] along x, small weight
    voxel_prior_weight = 0.1
    f_prior_v0 = Factor(
        id=FactorId(1),
        type="prior",
        var_ids=(NodeId(3),),
        params={"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": voxel_prior_weight},
    )
    f_prior_v1 = Factor(
        id=FactorId(2),
        type="prior",
        var_ids=(NodeId(4),),
        params={"target": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": voxel_prior_weight},
    )
    f_prior_v2 = Factor(
        id=FactorId(3),
        type="prior",
        var_ids=(NodeId(5),),
        params={"target": jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": voxel_prior_weight},
    )

    # Biased odometry along x (dx is learned later via theta)
    # For now, just put placeholders; 'measurement' will be overwritten by theta.
    f_odom01 = Factor(
        id=FactorId(4),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": jnp.array([0.9, 0, 0, 0, 0, 0], dtype=jnp.float32),
                "weight": 1.0},
    )
    f_odom12 = Factor(
        id=FactorId(5),
        type="odom_se3",
        var_ids=(NodeId(1), NodeId(2)),
        params={"measurement": jnp.array([1.1, 0, 0, 0, 0, 0], dtype=jnp.float32),
                "weight": 1.0},
    )

    # Voxel point observations – their 'point_world' will come from theta
    # We keep initial noisy guesses in params for readability; we overwrite at runtime.
    f_obs0 = Factor(
        id=FactorId(6),
        type="voxel_point_obs",
        var_ids=(NodeId(3),),
        params={
            "point_world": jnp.array([-0.5, 0.1, 0.0], dtype=jnp.float32),
            "weight": 1.0,
        },
    )
    f_obs1 = Factor(
        id=FactorId(7),
        type="voxel_point_obs",
        var_ids=(NodeId(4),),
        params={
            "point_world": jnp.array([0.7, -0.2, 0.0], dtype=jnp.float32),
            "weight": 1.0,
        },
    )
    f_obs2 = Factor(
        id=FactorId(8),
        type="voxel_point_obs",
        var_ids=(NodeId(5),),
        params={
            "point_world": jnp.array([2.4, 0.3, 0.0], dtype=jnp.float32),
            "weight": 1.0,
        },
    )

    for f in [f_prior_p0, f_prior_v0, f_prior_v1, f_prior_v2,
              f_odom01, f_odom12, f_obs0, f_obs1, f_obs2]:
        fg.add_factor(f)

    return fg, (NodeId(0), NodeId(1), NodeId(2)), (NodeId(3), NodeId(4), NodeId(5))


# -------------------------------------------------------------------
# 2. Hybrid parametric residual: depends on both odom meas and voxel obs
# -------------------------------------------------------------------

def build_hybrid_param_residual(fg: FactorGraph):
    """
    Build residual(x, theta) where:
      theta = {'odom': (K_odom, 6), 'obs': (K_obs, 3)}

    - odom_se3 factors get their 'measurement' from theta['odom'][odom_idx]
    - voxel_point_obs factors get 'point_world' from theta['obs'][obs_idx]
    - all other factors use their original params
    """
    factors = list(fg.factors.values())
    residual_fns = dict(fg.residual_fns)
    x0, index = fg.pack_state()

    def unpack(x: jnp.ndarray):
        return fg.unpack_state(x, index)

    def residual(x: jnp.ndarray, theta) -> jnp.ndarray:
        var_values = unpack(x)
        res_list = []
        odom_idx = 0
        obs_idx = 0

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn for factor type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids], axis=0)

            if f.type == "odom_se3":
                meas = theta["odom"][odom_idx]  # (6,)
                odom_idx += 1
                base_params = dict(f.params)
                base_params["measurement"] = meas
                params = base_params
            elif f.type == "voxel_point_obs":
                pt = theta["obs"][obs_idx]  # (3,)
                obs_idx += 1
                base_params = dict(f.params)
                base_params["point_world"] = pt
                params = base_params
            else:
                params = f.params

            r = res_fn(stacked, params)
            w = params.get("weight", 1.0)
            res_list.append(jnp.sqrt(w) * r)

        return jnp.concatenate(res_list, axis=0)

    return residual, x0, index


# -------------------------------------------------------------------
# 3. Inner solver: optimize x given theta via GD on 0.5 * ||r||^2
# -------------------------------------------------------------------

def solve_inner(fg: FactorGraph, residual_param, x0: jnp.ndarray, theta, gd_cfg: GDConfig):
    def obj(x):
        r = residual_param(x, theta)
        return 0.5 * jnp.sum(r * r)

    return gradient_descent(obj, x0, gd_cfg)


# -------------------------------------------------------------------
# 4. Outer supervised loss and joint learning of theta
# -------------------------------------------------------------------

def main():
    print("=== 4.d – Hybrid SE3 + Voxel joint param learning (exp15) ===\n")

    fg, pose_ids, voxel_ids = build_hybrid_factor_graph()
    residual_param, x0, index = build_hybrid_param_residual(fg)

    # Ground-truth we conceptually want:
    voxel_gt = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    # Initial theta for odom (two odom edges) and obs (three voxel obs)
    theta = {
        "odom": jnp.array(
            [
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],  # pose0 -> pose1
                [1.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # pose1 -> pose2
            ],
            dtype=jnp.float32,
        ),
        "obs": jnp.array(
            [
                [-0.5, 0.1, 0.0],
                [0.7, -0.2, 0.0],
                [2.4, 0.3, 0.0],
            ],
            dtype=jnp.float32,
        ),
    }

    # Inner GD config (state optimization)
    gd_cfg = GDConfig(learning_rate=0.05, max_iters=80)

    def supervised_loss(theta):
        x_opt = solve_inner(fg, residual_param, x0, theta, gd_cfg)
        values = fg.unpack_state(x_opt, index)

        p0 = values[pose_ids[0]]
        p1 = values[pose_ids[1]]
        p2 = values[pose_ids[2]]

        v0 = values[voxel_ids[0]]
        v1 = values[voxel_ids[1]]
        v2 = values[voxel_ids[2]]

        voxels = jnp.stack([v0, v1, v2], axis=0)

        # Pose supervision: push pose2.tx to 2.0
        pose_loss = (p2[0] - 2.0) ** 2

        # Voxel supervision: push voxel centers towards [0,1,2] on x
        voxel_loss = jnp.sum((voxels - voxel_gt) ** 2)

        # Combine with some weights
        return 1.0 * pose_loss + 0.5 * voxel_loss

    loss_fn = jax.jit(supervised_loss)
    grad_fn = jax.jit(jax.grad(supervised_loss))

    # Initial diagnostics
    loss0 = float(loss_fn(theta))
    g0 = grad_fn(theta)
    g_norm0 = float(
        jnp.sqrt(
            jnp.sum(g0["odom"] ** 2) + jnp.sum(g0["obs"] ** 2)
        )
    )

    print("Initial theta['odom']:\n", theta["odom"])
    print("Initial theta['obs']:\n", theta["obs"])
    print(f"Initial supervised loss: {loss0:.6f}")
    print(f"Initial grad norm: {g_norm0:.6f}\n")

    # Outer GD over theta
    outer_lr = 0.1
    steps = 30

    def tree_add(theta, update):
        return {
            "odom": theta["odom"] + update["odom"],
            "obs": theta["obs"] + update["obs"],
        }

    for it in range(steps):
        g = grad_fn(theta)
        update = {
            "odom": -outer_lr * g["odom"],
            "obs": -outer_lr * g["obs"],
        }
        theta = tree_add(theta, update)
        if it % 2 == 0 or it == steps - 1:
            loss_t = float(loss_fn(theta))
            g_norm = float(
                jnp.sqrt(
                    jnp.sum(g["odom"] ** 2) + jnp.sum(g["obs"] ** 2)
                )
            )
            print(
                f"iter {it:02d}: loss={loss_t:.6f}, "
                f"||g||={g_norm:.6f}"
            )

    # Final optimized state for inspection
    x_final = solve_inner(fg, residual_param, x0, theta, gd_cfg)
    values = fg.unpack_state(x_final, index)

    p0 = values[pose_ids[0]]
    p1 = values[pose_ids[1]]
    p2 = values[pose_ids[2]]

    v0 = values[voxel_ids[0]]
    v1 = values[voxel_ids[1]]
    v2 = values[voxel_ids[2]]

    print("\n=== Final learned parameters ===")
    print("theta['odom']:\n", theta["odom"])
    print("theta['obs']:\n", theta["obs"])

    print("\n=== Final optimized state ===")
    print("pose0:", p0)
    print("pose1:", p1)
    print("pose2:", p2)
    print("voxel0:", v0)
    print("voxel1:", v1)
    print("voxel2:", v2)

    print("\nGround truth voxel centers:")
    print(voxel_gt)


if __name__ == "__main__":
    main()