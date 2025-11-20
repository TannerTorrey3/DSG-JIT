import jax
import jax.numpy as jnp

from core.factor_graph import FactorGraph
from core.types import NodeId, FactorId, Variable, Factor
from optimization.solvers import GDConfig, gradient_descent
from slam.measurements import (
    prior_residual,
    odom_se3_residual,
    voxel_point_observation_residual,
)


def build_hybrid_graph_for_test():
    """
    Minimal 'hero-style' hybrid graph for regression:

    - 3 SE(3) poses: p0, p1, p2
    - 3 voxel3d cells: v0, v1, v2
    - prior(p0) ~ 0
    - odom_se3(p0->p1), odom_se3(p1->p2) with learnable measurements
    - voxel_point_obs(p_i, v_i) with learnable point_world

    Ground truth we aim for:
      pose tx ~ [0, 1, 2]
      voxel centers ~ [0, 1, 2] on x, y ~ 0
    """

    fg = FactorGraph()

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # --- Variables: 3 poses + 3 voxels ---

    pose_ids = []
    voxel_ids = []

    # Poses: small perturbations around [0,1,2] on x
    pose_init = [
        jnp.array([0.1, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.9, -0.01, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([1.8, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    ]

    for i, val in enumerate(pose_init):
        nid = NodeId(i)
        pose_ids.append(nid)
        fg.add_variable(Variable(id=nid, type="pose_se3", value=val))

    # Voxels: rough guesses near x ~ [0,1,2] with y offsets
    voxel_init = [
        jnp.array([-0.1, 0.1, 0.0], dtype=jnp.float32),
        jnp.array([1.2, -0.1, 0.0], dtype=jnp.float32),
        jnp.array([2.1, 0.1, 0.0], dtype=jnp.float32),
    ]

    for i, val in enumerate(voxel_init):
        nid = NodeId(10 + i)  # keep them distinct from poses
        voxel_ids.append(nid)
        fg.add_variable(Variable(id=nid, type="voxel3d", value=val))

    # --- Factors ---

    # Prior on p0: identity in se(3)
    fg.add_factor(
        Factor(
            id=FactorId(0),
            type="prior",
            var_ids=(pose_ids[0],),
            params={"target": jnp.zeros(6, dtype=jnp.float32)},
        )
    )

    # Odometry factors p0->p1, p1->p2 (measurements will be overridden by theta["odom"])
    # We still provide initial measurements so the graph is well-posed without learning.
    odom_meas_init = [
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    ]

    fg.add_factor(
        Factor(
            id=FactorId(1),
            type="odom_se3",
            var_ids=(pose_ids[0], pose_ids[1]),
            params={"measurement": odom_meas_init[0]},
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(2),
            type="odom_se3",
            var_ids=(pose_ids[1], pose_ids[2]),
            params={"measurement": odom_meas_init[1]},
        )
    )

    # Voxel observations: each (pose_i, voxel_i) with point_world as learnable param
    # Here we give a noisy initial guess; theta["obs"] will override.
    obs_init = [
        jnp.array([-0.2, 0.1, 0.0], dtype=jnp.float32),
        jnp.array([1.1, -0.2, 0.0], dtype=jnp.float32),
        jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32),
    ]

    fg.add_factor(
        Factor(
            id=FactorId(3),
            type="voxel_point_obs",
            var_ids=(pose_ids[0], voxel_ids[0]),
            params={"point_world": obs_init[0]},
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(4),
            type="voxel_point_obs",
            var_ids=(pose_ids[1], voxel_ids[1]),
            params={"point_world": obs_init[1]},
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(5),
            type="voxel_point_obs",
            var_ids=(pose_ids[2], voxel_ids[2]),
            params={"point_world": obs_init[2]},
        )
    )

    return fg, pose_ids, voxel_ids


def build_param_residual(fg: FactorGraph):
    """
    Hybrid parameterized residual:

      - theta["odom"]: shape (K_odom, 6), overrides 'measurement' in odom_se3
      - theta["obs"]:  shape (K_obs, 3), overrides 'point_world' in voxel_point_obs

    Returns:
      residual_param(x, theta), index, n_odom, n_obs
    """
    factors = list(fg.factors.values())
    residual_fns = fg.residual_fns
    _, index = fg.pack_state()

    odom_indices = [i for i, f in enumerate(factors) if f.type == "odom_se3"]
    obs_indices = [i for i, f in enumerate(factors) if f.type == "voxel_point_obs"]
    n_odom = len(odom_indices)
    n_obs = len(obs_indices)

    def residual(x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        var_values = fg.unpack_state(x, index)
        res_list = []
        odom_k = 0
        obs_k = 0

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn registered for type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

            # Override params with learned theta where appropriate
            if f.type == "odom_se3":
                params = dict(f.params)
                params["measurement"] = theta["odom"][odom_k]
                odom_k += 1
            elif f.type == "voxel_point_obs":
                params = dict(f.params)
                params["point_world"] = theta["obs"][obs_k]
                obs_k += 1
            else:
                params = f.params

            r = res_fn(stacked, params)
            res_list.append(r)

        return jnp.concatenate(res_list)

    return residual, index, n_odom, n_obs


def _build_initial_theta(n_odom: int, n_obs: int) -> dict:
    assert n_odom == 2, f"Expected 2 odom factors, got {n_odom}"
    assert n_obs == 3, f"Expected 3 voxel obs, got {n_obs}"

    # Slightly biased odom measurements
    theta_odom0 = jnp.stack(
        [
            jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ],
        axis=0,
    )

    # Noisy obs around [0,1,2]
    theta_obs0 = jnp.stack(
        [
            jnp.array([-0.5, 0.1, 0.0], dtype=jnp.float32),
            jnp.array([0.7, -0.2, 0.0], dtype=jnp.float32),
            jnp.array([2.4, 0.3, 0.0], dtype=jnp.float32),
        ],
        axis=0,
    )

    return {"odom": theta_odom0, "obs": theta_obs0}


def test_hero_hybrid_joint_learning_converges():
    """
    Hero-style regression test:

      - Hybrid SE3 + voxel graph.
      - Inner GD solves for poses + voxels given params θ.
      - Outer GD updates θ = {odom measurements, voxel obs points}.

    We assert:
      - The supervised loss decreases substantially.
      - Last pose.tx ≈ 2.0.
      - Voxels line up near x ~ [0,1,2] and y ~ 0.
    """

    fg, pose_ids, voxel_ids = build_hybrid_graph_for_test()
    residual_param, index, n_odom, n_obs = build_param_residual(fg)

    theta0 = _build_initial_theta(n_odom, n_obs)
    x0, _ = fg.pack_state()

    gd_cfg = GDConfig(learning_rate=0.05, max_iters=80)

    def inner_solve(theta: dict) -> jnp.ndarray:
        def objective(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_param(x, theta)
            return 0.5 * jnp.sum(r * r)

        return gradient_descent(objective, x0, gd_cfg)

    def supervised_loss(theta: dict) -> jnp.ndarray:
        x_opt = inner_solve(theta)
        values = fg.unpack_state(x_opt, index)

        # Pose loss on last pose (want tx -> 2.0)
        pose_last = values[pose_ids[-1]]
        tx_last = pose_last[0]
        loss_pose = (tx_last - 2.0) ** 2

        # Voxel loss: x near [0,1,2], y near 0
        loss_vox = 0.0
        for i, vid in enumerate(voxel_ids):
            v = values[vid]
            x_gt = float(i)
            loss_vox = loss_vox + (v[0] - x_gt) ** 2 + 0.1 * (v[1] ** 2)

        loss_vox = loss_vox / float(len(voxel_ids))
        return loss_pose + loss_vox

    loss_fn = supervised_loss
    grad_fn = jax.grad(supervised_loss)

    # Initial loss & grad sanity check
    loss0 = float(loss_fn(theta0))
    g0 = grad_fn(theta0)

    def norm_grad(g: dict) -> float:
        return float(
            jnp.sqrt(jnp.sum(g["odom"] ** 2) + jnp.sum(g["obs"] ** 2))
        )

    assert jnp.isfinite(loss0), "Initial loss is not finite"
    assert jnp.isfinite(norm_grad(g0)), "Initial gradient is not finite"

    # Outer GD on theta
    theta = theta0
    lr_outer = 0.2
    steps = 20

    for _ in range(steps):
        g = grad_fn(theta)
        theta = {
            "odom": theta["odom"] - lr_outer * g["odom"],
            "obs": theta["obs"] - lr_outer * g["obs"],
        }

    loss1 = float(loss_fn(theta))
    # Require a decent reduction in loss
    assert loss1 < 0.5 * loss0, f"Loss did not decrease enough: {loss0} -> {loss1}"

    # Final optimized state: geometric checks
    x_final = inner_solve(theta)
    values = fg.unpack_state(x_final, index)

    pose_last = values[pose_ids[-1]]
    tx_last = float(pose_last[0])
    assert abs(tx_last - 2.0) < 5e-2, f"pose2.tx not close to 2.0, got {tx_last}"

    # Voxel x positions near [0,1,2]
    voxel_x = jnp.array([values[vid][0] for vid in voxel_ids])
    gt_x = jnp.arange(len(voxel_ids), dtype=jnp.float32)
    mse_vox = float(jnp.mean((voxel_x - gt_x) ** 2))
    assert mse_vox < 5e-2, f"Voxel X MSE too large: {mse_vox}"