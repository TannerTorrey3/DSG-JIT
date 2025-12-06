import jax
import jax.numpy as jnp

from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.world.training import DSGTrainer, InnerGDConfig
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual


def build_scenegraph():
    sg = SceneGraphWorld()
    wm = sg.wm

    # Register residuals at the WorldModel level.
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)

    p0 = sg.add_pose_se3(
        jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )
    p1 = sg.add_pose_se3(
        jnp.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )
    p2 = sg.add_pose_se3(
        jnp.array([1.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )

    wm.add_factor(
        "prior",
        (p0,),
        {
            "target": jnp.zeros(6, dtype=jnp.float32),
            "weight": jnp.array(0.1, dtype=jnp.float32),  # weak
        },
    )

    sg.add_prior_pose_identity(p0)

    sg.add_odom_se3_additive(p0, p1, dx=0.5)
    sg.add_odom_se3_additive(p1, p2, dx=0.5)

    '''wm.add_factor(
        "prior",
        (p2,),
        {"target": jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)},
    )'''

    return sg, p0, p1, p2


def main():
    print("=== SceneGraph Learnable Type Weight Experiment (#12, Trainer) ===\n")

    sg, p0, p1, p2 = build_scenegraph()
    wm = sg.wm

    factor_type_order = ["prior", "odom_se3"]
    inner_cfg = InnerGDConfig(learning_rate=0.02, max_iters=40, max_step_norm=0.5)
    trainer = DSGTrainer(wm, factor_type_order, inner_cfg)

    # Debug: residual at x0 with unit weights, using WorldModel-level residuals.
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry
    type_to_idx = {t: i for i, t in enumerate(factor_type_order)}

    def residual_w(x: jnp.ndarray, log_scales: jnp.ndarray) -> jnp.ndarray:
        """
        Residual with per-type weights controlled by log_scales.

        log_scales[i] corresponds to factor_type_order[i].
        Missing types default to unit weight.
        """
        _, index = wm.pack_state()
        var_values = wm.unpack_state(x, index)
        res_list = []

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn registered for factor type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids], axis=0)
            r = res_fn(stacked, f.params)

            idx = type_to_idx.get(f.type, None)
            if idx is not None:
                scale = jnp.exp(log_scales[idx])
            else:
                scale = 1.0

            r_scaled = scale * r
            r_scaled = jnp.reshape(r_scaled, (-1,))
            res_list.append(r_scaled)

        return jnp.concatenate(res_list, axis=0)

    x0, index0 = wm.pack_state()
    log_scales_init = jnp.array([0.0, 0.0], dtype=jnp.float32)
    r0 = residual_w(x0, log_scales_init)
    print("DEBUG r0:", r0)
    print("DEBUG any NaN in r0:", bool(jnp.any(jnp.isnan(r0))))

    def supervised_loss_scalar(log_scale_odom: jnp.ndarray) -> jnp.ndarray:
        log_scales = jnp.array([0.0, log_scale_odom], dtype=jnp.float32)
        x_opt = trainer.solve_state(log_scales)
        values = trainer.unpack_state(x_opt)
        pose2 = values[p2]
        tx2 = pose2[0]
        return (tx2 - 2.0) ** 2

    # DO NOT jit yet – keep it simple for debugging
    loss_fn = supervised_loss_scalar
    grad_fn = jax.grad(supervised_loss_scalar)

    log_scale_odom = jnp.array(0.0, dtype=jnp.float32)
    loss0 = float(loss_fn(log_scale_odom))
    print(f"Initial log_scale_odom = {float(log_scale_odom):.4f}")
    print(f"Initial supervised loss (pose2.tx vs 2.0) = {loss0:.9f}")

    lr_outer = 5.0
    steps = 50

    for it in range(steps):
        g = float(grad_fn(log_scale_odom))
        log_scale_odom = log_scale_odom - lr_outer * g
        loss_t = float(loss_fn(log_scale_odom))
        print(
            f"iter {it:02d}: loss = {loss_t:.9f}, "
            f"log_scale_odom = {float(log_scale_odom):.9f}, grad = {g:.9e}"
        )

    final_log_scales = jnp.array([0.0, log_scale_odom], dtype=jnp.float32)
    x_final = trainer.solve_state(final_log_scales)
    values = trainer.unpack_state(x_final)

    pose0 = values[p0]
    pose1 = values[p1]
    pose2 = values[p2]

    print("\n--- Final optimized poses with learned odom_se3 weight ---")
    print("pose0:", pose0)
    print("pose1:", pose1)
    print("pose2:", pose2)
    print(f"\npose2.tx = {float(pose2[0]):.6f} (target 2.0)")


if __name__ == "__main__":
    main()