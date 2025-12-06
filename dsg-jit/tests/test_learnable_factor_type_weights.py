import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual
from dsg_jit.optimization.solvers import GDConfig, gradient_descent


def _to_slice(idx):
    """Normalize FactorGraph index entry (slice or (start, length)) into a slice."""
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def _build_chain():
    """
    Simple 2-pose SE(3) chain to test learnable factor-type weights.

    State:
        pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]

    Ground truth:
        pose0 = [0, 0, 0, 0, 0, 0]
        pose1 = [1, 0, 0, 0, 0, 0]

    Factors:
        - prior on pose0 (type: 'prior')
        - odom_se3 between pose0 and pose1 (type: 'odom_se3')

    We will:
        - build a weighted residual function with type-level log_scales
        - optimize x with GD for given log_scales
        - define a loss vs GT
        - backpropagate loss wrt log_scales
        - take one grad step on log_scales and check loss decreases
    """
    wm = WorldModel()

    # Slightly off initial guesses
    p0_val = jnp.array(
        [0.2, -0.1, 0.0, 0.01, -0.02, 0.0], dtype=jnp.float32
    )
    p1_val = jnp.array(
        [0.8, 0.2, 0.0, 0.02, 0.01, 0.0], dtype=jnp.float32
    )

    # Add pose variables via WorldModel and keep track of their NodeIds
    p0_id = wm.add_variable(var_type="pose_se3", value=p0_val)
    p1_id = wm.add_variable(var_type="pose_se3", value=p1_val)

    # Prior on pose0 at identity
    wm.add_factor(
        f_type="prior",
        var_ids=(p0_id,),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Additive odometry measurement (slightly off)
    meas01 = jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(p0_id, p1_id),
        params={"measurement": meas01},
    )

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)

    # Pack state and slices using WorldModel's packing
    x_init, index = wm.pack_state()
    p0_slice = _to_slice(index[p0_id])
    p1_slice = _to_slice(index[p1_id])

    # Ground truth poses
    gt0 = jnp.zeros(6, dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt_stack = jnp.stack([gt0, gt1], axis=0)

    return wm, x_init, (p0_slice, p1_slice), gt_stack


def test_learnable_factor_type_weights_scaling_and_grad():
    """
    Regression test for per-factor-type weights:

    We:
      - build a 2-pose SE(3) problem with a 'prior' and an 'odom_se3' factor,
      - build r(x, log_scales) where log_scales has one entry for 'odom_se3',
      - check that:
          * the prior residual components are unchanged by log_scales,
          * the odom residual components scale by exp(log_scale),
          * gradients wrt log_scales are finite (no NaNs).
    """
    wm, x_init, (p0_slice, p1_slice), _ = _build_chain()

    # Only learn/scale 'odom_se3'. 'prior' remains unweighted implicitly (scale=1).
    factor_type_order = ["odom_se3"]

    # Build the weighted residual using the WorldModel hyper-parameter builder.
    # This is the analogue of the old FactorGraph.build_residual_function_with_type_weights.
    residual_w = wm.build_residual_function_with_type_weights(factor_type_order)

    # Two different log_scales for odom: 0 -> scale 1,  ln(2) -> scale 2
    log_s0 = jnp.array([0.0], dtype=jnp.float32)           # scale = 1
    log_s1 = jnp.array([jnp.log(2.0)], dtype=jnp.float32)  # scale = 2

    # Compute residuals at the same x, different log_scales
    r0 = residual_w(x_init, log_s0)  # shape (m,)
    r1 = residual_w(x_init, log_s1)  # shape (m,)

    # Sanity: residuals must be same shape and finite
    assert r0.shape == r1.shape
    assert jnp.all(jnp.isfinite(r0))
    assert jnp.all(jnp.isfinite(r1))

    # We know residual vector layout: first factor is 'prior', second is 'odom_se3'.
    # Each residual is 6D, so the layout is:
    #   r = [r_prior(0:6), r_odom(6:12)]
    r_prior_0 = r0[0:6]
    r_odom_0  = r0[6:12]

    r_prior_1 = r1[0:6]
    r_odom_1  = r1[6:12]

    # Prior residual must not change when scaling odom
    assert jnp.allclose(r_prior_0, r_prior_1, atol=1e-6)

    # Odom residual must be scaled by ~2
    assert jnp.allclose(r_odom_1, 2.0 * r_odom_0, atol=1e-5)

    # Now define a simple scalar function of log_scales to test differentiability
    def phi(log_scales: jnp.ndarray) -> jnp.ndarray:
        r = residual_w(x_init, log_scales)
        return jnp.sum(r * r)

    grad_log = jax.grad(phi)(log_s0)

    # Gradient must be finite and non-NaN
    assert grad_log.shape == log_s0.shape
    assert jnp.all(jnp.isfinite(grad_log))