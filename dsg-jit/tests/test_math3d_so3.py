import jax.numpy as jnp
from core.math3d import so3_exp, so3_log


def test_so3_log_exp_roundtrip_small_angle():
    w = jnp.array([0.1, -0.05, 0.02])
    R = so3_exp(w)
    w_est = so3_log(R)
    assert jnp.all(jnp.isfinite(w_est))
    assert jnp.allclose(w_est, w, atol=1e-4)


def test_so3_log_no_nan_for_identity():
    R = jnp.eye(3)
    w = so3_log(R)
    assert jnp.all(jnp.isfinite(w))
    assert jnp.linalg.norm(w) < 1e-6