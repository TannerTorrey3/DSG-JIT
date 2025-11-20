
from __future__ import annotations

import jax.numpy as jnp
import pytest

from core.math3d import se3_identity, se3_retract_left, relative_pose_se3


def test_se3_retract_zero_delta_is_identity():
    pose = se3_identity()
    delta = jnp.zeros(6)

    pose_new = se3_retract_left(pose, delta)
    assert jnp.allclose(pose_new, pose, atol=1e-7)


def test_se3_retract_pure_translation():
    pose = se3_identity()
    delta = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pose_new = se3_retract_left(pose, delta)
    t_new = pose_new[:3]
    w_new = pose_new[3:]

    assert jnp.allclose(t_new, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert jnp.allclose(w_new, jnp.zeros(3), atol=1e-6)


def test_se3_retract_matches_relative_pose_for_small_delta():
    """
    For a small delta, if we apply it to identity, the relative pose from
    identity to the result should be approximately delta.
    """
    delta = jnp.array([0.1, -0.05, 0.02, 0.01, 0.0, -0.02])
    pose0 = se3_identity()
    pose1 = se3_retract_left(pose0, delta)

    xi_est = relative_pose_se3(pose0, pose1)
    assert jnp.allclose(xi_est, delta, atol=1e-3)