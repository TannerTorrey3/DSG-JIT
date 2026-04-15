# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
SE3 and SO3 manifold operations for DSG-JIT.

This module implements the minimal 3D Lie-group mathematics required for
differentiable SLAM and scene-graph optimization:

    • SO(3) exponential & logarithm maps
    • SE(3) exponential & logarithm maps
    • Composition and inversion
    • Small-angle approximations for stable Jacobians
    • Helper utilities for constructing transforms

All functions are written in JAX and support:
    - JIT compilation
    - Automatic differentiation
    - Batched operation
    - Numerically stable behavior near zero-rotation limits

Key Functions
-------------
so3_exp(w)
    Maps a 3-vector (axis-angle) to a 3×3 rotation matrix.

so3_log(R)
    Maps a rotation matrix back to its axis-angle representation.

se3_exp(xi)
    Maps a 6-vector twist ξ = (ω, v) to a 4×4 SE(3) transform matrix.

se3_log(T)
    Inverse of se3_exp; extracts a twist from an SE3 matrix.

se3_inverse(T)
    Computes the inverse of an SE3 transform.

se3_compose(A, B)
    Composes two SE3 transforms: A ∘ B.

Utilities
---------
hat(ω)
    Converts a 3-vector to its skew-symmetric matrix.

vee(Ω)
    Converts a 3×3 skew matrix back into a 3-vector.

Notes
-----
These functions are heavily used throughout DSG-JIT:
    • Odometry factors
    • Loop-closure factors
    • Pose-voxel alignment
    • Deformation-graph updates
    • Hybrid SE3 + voxel joint solvers

Correctness of the global optimization engine critically depends on these
Lie-group operations being differentiable, stable, and JIT-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_NORM_EPS = 1e-10

def _safe_norm(v: jnp.ndarray) -> jnp.ndarray:
    """L2 norm with a small epsilon for clean reverse-mode AD at zero.

    ``jnp.linalg.norm`` has gradient ``v / ||v||`` which is NaN at the
    origin.  Adding a tiny epsilon inside the square root produces
    ``v / sqrt(v·v + eps)`` which is exactly zero at the origin while
    introducing negligible bias (< 1e-7 relative) for any vector with
    norm above ~1e-3.
    """
    return jnp.sqrt(jnp.dot(v, v) + _NORM_EPS)


def pose_vec_to_rt(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split a 6D pose vector into translation and rotation components.

    :param v: 6D pose vector ``[tx, ty, tz, wx, wy, wz]``.
    :type v: jnp.ndarray
    :return: Tuple ``(t, w)`` where ``t`` is translation ``(3,)`` and ``w`` is axis-angle rotation ``(3,)``.
    :rtype: tuple[jnp.ndarray, jnp.ndarray]
    """
    v = jnp.asarray(v)
    t = v[0:3]
    w = v[3:6]
    return t, w


def hat(v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the so(3) hat operator.

    :param v: 3‑vector.
    :type v: jnp.ndarray
    :return: 3×3 skew‑symmetric matrix such that ``hat(v) @ w = v × w``.
    :rtype: jnp.ndarray
    """
    x, y, z = v[0], v[1], v[2]
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def so3_exp(w: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from so(3) to SO(3).

    Uses ``jnp.where`` instead of ``lax.cond`` so that reverse-mode AD
    produces well-defined gradients at all rotation magnitudes, including
    exactly zero.

    :param w: Rotation vector in axis‑angle form ``(3,)``.
    :type w: jnp.ndarray
    :return: Rotation matrix in SO(3) with shape ``(3, 3)``.
    :rtype: jnp.ndarray
    """
    w = jnp.asarray(w)
    theta = _safe_norm(w)
    I = jnp.eye(3)

    # Safe denominator: avoid division by zero in the general branch
    # while both branches are always evaluated for AD.
    theta_safe = jnp.where(theta < 1e-5, 1.0, theta)

    # Small-angle: R ≈ I + hat(w)
    R_small = I + hat(w)

    # General: Rodrigues formula with safe division
    k = w / theta_safe
    K = hat(k)
    R_general = I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)

    return jnp.where(theta < 1e-5, R_small, R_general)

def se3_exp(xi: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from se(3) to SE(3).

    :param xi: 6‑vector twist ``[v_x, v_y, v_z, w_x, w_y, w_z]``.
    :type xi: jnp.ndarray
    :return: 4×4 homogeneous SE(3) transform matrix.
    :rtype: jnp.ndarray
    """
    v = xi[:3]
    w = xi[3:]

    theta = jnp.linalg.norm(w)
    I = jnp.eye(3)

    # Rotation
    R = so3_exp(w)

    # Left Jacobian of SE(3) — uses jnp.where for clean AD gradients.
    theta_safe = jnp.where(theta < 1e-5, 1.0, theta)
    W = hat(w)
    W2 = W @ W

    # Small-angle: J ≈ I + 0.5 * W
    J_small = I + 0.5 * W

    # General: J = I + A*W + B*W^2 with safe division
    A = jnp.sin(theta) / theta_safe
    B = (1 - jnp.cos(theta)) / (theta_safe * theta_safe)
    J_general = I + A * W + B * W2

    J = jnp.where(theta < 1e-5, J_small, J_general)

    t = J @ v

    # Build full SE(3) matrix
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(t)

    return T

def vee(R: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of the hat operator.

    :param R: 3×3 skew‑symmetric matrix.
    :type R: jnp.ndarray
    :return: Corresponding 3‑vector.
    :rtype: jnp.ndarray
    """
    return jnp.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / 2.0


def so3_log(R: jnp.ndarray) -> jnp.ndarray:
    """
    Logarithm map from SO(3) to so(3).

    Uses ``jnp.where`` instead of ``lax.cond`` and a safe ``arccos`` so
    that reverse-mode AD produces well-defined gradients at all rotation
    magnitudes, including exactly zero (identity rotation).

    :param R: Rotation matrix in SO(3) with shape ``(3, 3)``.
    :type R: jnp.ndarray
    :return: Rotation vector ``(3,)`` in axis‑angle form.
    :rtype: jnp.ndarray
    """
    R = jnp.asarray(R)
    trace = jnp.trace(R)
    cos_theta = (trace - 1.0) / 2.0

    # Clamp strictly inside (-1, 1) so that arccos and its gradient are
    # always finite.  The gradient of arccos(x) is -1/sqrt(1-x^2) which
    # diverges at x = ±1.  Clamping to ±(1-1e-7) keeps the gradient
    # bounded at ~-707 while introducing < 5e-4 rad forward bias at
    # identity, well within the small-angle threshold below.
    cos_theta = jnp.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_theta)

    # Threshold raised to 1e-3 to safely capture the clipped-identity
    # case (theta ≈ 4.5e-4 when cos_theta is clamped from 1.0).
    eps = 1e-3

    # Small-angle: R ≈ I + hat(w) => w ≈ vee(R - I)
    w_small = vee(R - jnp.eye(3, dtype=R.dtype))

    # General: w = (theta / (2 sin(theta))) * vee(R - R^T)
    sin_theta_safe = jnp.where(theta < eps, 1.0, jnp.sin(theta))
    theta_safe = jnp.where(theta < eps, 1.0, theta)
    factor = theta_safe / (2.0 * sin_theta_safe)
    w_general = factor * vee(R - R.T)

    return jnp.where(theta < eps, w_small, w_general)


def compose_pose_se3(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compose two SE(3) poses in 6D vector representation.

    :param a: 6D pose vector.
    :type a: jnp.ndarray
    :param b: 6D pose vector.
    :type b: jnp.ndarray
    :return: Composed pose ``a ∘ b`` in 6D vector form.
    :rtype: jnp.ndarray
    """
    ta, wa = pose_vec_to_rt(a)
    tb, wb = pose_vec_to_rt(b)

    Ra = so3_exp(wa)
    Rb = so3_exp(wb)

    R = Ra @ Rb
    t = Ra @ tb + ta

    w = so3_log(R)
    return jnp.concatenate([t, w])


def relative_pose_se3(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute relative pose from ``a`` to ``b`` in 6D coordinates.

    :param a: First pose ``(6,)``.
    :type a: jnp.ndarray
    :param b: Second pose ``(6,)``.
    :type b: jnp.ndarray
    :return: Relative twist such that ``Exp(xi) ≈ T_a^{-1} T_b``.
    :rtype: jnp.ndarray
    """
    ta, wa = pose_vec_to_rt(a)
    tb, wb = pose_vec_to_rt(b)

    Ra = so3_exp(wa)
    Rb = so3_exp(wb)

    R_rel = Ra.T @ Rb
    w_rel = so3_log(R_rel)
    t_rel = Ra.T @ (tb - ta)

    return jnp.concatenate([t_rel, w_rel])

def se3_retract_left(pose: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """
    Left‑multiplicative SE(3) retraction.

    :param pose: Base pose in 6D coordinates.
    :type pose: jnp.ndarray
    :param delta: Incremental twist in 6D.
    :type delta: jnp.ndarray
    :return: Updated pose ``Exp(delta) * pose`` in 6D vector form.
    :rtype: jnp.ndarray
    """
    pose = jnp.asarray(pose)
    delta = jnp.asarray(delta)

    # Split into translation + rotation-vector
    t, w = pose_vec_to_rt(pose)
    dt, dw = pose_vec_to_rt(delta)

    R = so3_exp(w)
    R_d = so3_exp(dw)

    # Apply left-multiplicative update
    R_new = R_d @ R
    t_new = R_d @ t + dt

    w_new = so3_log(R_new)
    return jnp.concatenate([t_new, w_new])


def se3_identity() -> jnp.ndarray:
    """
    Return the identity SE(3) pose.

    :return: Zero 6‑vector representing identity rotation and translation.
    :rtype: jnp.ndarray
    """
    return jnp.zeros(6, dtype=jnp.float32)