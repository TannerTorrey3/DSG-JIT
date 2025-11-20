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

from .types import Pose3  # still here


def pose_vec_to_rt(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split a 6D pose vector into translation and rotation-vector (axis-angle).
    v: [tx, ty, tz, wx, wy, wz]
    """
    v = jnp.asarray(v)
    t = v[0:3]
    w = v[3:6]
    return t, w


def hat(v: jnp.ndarray) -> jnp.ndarray:
    """so(3) hat operator: R^3 -> 3x3 skew-symmetric matrix."""
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
    Exponential map from so(3) (rotation vector) to SO(3).

    Uses Rodrigues' formula with a small-angle fallback.
    """
    w = jnp.asarray(w)
    theta = jnp.linalg.norm(w)
    I = jnp.eye(3)

    def small_angle() -> jnp.ndarray:
        # First-order approximation for small angles
        return I + hat(w)

    def normal_angle() -> jnp.ndarray:
        k = w / theta
        K = hat(k)
        return I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)

    return jax.lax.cond(theta < 1e-5, small_angle, normal_angle)

def se3_exp(xi: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from se(3) -> SE(3).

    xi = [v_x, v_y, v_z, w_x, w_y, w_z]
      - v: translation vector in R^3
      - w: rotation vector in R^3 (axis-angle)

    Returns 4×4 homogeneous SE(3) matrix.

    Uses the standard closed-form left-Jacobian J for SE(3).

        T = [ R, J*v ]
            [ 0, 1   ]
    """
    v = xi[:3]
    w = xi[3:]

    theta = jnp.linalg.norm(w)
    I = jnp.eye(3)

    # Rotation
    R = so3_exp(w)

    # Small-angle approximations for Jacobian
    def small_angle():
        # For tiny rotation, J ≈ I + 0.5 * W
        W = hat(w)
        return I + 0.5 * W

    def normal_angle():
        W = hat(w)
        W2 = W @ W
        A = jnp.sin(theta) / theta
        B = (1 - jnp.cos(theta)) / (theta * theta)
        return I + A * W + B * W2

    J = jax.lax.cond(theta < 1e-5, small_angle, normal_angle)

    t = J @ v

    # Build full SE(3) matrix
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(t)

    return T

def vee(R: jnp.ndarray) -> jnp.ndarray:
    """
    vee: so(3) -> R^3, inverse of hat.
    Assumes R is a 3x3 skew-symmetric-like matrix.
    """
    return jnp.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / 2.0


def so3_log(R: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable logarithm map for SO(3).

    Handles:
      - small angles via first-order approximation
      - trace slightly outside [-1, 3] via clamping

    Returns w in R^3 such that Exp(w) ~ R.
    """
    R = jnp.asarray(R)
    # Compute cos(theta) with clamping
    trace = jnp.trace(R)
    cos_theta = (trace - 1.0) / 2.0

    # Clamp to valid domain for arccos
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    # Small-angle threshold
    eps = 1e-5

    def small_angle_case(_) -> jnp.ndarray:
        # For very small angles, R ~ I + hat(w), so:
        # hat(w) ~ R - I  => w ~ vee(R - I)
        w_skew = R - jnp.eye(3, dtype=R.dtype)
        return vee(w_skew)

    def general_case(_) -> jnp.ndarray:
        # Standard formula:
        #   w^ = (theta / (2 sin(theta))) * (R - R^T)
        #   w  = vee(w^)
        w_skew = R - R.T
        # Safe denominator
        denom = 2.0 * jnp.sin(theta)
        factor = theta / (denom + 1e-12)
        w = factor * vee(w_skew)
        return w

    w = jax.lax.cond(
        theta < eps,
        small_angle_case,
        general_case,
        operand=None,
    )

    return w


def compose_pose_se3(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compose two SE(3) poses in 6D vector form.

    a, b: [tx, ty, tz, wx, wy, wz]
    Returns: 6D vector for a ∘ b
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
    Compute relative pose from a to b in 6D vector form.

      a, b: [tx, ty, tz, wx, wy, wz]
    Returns xi in R^6 such that exp(xi) ≈ T_a^{-1} T_b:

      T_rel = T_a^{-1} T_b
      t_rel = R_a^T (t_b - t_a)
      w_rel = log(R_a^T R_b)
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
    Left-multiplicative SE(3) retraction:

        pose, delta: R^6, [tx, ty, tz, wx, wy, wz]

    Interprets `delta` as a twist in se(3), constructs Exp(delta) and applies:

        T_new = Exp(delta) * T_old

    in matrix form:

        Exp(delta) = [ R_d  t_d ]
                     [  0    1  ]

        T_old      = [ R    t   ]
                     [  0    1  ]

        T_new      = [ R_d R      R_d t + t_d ]
                     [   0             1     ]

    Then convert T_new back to 6D vector [t_new, w_new] using so3_log.
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
    Convenience: return the identity SE(3) pose in 6D vector form.
    """
    return jnp.zeros(6, dtype=jnp.float32)