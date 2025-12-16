"""
dsg-jit/tests/test_se3_manifold_gn.py

Regression test for manifold Gauss-Newton correctness under the new
WorldModel residual architecture.

This test compares:
  (1) a non-jitted Gauss–Newton solve (reference)
  (2) a JittedGNManifold solve built once and reused

Both solves operate on the same *active template* (fixed-capacity) pose-chain
problem so JAX compilation happens once and subsequent calls re-use the same
compiled executable.

You can run this file directly:

    python3 dsg-jit/tests/test_se3_manifold_gn.py
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, List

import jax.numpy as jnp
import numpy as np

from dsg_jit.world.model import WorldModel, ActiveWindowTemplate
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import GNConfig, gauss_newton_manifold
from dsg_jit.optimization.jit_wrappers import JittedGNManifold


def _register_required_residuals(wm: WorldModel) -> None:
    """Register residuals used by this test."""
    from dsg_jit.slam.measurements import prior_residual, odom_se3_residual

    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)


def _make_pose_chain_problem(
    num_poses: int = 10,
) -> Tuple[WorldModel, Dict[int, jnp.ndarray], Dict[int, int]]:
    """Build a small SE(3) chain in *active template* form.

    Returns:
        wm: configured WorldModel with active template initialized
        gt: dict (key->se3) ground truth
        key_to_nid: dict (key->node id)
    """
    assert 5 <= num_poses <= 20

    wm = WorldModel()
    _register_required_residuals(wm)

    POSE_DIM = 6
    W = num_poses

    # Variable slots: exactly one pose per slot.
    variable_slots = [("pose_se3", i, POSE_DIM) for i in range(W)]

    # Factor slots:
    #   - one prior on slot 0
    #   - odometry chain factors between consecutive slots (i-1 -> i)
    factor_slots: List[Tuple[str, int, Tuple[Tuple[str, int], ...]]] = []
    factor_slots.append(("prior", 0, (("pose_se3", 0),)))

    # Reserve odometry slots 1..W-1 mapping factor k -> (k-1, k)
    for k in range(1, W):
        factor_slots.append(("odom_se3", k, (("pose_se3", k - 1), ("pose_se3", k))))

    template = ActiveWindowTemplate(variable_slots=variable_slots, factor_slots=factor_slots)
    wm.init_active_template(template)

    # Ground truth chain along x with small sinusoidal y.
    gt: Dict[int, jnp.ndarray] = {}
    key_to_nid: Dict[int, int] = {}

    step_dx = 1.0
    for k in range(W):
        x = step_dx * k
        y = 0.25 * math.sin(0.2 * k)
        z = 0.0
        yaw = 0.0
        gt_k = jnp.array([x, y, z, 0.0, 0.0, yaw], dtype=jnp.float32)
        gt[k] = gt_k

        # Initialize each pose with a small perturbation.
        noise = 0.01 * np.random.randn(6).astype(np.float32)
        init_k = gt_k + jnp.array(noise, dtype=jnp.float32)

        nid = int(wm.set_variable_slot("pose_se3", k, init_k))
        key_to_nid[k] = nid

    # Prior on the first pose.
    wm.configure_factor_slot(
        factor_type="prior",
        slot_idx=0,
        var_ids=(key_to_nid[0],),
        params={"target": gt[0], "weight": 1.0},
        active=True,
    )

    # Odometry factors for the chain.
    for k in range(1, W):
        meas = gt[k] - gt[k - 1]
        wm.configure_factor_slot(
            factor_type="odom_se3",
            slot_idx=k,
            var_ids=(key_to_nid[k - 1], key_to_nid[k]),
            params={"measurement": meas, "weight": 1.0},
            active=True,
        )

    return wm, gt, key_to_nid


def _rms_translation_error(
    x: jnp.ndarray,
    gt: Dict[int, jnp.ndarray],
    key_to_nid: Dict[int, int],
    block_slices: Dict[int, slice],
) -> float:
    """Compute RMS translation (xyz) error using solver block slices.

    We intentionally use `block_slices` (nid -> slice) rather than the
    pack_state index map, because the active-template mode guarantees the
    solver metadata is the authoritative slicing used during optimization.
    """
    err2 = 0.0
    count = 0
    for k, nid in key_to_nid.items():
        sl = block_slices[nid]
        est = np.array(x[sl])
        diff = est[:3] - np.array(gt[k][:3])
        err2 += float(diff @ diff)
        count += 3
    return math.sqrt(err2 / max(1, count))


def test_active_template_jitted_matches_nonjit() -> None:
    """JittedGNManifold on an active template should match the non-jitted solve."""
    wm, gt, key_to_nid = _make_pose_chain_problem(num_poses=10)

    # Build residual once (use_jit=True ensures residual eval is compiled).
    residual_fn = wm.build_residual()

    x0, index = wm.pack_state()
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=(x0, index), fg=wm.fg
    )

    cfg = GNConfig(max_iters=10, damping=1e-6, max_step_norm=1.0)

    # Reference: non-jitted GN (solver path, not residual jit-wrapper).
    def _res(x: jnp.ndarray) -> jnp.ndarray:
        return residual_fn(x)

    x_ref = gauss_newton_manifold(residual_fn=_res, x0=x0, manifold_types=manifold_types, block_slices=block_slices, cfg=cfg)

    # Jitted solver wrapper.
    jgn = JittedGNManifold.from_residual(
        residual_fn=residual_fn,
        manifold_types=manifold_types,
        block_slices=block_slices,
        cfg=cfg,
    )

    # Warmup compile (exclude from comparisons)
    _ = jgn(x0).block_until_ready()

    x_jit = jgn(x0)
    x_jit.block_until_ready()

    # They should agree very closely.
    np.testing.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-5, atol=1e-5)

    # And both should be near GT on translation.
    err_ref = _rms_translation_error(x_ref, gt, key_to_nid, block_slices)
    err_jit = _rms_translation_error(x_jit, gt, key_to_nid, block_slices)

    assert err_ref < 1e-2
    assert err_jit < 1e-2


def main() -> None:
    """Allow running this test file directly without pytest."""
    np.random.seed(0)
    try:
        test_active_template_jitted_matches_nonjit()
    except Exception as e:
        raise SystemExit(f"FAILED: {e}")
    print("OK: test_active_template_jitted_matches_nonjit")


if __name__ == "__main__":
    main()