#!/usr/bin/env python3
"""Diagnose auto-weight scaling across noise levels.

Reads seed JSON files from exp36 runs at different noise levels and reports:
  1. Estimated noise model parameters (σ_noise, σ_process) per sequence
  2. Derived auto-weights (sw, rw) and their ratios
  3. How the outer loss term magnitudes compare (anchor vs reg vs smoothness)
  4. Correlation between weight ratios and improvement

This helps explain why performance is non-monotonic across noise levels
(e.g., σ_t=0.05 outperforms σ_t=0.10).

Usage:
  python -m experiments.diag_autoweight_scaling \
    --run-dirs /data/tkocher/exp_res/exp36_st0.01 \
               /data/tkocher/exp_res/exp36_st0.03 \
               /data/tkocher/exp_res/exp36_st0.05 \
               /data/tkocher/exp_res/exp36_st0.10
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_first_seed(run_dir: str) -> dict:
    """Load the first seed JSON from a run directory."""
    run_path = Path(run_dir)
    seed_files = sorted(run_path.glob("seed_*.json"))
    if not seed_files:
        raise FileNotFoundError(f"No seed_*.json files in {run_dir}")
    with open(seed_files[0]) as f:
        return json.load(f)


def extract_diagnostics(data: dict) -> list[dict]:
    """Extract noise model and weight info per sequence from a seed file."""
    config = data.get("config", {})
    sigma_trans = config.get("sigma_trans", "?")
    sigma_rot = config.get("sigma_rot", "?")

    results = []
    for seq in data["sequences"]:
        nm = seq["noise_model"]
        wu = seq["weights_used"]
        imp = seq["improvement_pct"]

        # Recompute what auto-weights would be (to verify).
        sn_t = nm["sigma_noise_trans"]
        sn_r = nm["sigma_noise_rot"]
        sp_t = nm["sigma_process_trans"]
        sp_r = nm["sigma_process_rot"]

        sw_t_expected = 1.0 / max(sp_t ** 2, 1e-12)
        sw_r_expected = 1.0 / max(sp_r ** 2, 1e-12)
        rw_t_expected = 1.0 / max(sn_t ** 2, 1e-12)
        rw_r_expected = 1.0 / max(sn_r ** 2, 1e-12)

        results.append({
            "sequence": seq["sequence"],
            "n_poses": seq["n_poses"],
            "sigma_trans_injected": sigma_trans,
            "sigma_rot_injected": sigma_rot,
            # Estimated noise model
            "sigma_noise_trans": sn_t,
            "sigma_noise_rot": sn_r,
            "sigma_process_trans": sp_t,
            "sigma_process_rot": sp_r,
            # Noise estimation accuracy
            "noise_est_ratio_trans": sn_t / sigma_trans if sigma_trans else None,
            "noise_est_ratio_rot": sn_r / sigma_rot if sigma_rot else None,
            # Actual weights used
            "sw_trans": wu["sw_trans"],
            "sw_rot": wu["sw_rot"],
            "rw_trans": wu["rw_trans"],
            "rw_rot": wu["rw_rot"],
            "snr_trans": wu.get("snr_trans"),
            "snr_rot": wu.get("snr_rot"),
            # Expected weights (verification)
            "sw_trans_expected": sw_t_expected,
            "rw_trans_expected": rw_t_expected,
            # Weight ratios — key diagnostic
            "sw_rw_ratio_trans": wu["sw_trans"] / max(wu["rw_trans"], 1e-12),
            "sw_rw_ratio_rot": wu["sw_rot"] / max(wu["rw_rot"], 1e-12),
            # Performance
            "delta_t": imp["trans_rmse"],
            "delta_r": imp["rot_rmse"],
            "delta_c": imp["combined"],
        })

    return results


def print_section(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose auto-weight scaling across noise levels")
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Exp36 run directories (one per noise level)")
    args = parser.parse_args()

    # Load all runs.
    all_runs = []
    for rd in args.run_dirs:
        data = load_first_seed(rd)
        diag = extract_diagnostics(data)
        sigma_t = data["config"]["sigma_trans"]
        all_runs.append((sigma_t, diag))

    all_runs.sort(key=lambda x: x[0])

    # ---------------------------------------------------------------
    # Section 1: Noise estimation accuracy
    # ---------------------------------------------------------------
    print_section("1. NOISE ESTIMATION ACCURACY")
    print(f"\n  Does the online EMA estimator correctly recover the "
          f"injected σ?")
    print(f"  Ratio = estimated_σ_noise / injected_σ  (ideal = 1.0)\n")

    header = (f"  {'σ_t':>6s}  {'Seq':>4s}  "
              f"{'σ̂_n_t':>10s}  {'ratio_t':>8s}  "
              f"{'σ̂_n_r':>10s}  {'ratio_r':>8s}  "
              f"{'σ̂_p_t':>10s}  {'σ̂_p_r':>10s}")
    print(header)
    print("  " + "-" * 76)

    for sigma_t, diag in all_runs:
        for d in diag:
            print(f"  {sigma_t:>6.3f}  {d['sequence']:>4s}  "
                  f"{d['sigma_noise_trans']:>10.6f}  "
                  f"{d['noise_est_ratio_trans']:>8.3f}  "
                  f"{d['sigma_noise_rot']:>10.6f}  "
                  f"{d['noise_est_ratio_rot']:>8.3f}  "
                  f"{d['sigma_process_trans']:>10.6f}  "
                  f"{d['sigma_process_rot']:>10.6f}")

        # Summary for this noise level.
        ratios_t = [d["noise_est_ratio_trans"] for d in diag]
        ratios_r = [d["noise_est_ratio_rot"] for d in diag]
        sp_t = [d["sigma_process_trans"] for d in diag]
        sp_r = [d["sigma_process_rot"] for d in diag]
        print(f"  {sigma_t:>6.3f}  {'AVG':>4s}  "
              f"{'':>10s}  {np.mean(ratios_t):>8.3f}  "
              f"{'':>10s}  {np.mean(ratios_r):>8.3f}  "
              f"{np.mean(sp_t):>10.6f}  {np.mean(sp_r):>10.6f}")
        print()

    # ---------------------------------------------------------------
    # Section 2: Auto-weight values across noise levels
    # ---------------------------------------------------------------
    print_section("2. AUTO-WEIGHT VALUES")
    print(f"\n  sw = 1/σ²_process (smoothness), rw = 1/σ²_noise (regularisation)")
    print(f"  sw/rw ratio controls smoothness-vs-fidelity trade-off\n")

    header = (f"  {'σ_t':>6s}  {'Seq':>4s}  "
              f"{'sw_t':>12s}  {'rw_t':>12s}  {'sw/rw_t':>10s}  "
              f"{'sw_r':>12s}  {'rw_r':>12s}  {'sw/rw_r':>10s}")
    print(header)
    print("  " + "-" * 82)

    for sigma_t, diag in all_runs:
        for d in diag:
            print(f"  {sigma_t:>6.3f}  {d['sequence']:>4s}  "
                  f"{d['sw_trans']:>12.1f}  {d['rw_trans']:>12.1f}  "
                  f"{d['sw_rw_ratio_trans']:>10.2f}  "
                  f"{d['sw_rot']:>12.1f}  {d['rw_rot']:>12.1f}  "
                  f"{d['sw_rw_ratio_rot']:>10.2f}")

        # Summary.
        sw_t = [d["sw_trans"] for d in diag]
        rw_t = [d["rw_trans"] for d in diag]
        ratio_t = [d["sw_rw_ratio_trans"] for d in diag]
        sw_r = [d["sw_rot"] for d in diag]
        rw_r = [d["rw_rot"] for d in diag]
        ratio_r = [d["sw_rw_ratio_rot"] for d in diag]
        print(f"  {sigma_t:>6.3f}  {'AVG':>4s}  "
              f"{np.mean(sw_t):>12.1f}  {np.mean(rw_t):>12.1f}  "
              f"{np.mean(ratio_t):>10.2f}  "
              f"{np.mean(sw_r):>12.1f}  {np.mean(rw_r):>12.1f}  "
              f"{np.mean(ratio_r):>10.2f}")
        print()

    # ---------------------------------------------------------------
    # Section 3: Cross-noise-level summary table
    # ---------------------------------------------------------------
    print_section("3. CROSS-NOISE-LEVEL SUMMARY")
    print(f"\n  Mean values across all 22 sequences (first seed)\n")

    header = (f"  {'σ_t':>6s}  {'σ_r':>6s}  "
              f"{'σ̂_n_t':>8s}  {'σ̂_p_t':>8s}  "
              f"{'sw_t':>10s}  {'rw_t':>10s}  {'sw/rw_t':>8s}  "
              f"{'ΔT%':>7s}  {'ΔR%':>7s}  {'ΔC%':>7s}")
    print(header)
    print("  " + "-" * 90)

    for sigma_t, diag in all_runs:
        sigma_r = diag[0]["sigma_rot_injected"]
        sn_t = np.mean([d["sigma_noise_trans"] for d in diag])
        sp_t = np.mean([d["sigma_process_trans"] for d in diag])
        sw_t = np.mean([d["sw_trans"] for d in diag])
        rw_t = np.mean([d["rw_trans"] for d in diag])
        ratio = np.mean([d["sw_rw_ratio_trans"] for d in diag])
        dt = np.mean([d["delta_t"] for d in diag])
        dr = np.mean([d["delta_r"] for d in diag])
        dc = np.mean([d["delta_c"] for d in diag])

        print(f"  {sigma_t:>6.3f}  {sigma_r:>6.3f}  "
              f"{sn_t:>8.5f}  {sp_t:>8.5f}  "
              f"{sw_t:>10.1f}  {rw_t:>10.1f}  {ratio:>8.2f}  "
              f"{dt:>+6.1f}%  {dr:>+6.1f}%  {dc:>+6.1f}%")

    # ---------------------------------------------------------------
    # Section 4: Outer loss term magnitude estimates
    # ---------------------------------------------------------------
    print_section("4. OUTER LOSS TERM MAGNITUDES (ESTIMATED)")
    print(f"\n  Approximate magnitude of each outer loss term,")
    print(f"  assuming typical correction δ and smoothness Δ.\n")
    print(f"  anchor_loss = aw * ||pose_err||²  (aw fixed at 5.0)")
    print(f"  reg_loss    = rw * ||θ - noisy||²")
    print(f"  smooth_loss = sw * ||Δθ||²\n")

    # For each noise level, estimate what the loss terms look like.
    # Assume correction magnitude ≈ σ_noise (denoiser corrects ~1σ),
    # and smoothness diff magnitude ≈ σ_process.
    print(f"  {'σ_t':>6s}  "
          f"{'δ≈σ_n':>8s}  {'Δ≈σ_p':>8s}  "
          f"{'rw*δ²':>12s}  {'sw*Δ²':>12s}  "
          f"{'reg/smooth':>11s}  {'Diagnosis':>20s}")
    print("  " + "-" * 82)

    for sigma_t, diag in all_runs:
        sn_t = np.mean([d["sigma_noise_trans"] for d in diag])
        sp_t = np.mean([d["sigma_process_trans"] for d in diag])
        sw_t = np.mean([d["sw_trans"] for d in diag])
        rw_t = np.mean([d["rw_trans"] for d in diag])

        # Estimated loss magnitudes per edge.
        reg_mag = rw_t * sn_t ** 2   # rw * δ²  where δ ≈ σ_noise
        smooth_mag = sw_t * sp_t ** 2  # sw * Δ²  where Δ ≈ σ_process

        ratio = reg_mag / max(smooth_mag, 1e-12)

        if ratio > 10:
            diagnosis = "REG DOMINATES"
        elif ratio < 0.1:
            diagnosis = "SMOOTH DOMINATES"
        elif 0.5 < ratio < 2.0:
            diagnosis = "balanced"
        else:
            diagnosis = "moderate imbalance"

        print(f"  {sigma_t:>6.3f}  "
              f"{sn_t:>8.5f}  {sp_t:>8.5f}  "
              f"{reg_mag:>12.4f}  {smooth_mag:>12.4f}  "
              f"{ratio:>11.4f}  {diagnosis:>20s}")

    # ---------------------------------------------------------------
    # Section 5: Per-sequence correlation analysis
    # ---------------------------------------------------------------
    print_section("5. CORRELATION: sw/rw RATIO vs IMPROVEMENT")
    print(f"\n  Does higher sw/rw ratio (more smoothness) correlate with")
    print(f"  better or worse performance?\n")

    for sigma_t, diag in all_runs:
        ratios = np.array([d["sw_rw_ratio_trans"] for d in diag])
        delta_t = np.array([d["delta_t"] for d in diag])
        delta_r = np.array([d["delta_r"] for d in diag])
        delta_c = np.array([d["delta_c"] for d in diag])

        if np.std(ratios) > 1e-10:
            corr_t = np.corrcoef(ratios, delta_t)[0, 1]
            corr_r = np.corrcoef(ratios, delta_r)[0, 1]
            corr_c = np.corrcoef(ratios, delta_c)[0, 1]
            print(f"  σ_t = {sigma_t:.3f}: "
                  f"r(sw/rw, ΔT) = {corr_t:+.3f}, "
                  f"r(sw/rw, ΔR) = {corr_r:+.3f}, "
                  f"r(sw/rw, ΔC) = {corr_c:+.3f}")
        else:
            print(f"  σ_t = {sigma_t:.3f}: sw/rw ratio has zero variance "
                  f"across sequences (all identical)")

    # ---------------------------------------------------------------
    # Section 6: SNR analysis
    # ---------------------------------------------------------------
    print_section("6. SIGNAL-TO-NOISE RATIO")
    print(f"\n  SNR = σ_process / σ_noise")
    print(f"  Low SNR = noise dominates signal → harder to denoise\n")

    for sigma_t, diag in all_runs:
        snr_t = [d["snr_trans"] for d in diag if d["snr_trans"] is not None]
        snr_r = [d["snr_rot"] for d in diag if d["snr_rot"] is not None]

        if snr_t:
            print(f"  σ_t = {sigma_t:.3f}: "
                  f"SNR_trans = {np.mean(snr_t):.4f} ± {np.std(snr_t):.4f}, "
                  f"SNR_rot = {np.mean(snr_r):.4f} ± {np.std(snr_r):.4f}")
        else:
            print(f"  σ_t = {sigma_t:.3f}: SNR not available")

    # ---------------------------------------------------------------
    # Section 7: Diagnosis summary
    # ---------------------------------------------------------------
    print_section("7. DIAGNOSIS SUMMARY")
    print()

    for sigma_t, diag in all_runs:
        sw_t = np.mean([d["sw_trans"] for d in diag])
        rw_t = np.mean([d["rw_trans"] for d in diag])
        sp_t = np.mean([d["sigma_process_trans"] for d in diag])
        dt = np.mean([d["delta_t"] for d in diag])
        dc = np.mean([d["delta_c"] for d in diag])
        ratio = np.mean([d["sw_rw_ratio_trans"] for d in diag])

        print(f"  σ_t = {sigma_t:.3f} (ΔT = {dt:+.1f}%, ΔC = {dc:+.1f}%):")

        if dt < 0:
            print(f"    PROBLEM: Translation denoising is harmful.")
            if rw_t > 1000:
                print(f"    → rw_trans = {rw_t:.0f} is very high — "
                      f"regularisation pulls corrections back toward noise")
            if sp_t < 1e-4:
                print(f"    → σ_process_trans ≈ {sp_t:.2e} collapsed to "
                      f"near-zero → sw_trans = {sw_t:.0f} is extreme")
        elif dc < 30:
            print(f"    SUBOPTIMAL: Lower than expected improvement.")
            if ratio > 100:
                print(f"    → sw/rw = {ratio:.0f} — smoothness overwhelms "
                      f"regularisation, capping correction magnitude")
            if sp_t < 1e-4:
                print(f"    → σ_process collapsed → smoothness term too "
                      f"aggressive for this noise level")
        else:
            print(f"    OK: Weights are well-balanced for this noise level.")

        print()

    print("=" * 80)
    print("  RECOMMENDATION")
    print("=" * 80)
    print()
    print("  If sw/rw ratio varies wildly across noise levels, consider:")
    print("    1. Capping sw/rw ratio (e.g., max 100)")
    print("    2. Scaling base_sw as f(σ_noise)")
    print("    3. Using batch MAD instead of online EMA (less σ_process collapse)")
    print("    4. Separate tuning of (base_sw, base_rw) per noise regime")
    print()


if __name__ == "__main__":
    main()
