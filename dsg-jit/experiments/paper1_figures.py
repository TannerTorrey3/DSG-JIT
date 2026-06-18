#!/usr/bin/env python3
"""Paper 1 figures and statistical analysis.

Generates:
  1. Violin + strip plots of per-sequence improvement distributions
  2. Hypothesis tests (per-noise-level, per-sequence, cross-noise-level)
  3. LaTeX-ready results tables

Usage:
  python -m experiments.paper1_figures \
    --run-dirs /data/tkocher/exp_res/exp36_st0.01 \
               /data/tkocher/exp_res/exp36_st0.03 \
               /data/tkocher/exp_res/exp36_st0.05 \
               /data/tkocher/exp_res/exp36_st0.10 \
    --output-dir /data/tkocher/paper1_figures

Each --run-dirs entry should be an exp36 run directory containing
seed_XXXX.json files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> dict:
    """Load all seed results from an exp36 run directory.

    Returns:
        {
            "config": {...},
            "sigma_trans": float,
            "sigma_rot": float,
            "sequences": {
                "00": {"n_poses": int, "seeds": {
                    42: {"trans": float, "rot": float, "combined": float},
                    ...
                }},
                ...
            }
        }
    """
    run_path = Path(run_dir)
    seed_files = sorted(run_path.glob("seed_*.json"))
    if not seed_files:
        raise FileNotFoundError(f"No seed_*.json files in {run_dir}")

    config = None
    sequences = {}

    for sf in seed_files:
        with open(sf) as f:
            data = json.load(f)

        if config is None:
            config = data.get("config", {})

        seed = data["seed"]
        for seq_result in data["sequences"]:
            seq_id = seq_result["sequence"]
            if seq_id not in sequences:
                sequences[seq_id] = {
                    "n_poses": seq_result["n_poses"],
                    "seeds": {},
                }
            seed_entry = {
                "trans": seq_result["improvement_pct"]["trans_rmse"],
                "rot": seq_result["improvement_pct"]["rot_rmse"],
                "combined": seq_result["improvement_pct"]["combined"],
            }
            if "throughput" in seq_result:
                seed_entry["throughput"] = seq_result["throughput"]
            sequences[seq_id]["seeds"][seed] = seed_entry

    sigma_trans = config.get("sigma_trans", None)
    sigma_rot = config.get("sigma_rot", None)

    return {
        "config": config,
        "sigma_trans": sigma_trans,
        "sigma_rot": sigma_rot,
        "sequences": sequences,
    }


def load_all_runs(run_dirs: list[str]) -> list[dict]:
    """Load and sort runs by sigma_trans."""
    runs = [load_run(d) for d in run_dirs]
    runs.sort(key=lambda r: r["sigma_trans"])
    return runs


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def get_all_values(run: dict, metric: str = "combined") -> np.ndarray:
    """Get all seed values for a given metric, flattened across sequences."""
    vals = []
    for seq in run["sequences"].values():
        for seed_data in seq["seeds"].values():
            vals.append(seed_data[metric])
    return np.array(vals)


def get_seq_values(run: dict, seq_id: str, metric: str = "combined"
                   ) -> np.ndarray:
    """Get all seed values for one sequence."""
    seeds = run["sequences"][seq_id]["seeds"]
    return np.array([seeds[s][metric] for s in sorted(seeds)])


def get_seq_means(run: dict, metric: str = "combined") -> dict:
    """Get {seq_id: mean_value} for a given metric."""
    return {
        seq_id: np.mean(get_seq_values(run, seq_id, metric))
        for seq_id in run["sequences"]
    }


# ---------------------------------------------------------------------------
# Figure 1: Violin + strip plot
# ---------------------------------------------------------------------------

def plot_violin_strip(runs: list[dict], output_path: str,
                      metric: str = "combined",
                      metric_label: str = r"$\Delta C$ (%)",
                      figsize: tuple = (7.16, 2.0)):
    """Create violin + strip plots faceted by noise level."""
    n_levels = len(runs)
    seq_ids = sorted(runs[0]["sequences"].keys())
    n_seqs = len(seq_ids)

    fig, axes = plt.subplots(n_levels, 1, figsize=(figsize[0],
                             figsize[1] * n_levels),
                             sharex=True, sharey=False)
    if n_levels == 1:
        axes = [axes]

    # Color by n_poses (proxy for sequence length/complexity).
    n_poses_list = [runs[0]["sequences"][s]["n_poses"] for s in seq_ids]
    n_poses_arr = np.array(n_poses_list)
    norm = plt.Normalize(n_poses_arr.min(), n_poses_arr.max())
    cmap = plt.cm.viridis

    for ax_i, (ax, run) in enumerate(zip(axes, runs)):
        sigma_t = run["sigma_trans"]
        sigma_r = run["sigma_rot"]

        # Collect data per sequence.
        all_data = []
        positions = []
        for si, seq_id in enumerate(seq_ids):
            vals = get_seq_values(run, seq_id, metric)
            all_data.append(vals)
            positions.append(si)

        # Violin plot.
        vp = ax.violinplot(all_data, positions=positions, showmedians=True,
                           showextrema=False, widths=0.7)

        # Color violins by sequence length.
        for si, body in enumerate(vp["bodies"]):
            color = cmap(norm(n_poses_list[si]))
            body.set_facecolor(color)
            body.set_alpha(0.5)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.0)

        # Strip (jittered scatter).
        rng = np.random.RandomState(42)
        for si, seq_id in enumerate(seq_ids):
            vals = get_seq_values(run, seq_id, metric)
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            color = cmap(norm(n_poses_list[si]))
            ax.scatter(si + jitter, vals, s=8, alpha=0.4, color=color,
                       edgecolors="none", zorder=3)

        # Zero line.
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        # Overall mean line.
        all_vals = get_all_values(run, metric)
        ax.axhline(y=np.mean(all_vals), color="blue", linestyle=":",
                   linewidth=0.8, alpha=0.6)

        ax.set_ylabel(metric_label, fontsize=8)
        ax.set_title(
            rf"$\sigma_t = {sigma_t}$, $\sigma_r = {sigma_r}$"
            f"  (mean = {np.mean(all_vals):+.1f}%)",
            fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    # X-axis labels on bottom panel only.
    axes[-1].set_xticks(range(n_seqs))
    axes[-1].set_xticklabels(seq_ids, fontsize=7)
    axes[-1].set_xlabel("KITTI Sequence", fontsize=8)

    # Colorbar for sequence length.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Sequence length (poses)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Per-Sequence Improvement Distributions Across Noise Seeds",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Translation vs Rotation scatter
# ---------------------------------------------------------------------------

def plot_trans_vs_rot(runs: list[dict], output_path: str,
                     figsize: tuple = (3.5, 2.8)):
    """Scatter of mean ΔT% vs mean ΔR% per sequence, colored by noise level."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(runs)))
    seq_ids = sorted(runs[0]["sequences"].keys())

    for ri, run in enumerate(runs):
        sigma_t = run["sigma_trans"]
        t_means = [np.mean(get_seq_values(run, s, "trans")) for s in seq_ids]
        r_means = [np.mean(get_seq_values(run, s, "rot")) for s in seq_ids]
        ax.scatter(t_means, r_means, c=[colors[ri]], s=40, alpha=0.7,
                   label=rf"$\sigma_t = {sigma_t}$", edgecolors="white",
                   linewidths=0.5)

    # Diagonal reference.
    lims = [ax.get_xlim(), ax.get_ylim()]
    lo = min(lims[0][0], lims[1][0])
    hi = max(lims[0][1], lims[1][1])
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel(r"$\Delta T$ (%) — Translation RMSE Improvement", fontsize=8)
    ax.set_ylabel(r"$\Delta R$ (%) — Rotation RMSE Improvement", fontsize=8)
    ax.set_title("Translation vs Rotation Improvement per Sequence",
                 fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Aggregate improvement vs noise level
# ---------------------------------------------------------------------------

def plot_noise_level_trend(runs: list[dict], output_path: str,
                           figsize: tuple = (3.5, 2.5)):
    """Bar chart: ΔC% (pose-weighted and unweighted) vs noise level."""
    fig, ax = plt.subplots(figsize=figsize)

    sigmas = []
    uw_means, uw_stds = [], []
    pw_means, pw_stds = [], []

    for run in runs:
        sigmas.append(run["sigma_trans"])
        seq_ids = sorted(run["sequences"].keys())

        # Unweighted: mean of per-sequence means.
        seq_means = np.array([
            np.mean(get_seq_values(run, s, "combined")) for s in seq_ids])
        uw_means.append(np.mean(seq_means))
        uw_stds.append(np.std(seq_means))

        # Pose-weighted: weight each sequence mean by n_poses.
        n_poses = np.array([run["sequences"][s]["n_poses"] for s in seq_ids])
        weights = n_poses / n_poses.sum()
        pw_means.append(np.sum(weights * seq_means))
        pw_stds.append(np.sqrt(np.sum(weights * (seq_means - pw_means[-1])**2)))

    x = np.arange(len(sigmas))
    width = 0.35

    bars1 = ax.bar(x - width/2, uw_means, width, yerr=uw_stds, capsize=4,
                   label="Unweighted mean", color="#4C72B0", alpha=0.8)
    bars2 = ax.bar(x + width/2, pw_means, width, yerr=pw_stds, capsize=4,
                   label="Pose-weighted mean", color="#DD8452", alpha=0.8)

    # Value labels on bars.
    for bar, val in zip(bars1, uw_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:+.1f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, pw_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:+.1f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\sigma_t = {s}$" for s in sigmas], fontsize=8)
    ax.set_ylabel(r"$\Delta C$ (%)", fontsize=8)
    ax.set_title("Aggregate Improvement Across Noise Levels", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Heatmap — sequences × noise levels
# ---------------------------------------------------------------------------

def plot_heatmap(runs: list[dict], output_path: str,
                 metric: str = "combined",
                 metric_label: str = r"$\Delta C$ (%)",
                 figsize: tuple = (7.16, 3.0)):
    """Heatmap of mean improvement: rows = sequences, columns = noise levels."""
    seq_ids = sorted(runs[0]["sequences"].keys())
    sigmas = [run["sigma_trans"] for run in runs]
    n_seqs = len(seq_ids)
    n_levels = len(runs)

    matrix = np.zeros((n_seqs, n_levels))
    for ri, run in enumerate(runs):
        for si, seq_id in enumerate(seq_ids):
            matrix[si, ri] = np.mean(get_seq_values(run, seq_id, metric))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.max(np.abs(matrix))
    im = ax.imshow(matrix.T, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(n_seqs))
    ax.set_xticklabels(seq_ids, fontsize=7)
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([rf"$\sigma_t={s}$" for s in sigmas], fontsize=8)
    ax.set_xlabel("KITTI Sequence", fontsize=8)

    for ri in range(n_levels):
        for si in range(n_seqs):
            val = matrix[si, ri]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(si, ri, f"{val:.0f}", ha="center", va="center",
                    fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(metric_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"Per-Sequence Improvement Across Noise Levels",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: CDF — cumulative distribution of improvements
# ---------------------------------------------------------------------------

def plot_cdf(runs: list[dict], output_path: str,
             metric: str = "combined",
             metric_label: str = r"$\Delta C$ (%)",
             figsize: tuple = (3.5, 2.5)):
    """Empirical CDF of improvement across all seq×seed pairs per noise level."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(runs)))

    for ri, run in enumerate(runs):
        sigma_t = run["sigma_trans"]
        all_vals = np.sort(get_all_values(run, metric))
        cdf = np.arange(1, len(all_vals) + 1) / len(all_vals)
        ax.plot(all_vals, cdf, color=colors[ri], linewidth=1.2,
                label=rf"$\sigma_t = {sigma_t}$")

    ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(metric_label, fontsize=8)
    ax.set_ylabel("Cumulative Proportion", fontsize=8)
    ax.set_title("Empirical CDF of Improvement", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Component trend — ΔT and ΔR vs noise level
# ---------------------------------------------------------------------------

def plot_component_trend(runs: list[dict], output_path: str,
                         figsize: tuple = (3.5, 2.5)):
    """Line plot of ΔT, ΔR, ΔC vs σ_t showing component divergence."""
    fig, ax = plt.subplots(figsize=figsize)

    sigmas = []
    t_means, t_stds = [], []
    r_means, r_stds = [], []
    c_means, c_stds = [], []

    for run in runs:
        sigmas.append(run["sigma_trans"])
        seq_ids = sorted(run["sequences"].keys())
        t_seq = np.array([np.mean(get_seq_values(run, s, "trans"))
                          for s in seq_ids])
        r_seq = np.array([np.mean(get_seq_values(run, s, "rot"))
                          for s in seq_ids])
        c_seq = np.array([np.mean(get_seq_values(run, s, "combined"))
                          for s in seq_ids])
        t_means.append(np.mean(t_seq)); t_stds.append(np.std(t_seq))
        r_means.append(np.mean(r_seq)); r_stds.append(np.std(r_seq))
        c_means.append(np.mean(c_seq)); c_stds.append(np.std(c_seq))

    x = np.arange(len(sigmas))
    ax.errorbar(x, t_means, yerr=t_stds, marker="s", markersize=4,
                capsize=3, linewidth=1.2, label=r"$\Delta T$", color="#E24A33")
    ax.errorbar(x, r_means, yerr=r_stds, marker="^", markersize=4,
                capsize=3, linewidth=1.2, label=r"$\Delta R$", color="#348ABD")
    ax.errorbar(x, c_means, yerr=c_stds, marker="o", markersize=4,
                capsize=3, linewidth=1.2, label=r"$\Delta C$", color="#2CA02C")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\sigma_t={s}$" for s in sigmas], fontsize=7)
    ax.set_ylabel("Improvement (%)", fontsize=8)
    ax.set_title("Translation, Rotation, and Combined vs Noise Level",
                 fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Per-sequence grouped bars at one noise level
# ---------------------------------------------------------------------------

def plot_per_sequence_bars(runs: list[dict], output_path: str,
                           target_sigma: float = 0.03,
                           figsize: tuple = (7.16, 4.5)):
    """Two-panel horizontal bar chart with error bars: Trans+Rot (left), Combined (right)."""
    run = None
    for r in runs:
        if abs(r["sigma_trans"] - target_sigma) < 1e-6:
            run = r
            break
    if run is None:
        print(f"  Skipped per-sequence bars: no run at sigma_t={target_sigma}")
        return

    seq_ids = sorted(run["sequences"].keys())
    n_seqs = len(seq_ids)
    n_poses = [run["sequences"][s]["n_poses"] for s in seq_ids]

    t_means = [np.mean(get_seq_values(run, s, "trans")) for s in seq_ids]
    r_means = [np.mean(get_seq_values(run, s, "rot")) for s in seq_ids]
    c_means = [np.mean(get_seq_values(run, s, "combined")) for s in seq_ids]
    t_stds = [np.std(get_seq_values(run, s, "trans")) for s in seq_ids]
    r_stds = [np.std(get_seq_values(run, s, "rot")) for s in seq_ids]
    c_stds = [np.std(get_seq_values(run, s, "combined")) for s in seq_ids]

    y = np.arange(n_seqs)
    h = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True,
                                    gridspec_kw={"width_ratios": [1.2, 1],
                                                 "wspace": 0.08})

    ax1.barh(y + h / 2, t_means, h, xerr=t_stds, capsize=1.5,
             label=r"$\Delta T$", color="#E24A33", alpha=0.85,
             error_kw={"linewidth": 0.6, "ecolor": "#888"})
    ax1.barh(y - h / 2, r_means, h, xerr=r_stds, capsize=1.5,
             label=r"$\Delta R$", color="#348ABD", alpha=0.85,
             error_kw={"linewidth": 0.6, "ecolor": "#888"})
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_yticks(y)
    labels = [f"{s}  ({n_poses[i]})" for i, s in enumerate(seq_ids)]
    ax1.set_yticklabels(labels, fontsize=6.5, family="monospace")
    ax1.set_xlabel("Improvement (%)", fontsize=8)
    ax1.set_ylabel("Sequence (poses)", fontsize=8)
    ax1.set_title("Translation and Rotation", fontsize=9)
    ax1.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax1.tick_params(labelsize=7)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    ax2.barh(y, c_means, h * 1.4, xerr=c_stds, capsize=1.5,
             color="#2CA02C", alpha=0.85,
             error_kw={"linewidth": 0.6, "ecolor": "#888"})
    for i, (m, s) in enumerate(zip(c_means, c_stds)):
        ax2.text(m + s + 0.8, i, f"{m:.1f}", va="center", fontsize=5.5,
                 color="#333", fontweight="bold")
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
    ax2.set_xlabel("Improvement (%)", fontsize=8)
    ax2.set_title(r"Combined ($\Delta C$)", fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.grid(axis="x", alpha=0.3)

    pw_mean = np.average(c_means, weights=n_poses)
    ax2.axvline(x=pw_mean, color="#2CA02C", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.text(pw_mean + 0.5, n_seqs - 0.5,
             rf"$\bar{{\mu}}_w$={pw_mean:.1f}%", fontsize=6.5,
             color="#2CA02C", fontweight="bold")

    fig.suptitle(rf"Per-Sequence RMSE Improvement at $\sigma_t = {target_sigma}$"
                 rf", $\sigma_r = {run['sigma_rot']}$"
                 f"  (mean $\\pm$ std over 20 seeds)",
                 fontsize=9, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Variance reduction — std across seeds per sequence
# ---------------------------------------------------------------------------

def plot_variance_by_noise(runs: list[dict], output_path: str,
                           figsize: tuple = (3.5, 2.5)):
    """Show how result variance (std across seeds) changes with noise level."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(runs)))

    seq_ids = sorted(runs[0]["sequences"].keys())
    n_seqs = len(seq_ids)
    x = np.arange(n_seqs)
    width = 0.8 / len(runs)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(runs))

    for ri, run in enumerate(runs):
        sigma_t = run["sigma_trans"]
        stds = [np.std(get_seq_values(run, s, "combined")) for s in seq_ids]
        ax.bar(x + offsets[ri], stds, width, label=rf"$\sigma_t = {sigma_t}$",
               color=colors[ri], alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, fontsize=5, rotation=45, ha="right")
    ax.set_xlabel("KITTI Sequence", fontsize=8)
    ax.set_ylabel(r"Std of $\Delta C$ across seeds (%)", fontsize=8)
    ax.set_title("Result Variance by Noise Level", fontsize=9)
    ax.legend(fontsize=6, framealpha=0.9, ncol=2, loc="upper right")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Throughput — poses/sec per sequence
# ---------------------------------------------------------------------------

def _get_throughput_values(run: dict, seq_id: str, key: str) -> np.ndarray:
    """Extract throughput values across seeds for one sequence."""
    seeds = run["sequences"][seq_id]["seeds"]
    vals = []
    for s in sorted(seeds):
        tp = seeds[s].get("throughput")
        if tp and key in tp:
            vals.append(tp[key])
    return np.array(vals) if vals else np.array([])


def plot_throughput_by_sequence(runs: list[dict], output_path: str,
                                target_sigma: float = 0.03,
                                figsize: tuple = (7.16, 2.8)):
    """Bar chart of poses/sec per sequence with real-time reference lines."""
    run = None
    for r in runs:
        if abs(r["sigma_trans"] - target_sigma) < 1e-6:
            run = r
            break
    if run is None:
        print(f"  Skipped throughput plot: no run at sigma_t={target_sigma}")
        return

    seq_ids = sorted(run["sequences"].keys())
    means = []
    stds = []
    for s in seq_ids:
        vals = _get_throughput_values(run, s, "poses_per_sec")
        if len(vals) == 0:
            print(f"  Skipped throughput plot: no throughput data")
            return
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    x = np.arange(len(seq_ids))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, means, yerr=stds, capsize=2, color="#348ABD", alpha=0.85,
           error_kw={"linewidth": 0.6, "ecolor": "#888"}, edgecolor="white",
           linewidth=0.3)

    ax.axhline(y=10, color="#E24A33", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(len(seq_ids) - 0.5, 10 + 1, "10 Hz real-time", fontsize=6.5,
            color="#E24A33", ha="right", fontweight="bold")
    ax.axhline(y=100, color="#E24A33", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(len(seq_ids) - 0.5, 100 + 2, "100 Hz", fontsize=6, color="#E24A33",
            ha="right", alpha=0.7)

    overall_mean = np.mean(means)
    ax.axhline(y=overall_mean, color="#2CA02C", linestyle="-", linewidth=0.8,
               alpha=0.6)
    ax.text(0.5, overall_mean + 1,
            f"Mean: {overall_mean:.0f} poses/s ({overall_mean/10:.1f}x real-time)",
            fontsize=6.5, color="#2CA02C", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, fontsize=7)
    ax.set_xlabel("KITTI Sequence", fontsize=8)
    ax.set_ylabel("Poses / sec", fontsize=8)
    ax.set_title(rf"Throughput at $\sigma_t = {target_sigma}$", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_pose_latency(runs: list[dict], output_path: str,
                      target_sigma: float = 0.03,
                      figsize: tuple = (3.5, 2.5)):
    """Per-pose latency (ms) = 1000 / poses_per_sec, with real-time budget line."""
    run = None
    for r in runs:
        if abs(r["sigma_trans"] - target_sigma) < 1e-6:
            run = r
            break
    if run is None:
        print(f"  Skipped pose latency plot: no run at sigma_t={target_sigma}")
        return

    seq_ids = sorted(run["sequences"].keys())
    latency_means = []
    latency_stds = []
    for s in seq_ids:
        pps = _get_throughput_values(run, s, "poses_per_sec")
        if len(pps) == 0:
            print(f"  Skipped pose latency plot: no throughput data")
            return
        per_pose_ms = 1000.0 / pps
        latency_means.append(np.mean(per_pose_ms))
        latency_stds.append(np.std(per_pose_ms))

    x = np.arange(len(seq_ids))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, latency_means, yerr=latency_stds, capsize=2,
           color="#348ABD", alpha=0.85,
           error_kw={"linewidth": 0.6, "ecolor": "#888"},
           edgecolor="white", linewidth=0.3)

    budget_10hz = 100.0
    ax.axhline(y=budget_10hz, color="#E24A33", linestyle="--", linewidth=1.0,
               alpha=0.8)
    ax.text(len(seq_ids) - 0.5, budget_10hz + 2,
            "100 ms (10 Hz budget)", fontsize=6.5,
            color="#E24A33", ha="right", fontweight="bold")

    overall_mean = np.mean(latency_means)
    ax.axhline(y=overall_mean, color="#2CA02C", linestyle="-", linewidth=0.8,
               alpha=0.6)
    ax.text(0.5, overall_mean + 1,
            f"Mean: {overall_mean:.1f} ms/pose",
            fontsize=6.5, color="#2CA02C", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, fontsize=7)
    ax.set_xlabel("KITTI Sequence", fontsize=8)
    ax.set_ylabel("Latency per pose (ms)", fontsize=8)
    ax.set_title(rf"Per-Pose Denoising Latency at $\sigma_t = {target_sigma}$",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: Ablation -- two-phase vs three-phase (Claim 2)
# ---------------------------------------------------------------------------

def plot_phase_ablation(run_a: dict, run_b: dict, output_path: str,
                        label_a: str = "Two-phase (T+R)",
                        label_b: str = "Three-phase (T+R+T)",
                        figsize: tuple = (3.5, 2.8)):
    """Side-by-side comparison of two experiment configs at the same noise level."""
    seq_ids = sorted(set(run_a["sequences"].keys()) &
                     set(run_b["sequences"].keys()))

    metrics = ["trans", "rot", "combined"]
    labels = [r"$\Delta T$", r"$\Delta R$", r"$\Delta C$"]

    a_means = [np.mean([np.mean(get_seq_values(run_a, s, m)) for s in seq_ids])
               for m in metrics]
    b_means = [np.mean([np.mean(get_seq_values(run_b, s, m)) for s in seq_ids])
               for m in metrics]
    a_stds = [np.std([np.mean(get_seq_values(run_a, s, m)) for s in seq_ids])
              for m in metrics]
    b_stds = [np.std([np.mean(get_seq_values(run_b, s, m)) for s in seq_ids])
              for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, a_means, w, yerr=a_stds, capsize=3,
           label=label_a, color="#348ABD", alpha=0.85,
           error_kw={"linewidth": 0.8, "ecolor": "#555"})
    ax.bar(x + w / 2, b_means, w, yerr=b_stds, capsize=3,
           label=label_b, color="#2CA02C", alpha=0.85,
           error_kw={"linewidth": 0.8, "ecolor": "#555"})

    for i, (a, b) in enumerate(zip(a_means, b_means)):
        diff = b - a
        sign = "+" if diff >= 0 else ""
        ax.text(i, max(a, b) + max(a_stds[i], b_stds[i]) + 1.5,
                f"{sign}{diff:.1f}pp", ha="center", fontsize=6.5,
                fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Improvement (%)", fontsize=8)
    ax.set_title("Phase-Decoupled Optimization Ablation", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: Ablation -- auto vs manual weights (Claim 3)
# ---------------------------------------------------------------------------

def plot_weight_ablation(run_auto: dict, run_manual: dict, output_path: str,
                         figsize: tuple = (7.16, 2.8)):
    """Compare auto-tuned vs manual weights per sequence at one noise level."""
    seq_ids = sorted(set(run_auto["sequences"].keys()) &
                     set(run_manual["sequences"].keys()))

    auto_means = [np.mean(get_seq_values(run_auto, s, "combined"))
                  for s in seq_ids]
    manual_means = [np.mean(get_seq_values(run_manual, s, "combined"))
                    for s in seq_ids]
    auto_stds = [np.std(get_seq_values(run_auto, s, "combined"))
                 for s in seq_ids]
    manual_stds = [np.std(get_seq_values(run_manual, s, "combined"))
                   for s in seq_ids]

    x = np.arange(len(seq_ids))
    w = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, auto_means, w, yerr=auto_stds, capsize=2,
           label="Auto (MAP-derived)", color="#2CA02C", alpha=0.85,
           error_kw={"linewidth": 0.5, "ecolor": "#888"})
    ax.bar(x + w / 2, manual_means, w, yerr=manual_stds, capsize=2,
           label="Manual (fixed)", color="#E24A33", alpha=0.85,
           error_kw={"linewidth": 0.5, "ecolor": "#888"})

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, fontsize=7)
    ax.set_xlabel("KITTI Sequence", fontsize=8)
    ax.set_ylabel(r"$\Delta C$ (%)", fontsize=8)
    sigma_t = run_auto["sigma_trans"]
    ax.set_title(rf"Auto vs Manual Weights at $\sigma_t = {sigma_t}$",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    auto_pw = np.mean(auto_means)
    manual_pw = np.mean(manual_means)
    ax.text(0.02, 0.95,
            f"Auto mean: {auto_pw:.1f}%  |  Manual mean: {manual_pw:.1f}%",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="#ccc"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_weight_ablation_multi_noise(runs_auto: list, runs_manual: list,
                                      output_path: str,
                                      figsize: tuple = (3.5, 2.8)):
    """Aggregate auto vs manual comparison across noise levels."""
    auto_by_sigma = {}
    for run in runs_auto:
        st = run["sigma_trans"]
        seq_ids = sorted(run["sequences"].keys())
        vals = [np.mean(get_seq_values(run, s, "combined")) for s in seq_ids]
        auto_by_sigma[st] = (np.mean(vals), np.std(vals))

    manual_by_sigma = {}
    for run in runs_manual:
        st = run["sigma_trans"]
        seq_ids = sorted(run["sequences"].keys())
        vals = [np.mean(get_seq_values(run, s, "combined")) for s in seq_ids]
        manual_by_sigma[st] = (np.mean(vals), np.std(vals))

    sigmas = sorted(set(auto_by_sigma.keys()) & set(manual_by_sigma.keys()))
    if not sigmas:
        print("  Skipped multi-noise weight ablation: no matching noise levels")
        return

    x = np.arange(len(sigmas))
    w = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    auto_m = [auto_by_sigma[s][0] for s in sigmas]
    auto_s = [auto_by_sigma[s][1] for s in sigmas]
    manual_m = [manual_by_sigma[s][0] for s in sigmas]
    manual_s = [manual_by_sigma[s][1] for s in sigmas]

    ax.bar(x - w / 2, auto_m, w, yerr=auto_s, capsize=3,
           label="Auto (MAP)", color="#2CA02C", alpha=0.85,
           error_kw={"linewidth": 0.6, "ecolor": "#888"})
    ax.bar(x + w / 2, manual_m, w, yerr=manual_s, capsize=3,
           label="Manual", color="#E24A33", alpha=0.85,
           error_kw={"linewidth": 0.6, "ecolor": "#888"})

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\sigma_t={s}$" for s in sigmas], fontsize=8)
    ax.set_ylabel(r"$\Delta C$ (%)", fontsize=8)
    ax.set_title("Auto vs Manual Weights Across Noise Levels", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def hypothesis_tests(runs: list[dict], output_path: str):
    """Run all statistical tests and write results to a text file."""
    lines = []
    lines.append("=" * 72)
    lines.append("STATISTICAL ANALYSIS — Paper 1 Results")
    lines.append("=" * 72)

    seq_ids = sorted(runs[0]["sequences"].keys())
    n_seqs = len(seq_ids)

    # -------------------------------------------------------------------
    # Test 1: Per-noise-level — does the denoiser improve over raw?
    #   H₀: median ΔC% = 0  (Wilcoxon signed-rank, one-sample)
    # -------------------------------------------------------------------
    lines.append("\n" + "-" * 72)
    lines.append("TEST 1: Per-Noise-Level — H₀: median ΔC% = 0")
    lines.append("  Method: Wilcoxon signed-rank (one-sample)")
    lines.append("-" * 72)

    for run in runs:
        sigma_t = run["sigma_trans"]
        all_vals = get_all_values(run, "combined")
        stat, p = stats.wilcoxon(all_vals, alternative="greater")
        n = len(all_vals)
        # Effect size: rank-biserial correlation r = 1 - (2W-)/(n(n+1)/2)
        # W- is the negative rank sum. scipy returns W+ as stat.
        w_plus = stat
        w_total = n * (n + 1) / 2
        w_minus = w_total - w_plus
        r_rb = 1 - (2 * w_minus) / w_total
        median = np.median(all_vals)
        mean = np.mean(all_vals)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(
            f"\n  σ_t = {sigma_t}: n={n}, median={median:+.2f}%, "
            f"mean={mean:+.2f}%")
        lines.append(
            f"    W+ = {w_plus:.0f}, p = {p:.2e}, r_rb = {r_rb:.3f}  [{sig}]")

    # -------------------------------------------------------------------
    # Test 2: Per-sequence — which sequences benefit significantly?
    #   H₀: median ΔC% = 0 for each sequence (across seeds)
    #   Benjamini-Hochberg correction for 22 comparisons
    # -------------------------------------------------------------------
    lines.append("\n" + "-" * 72)
    lines.append("TEST 2: Per-Sequence Significance (BH-corrected)")
    lines.append("  Method: Wilcoxon signed-rank per sequence, "
                 "Benjamini-Hochberg correction")
    lines.append("-" * 72)

    for run in runs:
        sigma_t = run["sigma_trans"]
        lines.append(f"\n  --- σ_t = {sigma_t} ---")
        lines.append(f"  {'Seq':>4s}  {'N':>4s}  {'Median':>8s}  "
                     f"{'Mean':>8s}  {'p_raw':>10s}  {'p_BH':>10s}  "
                     f"{'r_rb':>6s}  {'Sig':>4s}")

        p_values = []
        seq_stats = []
        for seq_id in seq_ids:
            vals = get_seq_values(run, seq_id, "combined")
            n = len(vals)
            median = np.median(vals)
            mean = np.mean(vals)

            # Wilcoxon needs at least some non-zero differences.
            non_zero = vals[vals != 0]
            if len(non_zero) < 5:
                p_raw = 1.0
                r_rb = 0.0
                w_plus = 0
            else:
                try:
                    w_plus, p_raw = stats.wilcoxon(
                        vals, alternative="greater")
                    w_total = n * (n + 1) / 2
                    w_minus = w_total - w_plus
                    r_rb = 1 - (2 * w_minus) / w_total
                except ValueError:
                    p_raw = 1.0
                    r_rb = 0.0

            p_values.append(p_raw)
            seq_stats.append((seq_id, n, median, mean, p_raw, r_rb))

        # Benjamini-Hochberg correction.
        p_arr = np.array(p_values)
        n_tests = len(p_arr)
        sorted_idx = np.argsort(p_arr)
        p_bh = np.empty(n_tests)
        for rank_i, orig_i in enumerate(sorted_idx):
            p_bh[orig_i] = p_arr[orig_i] * n_tests / (rank_i + 1)
        # Enforce monotonicity.
        p_bh_sorted = p_bh[sorted_idx]
        for i in range(n_tests - 2, -1, -1):
            p_bh_sorted[i] = min(p_bh_sorted[i], p_bh_sorted[i + 1])
        for rank_i, orig_i in enumerate(sorted_idx):
            p_bh[orig_i] = min(p_bh_sorted[rank_i], 1.0)

        n_sig = 0
        for i, (seq_id, n, median, mean, p_raw, r_rb) in enumerate(seq_stats):
            sig = ("***" if p_bh[i] < 0.001 else
                   "**" if p_bh[i] < 0.01 else
                   "*" if p_bh[i] < 0.05 else "ns")
            if p_bh[i] < 0.05:
                n_sig += 1
            lines.append(
                f"  {seq_id:>4s}  {n:>4d}  {median:>+7.2f}%  "
                f"{mean:>+7.2f}%  {p_raw:>10.2e}  {p_bh[i]:>10.2e}  "
                f"{r_rb:>+5.3f}  {sig:>4s}")

        lines.append(
            f"\n  Significant: {n_sig}/{n_seqs} sequences (p_BH < 0.05)")

    # -------------------------------------------------------------------
    # Test 3: Cross-noise-level — does improvement increase with noise?
    #   Friedman test across noise levels, paired by sequence
    #   Nemenyi post-hoc for pairwise differences
    # -------------------------------------------------------------------
    lines.append("\n" + "-" * 72)
    lines.append("TEST 3: Cross-Noise-Level Monotonicity")
    lines.append("  Method: Friedman test (non-parametric repeated measures)")
    lines.append("-" * 72)

    if len(runs) >= 3:
        # Build matrix: rows = sequences, cols = noise levels.
        # Each cell = mean ΔC% across seeds for that sequence at that level.
        matrix = np.zeros((n_seqs, len(runs)))
        for ri, run in enumerate(runs):
            for si, seq_id in enumerate(seq_ids):
                matrix[si, ri] = np.mean(
                    get_seq_values(run, seq_id, "combined"))

        stat_f, p_f = stats.friedmanchisquare(*[matrix[:, i]
                                                 for i in range(len(runs))])
        lines.append(f"\n  Friedman χ² = {stat_f:.2f}, "
                     f"p = {p_f:.2e}  "
                     f"{'***' if p_f < 0.001 else '**' if p_f < 0.01 else '*' if p_f < 0.05 else 'ns'}")

        # Pairwise Wilcoxon signed-rank with BH correction.
        lines.append("\n  Pairwise comparisons (Wilcoxon, BH-corrected):")
        pairs = []
        pair_pvals = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                s1 = runs[i]["sigma_trans"]
                s2 = runs[j]["sigma_trans"]
                v1 = matrix[:, i]
                v2 = matrix[:, j]
                try:
                    _, p_pw = stats.wilcoxon(v2 - v1, alternative="greater")
                except ValueError:
                    p_pw = 1.0
                pairs.append((i, j, s1, s2))
                pair_pvals.append(p_pw)

        # BH correction on pairwise tests.
        p_pw_arr = np.array(pair_pvals)
        n_pw = len(p_pw_arr)
        sorted_pw = np.argsort(p_pw_arr)
        p_pw_bh = np.empty(n_pw)
        for rank_i, orig_i in enumerate(sorted_pw):
            p_pw_bh[orig_i] = p_pw_arr[orig_i] * n_pw / (rank_i + 1)
        p_pw_bh = np.minimum(p_pw_bh, 1.0)

        for pi, (i, j, s1, s2) in enumerate(pairs):
            diff = np.mean(matrix[:, j]) - np.mean(matrix[:, i])
            sig = ("***" if p_pw_bh[pi] < 0.001 else
                   "**" if p_pw_bh[pi] < 0.01 else
                   "*" if p_pw_bh[pi] < 0.05 else "ns")
            lines.append(
                f"    σ_t {s1} → {s2}: "
                f"Δmean = {diff:+.2f}%, "
                f"p_BH = {p_pw_bh[pi]:.2e}  [{sig}]")
    else:
        lines.append("\n  Skipped: need >= 3 noise levels for Friedman test.")

    # -------------------------------------------------------------------
    # Test 4: Per-noise-level paired t-test (parametric complement)
    # -------------------------------------------------------------------
    lines.append("\n" + "-" * 72)
    lines.append("TEST 4: Parametric Complement — One-Sample t-test")
    lines.append("  Method: One-sample t-test on ΔC% = 0 (all seq×seed pairs)")
    lines.append("-" * 72)

    for run in runs:
        sigma_t = run["sigma_trans"]
        all_vals = get_all_values(run, "combined")
        t_stat, p_val = stats.ttest_1samp(all_vals, 0, alternative="greater")
        d = np.mean(all_vals) / np.std(all_vals, ddof=1)  # Cohen's d
        sig = ("***" if p_val < 0.001 else
               "**" if p_val < 0.01 else
               "*" if p_val < 0.05 else "ns")
        lines.append(
            f"\n  σ_t = {sigma_t}: t({len(all_vals)-1}) = {t_stat:.2f}, "
            f"p = {p_val:.2e}, Cohen's d = {d:.2f}  [{sig}]")

    # -------------------------------------------------------------------
    # Summary for paper text
    # -------------------------------------------------------------------
    lines.append("\n" + "=" * 72)
    lines.append("PAPER-READY SUMMARY")
    lines.append("=" * 72)

    # Count significant sequences at each noise level.
    for run in runs:
        sigma_t = run["sigma_trans"]
        n_sig = 0
        for seq_id in seq_ids:
            vals = get_seq_values(run, seq_id, "combined")
            try:
                _, p = stats.wilcoxon(vals, alternative="greater")
            except ValueError:
                p = 1.0
            # Using raw p < 0.05 / n_seqs as Bonferroni-conservative count.
            if p < 0.05 / n_seqs:
                n_sig += 1
        lines.append(
            f"  σ_t = {sigma_t}: {n_sig}/{n_seqs} sequences significant "
            f"(Bonferroni p < {0.05/n_seqs:.4f})")

    result = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(result + "\n")
    print(f"  Saved: {output_path}")
    print()
    print(result)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(runs: list[dict], output_path: str):
    """Generate the main aggregate results table in LaTeX."""
    seq_ids = sorted(runs[0]["sequences"].keys())

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Aggregate measurement denoising performance "
                 r"across noise levels. "
                 r"$\Delta T$, $\Delta R$, $\Delta C$: percentage improvement "
                 r"in translation, rotation, and combined RMSE. "
                 r"Subscript \textit{pw} denotes pose-weighted means. "
                 r"All values are means $\pm$ std across 20 noise seeds.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\begin{tabular}{cc"
                 r"r@{\,\scriptsize$\pm$\,}l"
                 r"r@{\,\scriptsize$\pm$\,}l"
                 r"r@{\,\scriptsize$\pm$\,}l"
                 r"r@{\,\scriptsize$\pm$\,}l"
                 r"r@{\,\scriptsize$\pm$\,}l"
                 r"r@{\,\scriptsize$\pm$\,}l}")
    lines.append(r"\toprule")
    lines.append(r"$\sigma_t$ & $\sigma_r$ "
                 r"& \multicolumn{2}{c}{$\Delta T$\%} "
                 r"& \multicolumn{2}{c}{$\Delta R$\%} "
                 r"& \multicolumn{2}{c}{$\Delta C$\%} "
                 r"& \multicolumn{2}{c}{$\Delta T_\text{pw}$\%} "
                 r"& \multicolumn{2}{c}{$\Delta R_\text{pw}$\%} "
                 r"& \multicolumn{2}{c}{$\Delta C_\text{pw}$\%} \\")
    lines.append(r"\midrule")

    for run in runs:
        sigma_t = run["sigma_trans"]
        sigma_r = run["sigma_rot"]

        # Unweighted: mean and std of per-sequence means.
        t_means = np.array([np.mean(get_seq_values(run, s, "trans"))
                            for s in seq_ids])
        r_means = np.array([np.mean(get_seq_values(run, s, "rot"))
                            for s in seq_ids])
        c_means = np.array([np.mean(get_seq_values(run, s, "combined"))
                            for s in seq_ids])

        # Pose-weighted.
        n_poses = np.array([run["sequences"][s]["n_poses"] for s in seq_ids])
        w = n_poses / n_poses.sum()

        uw_t = (np.mean(t_means), np.std(t_means))
        uw_r = (np.mean(r_means), np.std(r_means))
        uw_c = (np.mean(c_means), np.std(c_means))
        pw_t = (np.sum(w * t_means), np.sqrt(np.sum(w * (t_means - np.sum(w * t_means))**2)))
        pw_r = (np.sum(w * r_means), np.sqrt(np.sum(w * (r_means - np.sum(w * r_means))**2)))
        pw_c = (np.sum(w * c_means), np.sqrt(np.sum(w * (c_means - np.sum(w * c_means))**2)))

        def fmt(val, std):
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.1f} & {std:.1f}"

        lines.append(
            f"{sigma_t} & {sigma_r} & "
            f"{fmt(*uw_t)} & {fmt(*uw_r)} & {fmt(*uw_c)} & "
            f"{fmt(*pw_t)} & {fmt(*pw_r)} & {fmt(*pw_c)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    result = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(result + "\n")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paper 1: figures and statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Exp36 run directories (one per noise level)")
    parser.add_argument("--output-dir", type=str, default="paper1_figures",
                        help="Output directory for figures and tables")
    parser.add_argument("--metric", type=str, default="combined",
                        choices=["trans", "rot", "combined"],
                        help="Primary metric for violin plots")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Figure output format")
    parser.add_argument("--phase-baseline-dirs", nargs="+", default=None,
                        help="Two-phase (exp40) run dirs for phase ablation")
    parser.add_argument("--manual-weight-dirs", nargs="+", default=None,
                        help="Manual-weight run dirs for weight ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {len(args.run_dirs)} runs...")
    runs = load_all_runs(args.run_dirs)

    for run in runs:
        n_seeds = sum(len(s["seeds"]) for s in run["sequences"].values())
        n_seqs = len(run["sequences"])
        total_seeds = n_seeds // n_seqs if n_seqs > 0 else 0
        print(f"  σ_t = {run['sigma_trans']}, σ_r = {run['sigma_rot']}: "
              f"{n_seqs} sequences × {total_seeds} seeds")

    ext = args.format

    # Figure 1: Violin + strip.
    metric_labels = {
        "combined": r"$\Delta C$ (%)",
        "trans": r"$\Delta T$ (%)",
        "rot": r"$\Delta R$ (%)",
    }
    print("\nGenerating violin + strip plot...")
    plot_violin_strip(
        runs,
        os.path.join(args.output_dir, f"fig_violin_combined.{ext}"),
        metric=args.metric,
        metric_label=metric_labels[args.metric])

    # Also generate T and R separately.
    for m in ["trans", "rot"]:
        plot_violin_strip(
            runs,
            os.path.join(args.output_dir, f"fig_violin_{m}.{ext}"),
            metric=m,
            metric_label=metric_labels[m])

    # Figure 2: Trans vs Rot scatter.
    print("Generating trans vs rot scatter...")
    plot_trans_vs_rot(
        runs,
        os.path.join(args.output_dir, f"fig_trans_vs_rot.{ext}"))

    # Figure 3: Noise level trend.
    print("Generating noise level trend...")
    plot_noise_level_trend(
        runs,
        os.path.join(args.output_dir, f"fig_noise_trend.{ext}"))

    # Figure 4: Heatmap (sequences × noise levels).
    print("Generating heatmap...")
    plot_heatmap(
        runs,
        os.path.join(args.output_dir, f"fig_heatmap.{ext}"))

    # Figure 5: CDF of combined improvement.
    print("Generating CDF...")
    plot_cdf(
        runs,
        os.path.join(args.output_dir, f"fig_cdf.{ext}"))

    # Figure 6: Component trend (ΔT, ΔR, ΔC vs σ_t).
    print("Generating component trend...")
    plot_component_trend(
        runs,
        os.path.join(args.output_dir, f"fig_component_trend.{ext}"))

    # Figure 7: Per-sequence bars at primary noise level.
    print("Generating per-sequence bars...")
    plot_per_sequence_bars(
        runs,
        os.path.join(args.output_dir, f"fig_per_sequence_bars.{ext}"))

    # Figure 8: Variance by noise level.
    print("Generating variance by noise...")
    plot_variance_by_noise(
        runs,
        os.path.join(args.output_dir, f"fig_variance_by_noise.{ext}"))

    # Figure 9: Throughput by sequence.
    print("Generating throughput plot...")
    plot_throughput_by_sequence(
        runs,
        os.path.join(args.output_dir, f"fig_throughput.{ext}"))

    # Figure 10: Per-pose latency.
    print("Generating per-pose latency plot...")
    plot_pose_latency(
        runs,
        os.path.join(args.output_dir, f"fig_latency.{ext}"))

    # Ablation: phase-decoupled (Claim 2).
    if args.phase_baseline_dirs:
        print("\nGenerating phase ablation figure...")
        baseline_runs = load_all_runs(args.phase_baseline_dirs)
        for bl_run in baseline_runs:
            match = [r for r in runs
                     if abs(r["sigma_trans"] - bl_run["sigma_trans"]) < 1e-6]
            if match:
                sigma_t = bl_run["sigma_trans"]
                plot_phase_ablation(
                    bl_run, match[0],
                    os.path.join(args.output_dir,
                                 f"fig_phase_ablation_{sigma_t}.{ext}"))

    # Ablation: auto vs manual weights (Claim 3).
    if args.manual_weight_dirs:
        print("\nGenerating weight ablation figures...")
        manual_runs = load_all_runs(args.manual_weight_dirs)
        for man_run in manual_runs:
            match = [r for r in runs
                     if abs(r["sigma_trans"] - man_run["sigma_trans"]) < 1e-6]
            if match:
                sigma_t = man_run["sigma_trans"]
                plot_weight_ablation(
                    match[0], man_run,
                    os.path.join(args.output_dir,
                                 f"fig_weight_ablation_{sigma_t}.{ext}"))
        if len(manual_runs) > 1 and len(runs) > 1:
            plot_weight_ablation_multi_noise(
                runs, manual_runs,
                os.path.join(args.output_dir,
                             f"fig_weight_ablation_aggregate.{ext}"))

    # Hypothesis tests.
    print("\nRunning hypothesis tests...")
    hypothesis_tests(
        runs,
        os.path.join(args.output_dir, "statistical_tests.txt"))

    # LaTeX table.
    print("\nGenerating LaTeX table...")
    generate_latex_table(
        runs,
        os.path.join(args.output_dir, "table_main_results.tex"))

    print(f"\nAll outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
