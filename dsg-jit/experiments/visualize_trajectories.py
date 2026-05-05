"""
3D trajectory visualizer for exp34 denoising results.

Usage:
    python -m experiments.visualize_trajectories /data/tkocher/exp_res/exp34_*.json
    python -m experiments.visualize_trajectories results.json --seq 00
    python -m experiments.visualize_trajectories results.json --seq 00,03,07
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_sequence(ax, seq_data: dict, title: str):
    """Plot GT, noisy, and denoised trajectories for one sequence on a 3D axis."""
    gt = np.array(seq_data["trajectories"]["gt"])
    noisy = np.array(seq_data["trajectories"]["noisy"])
    denoised = np.array(seq_data["trajectories"]["denoised"])

    # Poses are [tx, ty, tz, wx, wy, wz] — plot translation (x, y, z).
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
            color="#2196F3", linewidth=1.5, label="Ground Truth", alpha=0.9)
    ax.plot(noisy[:, 0], noisy[:, 1], noisy[:, 2],
            color="#F44336", linewidth=0.8, label="Noisy", alpha=0.5)
    ax.plot(denoised[:, 0], denoised[:, 1], denoised[:, 2],
            color="#4CAF50", linewidth=1.2, label="Denoised", alpha=0.8)

    # Mark start and end.
    ax.scatter(*gt[0, :3], color="black", s=50, marker="o", zorder=5)
    ax.scatter(*gt[-1, :3], color="black", s=50, marker="x", zorder=5)

    imp = seq_data["improvement_pct"]
    ax.set_title(f"{title}\n"
                 f"Trans {imp['trans_rmse']:+.1f}%  "
                 f"Rot {imp['rot_rmse']:+.1f}%  "
                 f"Combined {imp['combined']:+.1f}%",
                 fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8, loc="upper left")


def main():
    parser = argparse.ArgumentParser(
        description="3D trajectory visualizer for exp34 results")
    parser.add_argument("json_file", type=str,
                        help="Path to exp34 results JSON")
    parser.add_argument("--seq", type=str, default=None,
                        help="Comma-separated sequence IDs to plot (default: all)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    sequences = data["sequences"]

    # Filter sequences if requested.
    if args.seq:
        seq_ids = set(args.seq.split(","))
        sequences = [s for s in sequences if s["sequence"] in seq_ids]

    if not sequences:
        print("No matching sequences found.")
        sys.exit(1)

    # Filter to only sequences that have trajectory data.
    sequences = [s for s in sequences if "trajectories" in s]
    if not sequences:
        print("No trajectory data in this results file. "
              "Re-run exp34 to generate trajectory data.")
        sys.exit(1)

    n = len(sequences)
    if n == 1:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        seq = sequences[0]
        plot_sequence(ax, seq, f"Seq {seq['sequence']} ({seq['n_poses']} poses)")
    else:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(7 * cols, 6 * rows))
        for i, seq in enumerate(sequences):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            plot_sequence(ax, seq, f"Seq {seq['sequence']} ({seq['n_poses']} poses)")

    config = data.get("config", {})
    fig.suptitle(
        f"exp34 Denoising — sw={config.get('sw')}, lr={config.get('lr')}, "
        f"window={config.get('window_size')}",
        fontsize=13, y=0.98)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
