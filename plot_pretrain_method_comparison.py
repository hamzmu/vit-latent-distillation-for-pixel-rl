#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SINGLE_SUBSETS = ["keep_cam0", "keep_cam1", "keep_cam2"]
FULL_SUBSET = "keep_cam0_cam1_cam2"
METHOD_ORDER = ["joint", "curriculum", "distill"]
METHOD_LABELS = {
    "joint": "Joint",
    "curriculum": "Curriculum",
    "distill": "Two-Stage Distill",
}
METHOD_COLORS = {
    "joint": "#1f77b4",
    "curriculum": "#2ca02c",
    "distill": "#d62728",
}
SUBSET_LABELS = {
    "keep_cam0": "Cam0 Only",
    "keep_cam1": "Cam1 Only",
    "keep_cam2": "Cam2 Only",
    "keep_cam0_cam1_cam2": "All 3 Cams",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cast_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    table: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        subset = row["subset"]
        method = row["method"]
        table[subset][method] = {
            "total": float(row["total"]),
            "v0": float(row["v0"]),
            "v1": float(row["v1"]),
            "v2": float(row["v2"]),
            "z_cos_to_full": float(row["z_cos_to_full"]),
            "z_mse_to_full": float(row["z_mse_to_full"]),
        }
    return table


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_total_recon(metrics: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(SINGLE_SUBSETS))
    width = 0.22
    offsets = [-width, 0.0, width]

    for off, method in zip(offsets, METHOD_ORDER):
        vals = [metrics[s][method]["total"] for s in SINGLE_SUBSETS]
        ax.bar(x + off, vals, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SINGLE_SUBSETS])
    ax.set_ylabel("Total Reconstruction Loss")
    ax.set_title("Single-Camera Reconstruction Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_alignment(metrics: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(SINGLE_SUBSETS))
    width = 0.22
    offsets = [-width, 0.0, width]

    for off, method in zip(offsets, METHOD_ORDER):
        mse_vals = [metrics[s][method]["z_mse_to_full"] for s in SINGLE_SUBSETS]
        ax.bar(x + off, mse_vals, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_title("Latent MSE to Full")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SINGLE_SUBSETS], rotation=0)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_recon_heatmaps(metrics: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.2, 4.0))
    grid = fig.add_gridspec(1, 5, width_ratios=[0.08, 0.12, 1.0, 1.0, 1.0], wspace=0.18)
    cax = fig.add_subplot(grid[0, 0])
    first_ax = fig.add_subplot(grid[0, 2])
    axes = [
        first_ax,
        fig.add_subplot(grid[0, 3], sharey=first_ax),
        fig.add_subplot(grid[0, 4], sharey=first_ax),
    ]
    target_keys = ["v0", "v1", "v2"]
    vmin, vmax = 0.0, max(
        metrics[s][m][k]
        for s in SINGLE_SUBSETS
        for m in METHOD_ORDER
        for k in target_keys
    )

    for idx, (ax, method) in enumerate(zip(axes, METHOD_ORDER)):
        arr = np.array([[metrics[s][method][k] for k in target_keys] for s in SINGLE_SUBSETS], dtype=float)
        im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(METHOD_LABELS[method])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Recon v0", "Recon v1", "Recon v2"])
        ax.set_yticks(range(3))
        if idx == 0:
            ax.set_yticklabels([SUBSET_LABELS[s] for s in SINGLE_SUBSETS])
        else:
            ax.tick_params(left=False, labelleft=False)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.4f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Per-View Reconstruction Loss")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.tick_left()
    fig.suptitle("Per-View Reconstruction by Method and Input Camera", y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(metrics: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "best_total_method", "best_total", "best_cos_method", "best_cos", "best_mse_method", "best_mse"])
        for subset in [FULL_SUBSET] + SINGLE_SUBSETS:
            total_best = min(METHOD_ORDER, key=lambda m: metrics[subset][m]["total"])
            cos_best = max(METHOD_ORDER, key=lambda m: metrics[subset][m]["z_cos_to_full"])
            mse_best = min(METHOD_ORDER, key=lambda m: metrics[subset][m]["z_mse_to_full"])
            writer.writerow(
                [
                    subset,
                    total_best,
                    metrics[subset][total_best]["total"],
                    cos_best,
                    metrics[subset][cos_best]["z_cos_to_full"],
                    mse_best,
                    metrics[subset][mse_best]["z_mse_to_full"],
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot comparison figures for joint/distill/curriculum pretraining results.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("recon_method_figures"))
    args = p.parse_args()

    rows = load_rows(args.csv)
    metrics = cast_metrics(rows)
    ensure_dir(args.out_dir)

    plot_total_recon(metrics, args.out_dir / "single_cam_total_reconstruction.png")
    plot_latent_alignment(metrics, args.out_dir / "single_cam_latent_alignment.png")
    plot_recon_heatmaps(metrics, args.out_dir / "single_cam_per_view_heatmaps.png")
    save_summary_table(metrics, args.out_dir / "summary_rankings.csv")

    print(f"Saved figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
