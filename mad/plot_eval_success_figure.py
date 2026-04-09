#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PANEL_ORDER = [
    ("average_success", "Average"),
    ("first+third1+third2_success", "All Cameras"),
    ("first_success", "First Person"),
    ("third1_success", "Third Person A"),
    ("third2_success", "Third Person B"),
]


def load_eval_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def add_average_column(rows: list[dict[str, float]]) -> None:
    for row in rows:
        row["average_success"] = (
            row["first_success"]
            + row["third1_success"]
            + row["third2_success"]
            + row["first+third1+third2_success"]
        ) / 4.0


def millions_formatter(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    return f"{x/1e6:.2f}M".rstrip("0").rstrip(".")


def pct_formatter(y: float, _pos: int) -> str:
    return f"{int(round(100 * y))}"


def save_summary_table(rows: list[dict[str, float]], out_path: Path) -> None:
    headers = ["step", "Average", "All Cameras", "First Person", "Third Person A", "Third Person B"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    int(row["step"]),
                    round(100 * row["average_success"], 2),
                    round(100 * row["first+third1+third2_success"], 2),
                    round(100 * row["first_success"], 2),
                    round(100 * row["third1_success"], 2),
                    round(100 * row["third2_success"], 2),
                ]
            )


def make_plot(rows: list[dict[str, float]], title: str, out_path: Path) -> None:
    steps = [row["step"] for row in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    line_color = "#1f77b4"

    for ax, (key, panel_title) in zip(axes, PANEL_ORDER):
        values = [100.0 * row[key] for row in rows]
        ax.plot(steps, values, color=line_color, linewidth=2.8, marker="o", markersize=4.5)
        ax.set_title(panel_title, pad=8)
        ax.set_ylim(0, 100)
        ax.set_xlim(min(steps), max(steps))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y)}"))
        ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Success Rate")
    fig.supxlabel("Environment Steps", y=0.04)
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot MAD-style eval success panels from one eval CSV.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("eval_success_figure.png"))
    p.add_argument("--table_out", type=Path, default=None)
    p.add_argument("--title", type=str, default="Button-Press-Topdown Success")
    args = p.parse_args()

    rows = load_eval_csv(args.csv)
    add_average_column(rows)
    make_plot(rows, args.title, args.out)
    table_out = args.table_out or args.out.with_suffix(".csv")
    save_summary_table(rows, table_out)
    print(f"Saved figure: {args.out.resolve()}")
    print(f"Saved table: {table_out.resolve()}")


if __name__ == "__main__":
    main()
