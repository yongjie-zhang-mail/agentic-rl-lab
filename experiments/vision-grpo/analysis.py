"""Plot the focused GeoQA evaluation comparison before and after GRPO."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


TICK_STEP = 4

BASE_COLOR = "#AEB4BE"
GRPO_COLOR = "#4A86E8"
INK = "#252525"
MUTED_INK = "#777777"
GRID_COLOR = "#D6D9DE"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_RESULTS = SCRIPT_DIR / "eval-results-base.json"
DEFAULT_GRPO_RESULTS = SCRIPT_DIR / "eval-results-step-100.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "eval-comparison.png"


def configure_style() -> None:
    """Configure the serif typography and restrained paper-figure palette."""
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "hatch.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: Axes) -> None:
    """Match the framed, inward-tick axes used in the reference figure."""
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle="-.",
        linewidth=0.65,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    ax.tick_params(top=True, right=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def load_eval_results(path: Path) -> dict:
    """Read an eval.py output JSON and convert metrics to percentages."""
    if not path.is_file():
        raise SystemExit(f"Eval results not found: {path}. Run eval.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "accuracy": data["metrics"]["accuracy"] * 100,
        "format_rate": data["metrics"]["format_rate"] * 100,
        "eval_size": data["eval_size"],
        "eval_seed": data["eval_seed"],
        "base_model": data["base_model"],
        "model_path": data["model_path"],
    }


def checkpoint_label(model_path: str | None) -> str:
    match = re.search(r"step-(\d+)", model_path or "")
    return f"Step {match.group(1)}" if match else "GRPO"


def focused_axes(
    metric_values: list[tuple[float, float]],
) -> list[tuple[tuple[int, int], list[int]]]:
    """Per-metric focused y-limits that share one span for equal visual scale."""
    # 3 pp below the lower bar, 5 pp above the higher bar for its value label.
    bounds = [
        (
            math.floor((min(values) - 3) / TICK_STEP) * TICK_STEP,
            math.ceil((max(values) + 5) / TICK_STEP) * TICK_STEP,
        )
        for values in metric_values
    ]
    span = min(max(upper - lower for lower, upper in bounds), 100)
    axes = []
    for lower, _ in bounds:
        lower = min(max(lower, 0), 100 - span)
        upper = lower + span
        axes.append(((lower, upper), list(range(lower, upper + 1, TICK_STEP))))
    return axes


def plot_metric(
    ax: Axes,
    *,
    title: str,
    base_value: float,
    grpo_value: float,
    checkpoint: str,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    show_ylabel: bool,
) -> None:
    """Plot one Base-versus-checkpoint comparison on a focused y-axis."""
    x_positions = [0, 1]
    bars = ax.bar(
        x_positions,
        [base_value, grpo_value],
        width=0.48,
        color=[BASE_COLOR, GRPO_COLOR],
        edgecolor=INK,
        linewidth=0.85,
        zorder=3,
    )
    bars[1].set_hatch("///")

    value_offset = (y_limits[1] - y_limits[0]) * 0.04
    for x_position, value in zip(
        x_positions,
        [base_value, grpo_value],
        strict=True,
    ):
        ax.text(
            x_position,
            value + value_offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.0,
        )

    delta = grpo_value - base_value
    delta_y = (base_value + grpo_value) / 2 + value_offset
    ax.text(
        0.5,
        delta_y,
        f"{delta:+.1f} pp",
        ha="center",
        va="center",
        color="#2563C4",
        fontsize=8.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.23",
            "facecolor": "white",
            "edgecolor": "#7DB4FF",
            "linewidth": 0.9,
        },
        zorder=5,
    )

    ax.text(
        0.02,
        0.965,
        "focused y-axis",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#9A9A9A",
        fontsize=6.7,
        fontstyle="italic",
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Model / Checkpoint", labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Score")
    ax.set_xticks(x_positions, ["Base", checkpoint])
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    style_axis(ax)


def build_figure(
    base: dict,
    grpo: dict,
    *,
    checkpoint: str,
    eval_size: int,
    model_name: str,
) -> plt.Figure:
    """Build the two-panel publication-style comparison figure."""
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    metrics = [("accuracy", "(a) Accuracy"), ("format_rate", "(b) Format Rate")]
    axis_specs = focused_axes([(base[key], grpo[key]) for key, _ in metrics])
    for index, ((key, title), (y_limits, y_ticks)) in enumerate(
        zip(metrics, axis_specs, strict=True)
    ):
        plot_metric(
            axes[index],
            title=title,
            base_value=base[key],
            grpo_value=grpo[key],
            checkpoint=checkpoint,
            y_limits=y_limits,
            y_ticks=y_ticks,
            show_ylabel=index == 0,
        )

    figure.suptitle(
        "GeoQA · Vision GRPO Evaluation",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=12.0,
        fontweight="normal",
        color=INK,
    )
    figure.text(
        0.5,
        0.875,
        f"Fixed {eval_size}-question evaluation set · {model_name} · independent focused y-axes",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED_INK,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.72,
        wspace=0.22,
    )
    return figure


def save_figure(figure: plt.Figure, output: Path) -> Path:
    """Save the comparison as a 600 DPI PNG."""
    png_path = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(png_path, dpi=600)
    plt.close(figure)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Base vs. GRPO checkpoint GeoQA evaluation results."
    )
    parser.add_argument(
        "--base-results",
        type=Path,
        default=DEFAULT_BASE_RESULTS,
        help="eval.py output JSON for the base model.",
    )
    parser.add_argument(
        "--grpo-results",
        type=Path,
        default=DEFAULT_GRPO_RESULTS,
        help="eval.py output JSON for the GRPO checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_eval_results(args.base_results)
    grpo = load_eval_results(args.grpo_results)
    for key in ("eval_size", "eval_seed"):
        if base[key] != grpo[key]:
            raise SystemExit(f"Mismatched {key}: base={base[key]}, grpo={grpo[key]}")

    figure = build_figure(
        base,
        grpo,
        checkpoint=checkpoint_label(grpo["model_path"]),
        eval_size=base["eval_size"],
        model_name=base["base_model"].split("/")[-1],
    )
    png_path = save_figure(figure, args.output)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
