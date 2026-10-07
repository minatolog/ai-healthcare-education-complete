"""Static visualisations for model training progress."""

from __future__ import annotations

import math
import os
import statistics
import tempfile
from pathlib import Path

# Keep headless rendering caches in a writable temporary location. This must be
# configured before Matplotlib is imported.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "comp9900_diabetes_model_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .model_artifact import FEATURE_IMPORTANCE_NOTE, FeatureImportanceEntry
from .robustness import ROBUSTNESS_NOTE, SelectionRobustnessReport
from .training import RandomForestTrainingResult


def plot_training_history(
    training_result: RandomForestTrainingResult,
    output_path: str | Path,
) -> Path:
    """Render an English, client-ready training/validation accuracy chart."""

    if not training_result.history:
        raise ValueError("Training history must not be empty.")

    tree_counts = [point.trees_built for point in training_result.history]
    training_scores = [point.training_accuracy * 100 for point in training_result.history]
    validation_scores = [
        point.validation_accuracy * 100 for point in training_result.history
    ]
    minimum_score = min(training_scores + validation_scores)
    y_minimum = max(0, math.floor((minimum_score - 10) / 10) * 10)

    figure = Figure(figsize=(9.6, 5.8), dpi=160, facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#ffffff")

    axis.plot(
        tree_counts,
        training_scores,
        color="#2563eb",
        linewidth=2.6,
        marker="o",
        markersize=4.5,
        label="Training accuracy",
    )
    axis.plot(
        tree_counts,
        validation_scores,
        color="#f97316",
        linewidth=2.6,
        marker="o",
        markersize=4.5,
        label="Validation accuracy",
    )

    best_x = training_result.best_tree_count
    best_y = training_result.best_validation_accuracy * 100
    axis.scatter(
        [best_x],
        [best_y],
        s=150,
        marker="*",
        color="#dc2626",
        edgecolor="#ffffff",
        linewidth=0.9,
        zorder=5,
        label="Selected model",
    )
    axis.annotate(
        f"Selected: {best_x} trees ({best_y:.0f}%)",
        xy=(best_x, best_y),
        xytext=(best_x - 33, best_y - 8),
        fontsize=9.5,
        color="#991b1b",
        arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 1.2},
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fef2f2",
            "edgecolor": "#fecaca",
        },
    )

    axis.set_title(
        "Random Forest Training Progress",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
        pad=15,
    )
    axis.text(
        0.5,
        1.01,
        "20 training patients • 10 validation patients • synthetic educational data",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    axis.set_xlabel("Trees built", fontsize=11, color="#334155")
    axis.set_ylabel("Accuracy (%)", fontsize=11, color="#334155")
    axis.set_xlim(min(tree_counts) - 2, max(tree_counts) + 3)
    axis.set_ylim(y_minimum, 102)
    axis.grid(axis="y", color="#cbd5e1", alpha=0.65, linewidth=0.8)
    axis.tick_params(colors="#475569")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#cbd5e1")
    axis.spines["bottom"].set_color("#cbd5e1")
    axis.legend(loc="lower right", frameon=False, fontsize=9.5)

    figure.text(
        0.5,
        0.015,
        "Validation accuracy selects the model; the held-out test set is not shown here.",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.04, 0.055, 0.98, 0.97))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", facecolor=figure.get_facecolor())
    return path


def plot_feature_importance(
    entries: tuple[FeatureImportanceEntry, ...],
    output_path: str | Path,
) -> Path:
    """Render a client-ready global Random Forest importance chart."""

    if not entries:
        raise ValueError("Feature importance entries must not be empty.")

    ordered = sorted(entries, key=lambda entry: entry.importance)
    labels = [entry.display_name for entry in ordered]
    percentages = [entry.importance * 100 for entry in ordered]

    figure = Figure(figsize=(9.6, 5.8), dpi=160, facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#ffffff")
    bars = axis.barh(labels, percentages, color="#2563eb", height=0.62)

    for bar, percentage in zip(bars, percentages):
        axis.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%",
            va="center",
            fontsize=9.5,
            color="#334155",
        )

    axis.set_title(
        "What Influenced the Random Forest?",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
        pad=15,
    )
    axis.text(
        0.5,
        1.01,
        "Global split importance • synthetic educational model",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    axis.set_xlabel("Relative feature importance (%)", fontsize=11, color="#334155")
    axis.set_xlim(0, max(percentages) * 1.22)
    axis.grid(axis="x", color="#cbd5e1", alpha=0.65, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#475569")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#cbd5e1")
    axis.spines["bottom"].set_color("#cbd5e1")

    figure.text(
        0.5,
        0.015,
        FEATURE_IMPORTANCE_NOTE,
        ha="center",
        fontsize=8.6,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.06, 0.06, 0.98, 0.97))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", facecolor=figure.get_facecolor())
    return path


def plot_selection_robustness(
    report: SelectionRobustnessReport,
    output_path: str | Path,
) -> Path:
    """Show how validation and test accuracy vary by student selection."""

    if not report.scenarios:
        raise ValueError("Robustness report must contain at least one scenario.")

    scenario_numbers = [
        scenario.scenario_number for scenario in report.scenarios
    ]
    validation_scores = [
        scenario.best_validation_accuracy * 100
        for scenario in report.scenarios
    ]
    test_scores = [
        scenario.test_accuracy * 100 for scenario in report.scenarios
    ]

    figure = Figure(figsize=(10.2, 5.9), dpi=160, facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#ffffff")
    axis.scatter(
        scenario_numbers,
        validation_scores,
        color="#f97316",
        marker="o",
        s=42,
        alpha=0.85,
        label="Best validation accuracy",
    )
    axis.scatter(
        scenario_numbers,
        test_scores,
        color="#2563eb",
        marker="s",
        s=42,
        alpha=0.85,
        label="Held-out test accuracy",
    )
    axis.axhline(
        statistics.fmean(validation_scores),
        color="#f97316",
        linewidth=1.4,
        linestyle="--",
        alpha=0.7,
    )
    axis.axhline(
        statistics.fmean(test_scores),
        color="#2563eb",
        linewidth=1.4,
        linestyle="--",
        alpha=0.7,
    )

    axis.set_title(
        "Performance Changes with Student Selection",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
        pad=15,
    )
    axis.text(
        0.5,
        1.01,
        f"{report.scenario_count} valid selections • 20 training patients each",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    axis.set_xlabel("Selection scenario", fontsize=11, color="#334155")
    axis.set_ylabel("Accuracy (%)", fontsize=11, color="#334155")
    axis.set_xlim(0, report.scenario_count + 1)
    axis.set_ylim(0, 102)
    axis.grid(axis="y", color="#cbd5e1", alpha=0.65, linewidth=0.8)
    axis.tick_params(colors="#475569")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#cbd5e1")
    axis.spines["bottom"].set_color("#cbd5e1")
    axis.legend(loc="lower right", frameon=False, fontsize=9.5)

    figure.text(
        0.5,
        0.015,
        ROBUSTNESS_NOTE,
        ha="center",
        fontsize=8.6,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.97))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", facecolor=figure.get_facecolor())
    return path
