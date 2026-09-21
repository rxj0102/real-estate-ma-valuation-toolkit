"""Football field valuation summary: combines DCF, comps, and precedent
transactions into the standard investment-banking range-of-outcomes chart.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class MethodologyRange:
    """A single valuation methodology's implied equity value range.

    Attributes:
        label: Display label for the chart/table (e.g. "DCF - Gordon Growth").
        low: Low end of the implied equity value range, $.
        high: High end of the implied equity value range, $.
    """

    label: str
    low: float
    high: float

    def midpoint(self) -> float:
        return (self.low + self.high) / 2


def build_summary_table(ranges: list[MethodologyRange]) -> pd.DataFrame:
    """Tabular version of the football field: one row per methodology."""
    return pd.DataFrame(
        [{"Methodology": r.label, "Low ($M)": r.low / 1e6, "High ($M)": r.high / 1e6, "Midpoint ($M)": r.midpoint() / 1e6} for r in ranges]
    ).set_index("Methodology")


def overall_recommended_range(ranges: list[MethodologyRange]) -> tuple[float, float, float]:
    """A blended recommended range: the band where every methodology's
    range overlaps (the intersection: the highest low and the lowest
    high), which is the standard, more defensible way to narrow a
    football field to a headline "recommended range" rather than citing
    the full min-to-max span across all methods, which is typically far
    wider than any single method's own range and overstates uncertainty.

    Falls back to the full union (min of lows, max of highs) only if the
    methodologies do not all overlap.

    Returns:
        (low, high, midpoint_of_midpoints), all in $.
    """
    intersect_low = max(r.low for r in ranges)
    intersect_high = min(r.high for r in ranges)
    mid = sum(r.midpoint() for r in ranges) / len(ranges)
    if intersect_low <= intersect_high:
        return intersect_low, intersect_high, mid
    return min(r.low for r in ranges), max(r.high for r in ranges), mid


def plot_football_field(ranges: list[MethodologyRange], company_name: str, output_path: str) -> str:
    """Render the football field chart (horizontal range bars, one per methodology).

    Args:
        ranges: Valuation ranges, one per methodology, in the order they
            should appear top-to-bottom.
        company_name: Used in the chart title.
        output_path: File path to save the PNG to.

    Returns:
        The output path, for convenience.
    """
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(ranges) + 2))

    labels = [r.label for r in ranges][::-1]
    lows = [r.low / 1e6 for r in ranges][::-1]
    highs = [r.high / 1e6 for r in ranges][::-1]
    mids = [r.midpoint() / 1e6 for r in ranges][::-1]

    y_positions = range(len(ranges))
    bar_color = "#2C5F7C"
    for y, low, high, mid in zip(y_positions, lows, highs, mids):
        ax.barh(y, high - low, left=low, height=0.5, color=bar_color, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.plot([mid, mid], [y - 0.28, y + 0.28], color="white", linewidth=2)
        ax.text(low, y, f" ${low:,.0f}M", va="center", ha="right", fontsize=9)
        ax.text(high, y, f" ${high:,.0f}M ", va="center", ha="left", fontsize=9)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Implied Equity Value ($ millions)", fontsize=10)
    ax.set_title(f"{company_name} -- Valuation Summary (Football Field)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Axis limits must span the full union of every bar (in the same $M
    # units the bars themselves are plotted in), not the narrower overlap
    # range used for the headline "recommended range" figure.
    axis_low = min(lows)
    axis_high = max(highs)
    padding = (axis_high - axis_low) * 0.15
    ax.set_xlim(max(0, axis_low - padding), axis_high + padding)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
