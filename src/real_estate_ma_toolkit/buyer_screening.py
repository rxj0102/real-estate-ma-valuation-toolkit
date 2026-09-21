"""Buyer/investor screening and ranking framework.

Mirrors the buyer-side workstream of a real sell-side engagement: build a
structured universe of strategic and financial buyers, score each on
criteria that actually drive a real bid (portfolio/geography fit, balance
sheet capacity relative to the deal size, and alignment with the buyer's
stated investment thesis), and produce a ranked target list to prioritize
outreach.

The fit and thesis-alignment scores in ``data/buyers.csv`` are analyst
judgment calls, exactly as they would be in a real buyer screen -- this
module does not pretend to infer them automatically from free text. What
it does compute directly from the deal is the capacity score, since that
is a mechanical function of the buyer's stated check size range versus the
actual implied deal size coming out of the valuation work.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Buyer:
    """A single prospective buyer or investor.

    Attributes:
        name: Buyer name.
        buyer_type: ``"Strategic"`` or ``"Financial"``.
        description: One-line description of the buyer and its strategy.
        equity_capacity_millions: Approximate available equity capital, $ millions.
        target_check_size_min_millions: Smallest deal size the buyer
            typically pursues, $ millions.
        target_check_size_max_millions: Largest deal size the buyer
            typically pursues, $ millions.
        geographic_fit_score: Analyst-assessed fit with the target's
            footprint, 1 (poor) to 5 (excellent).
        asset_class_fit_score: Analyst-assessed fit with the target's asset
            type/quality, 1 to 5.
        thesis_alignment_score: Analyst-assessed fit with the buyer's
            stated investment thesis, 1 to 5.
        leverage_appetite: ``"Low"``, ``"Medium"``, or ``"High"``.
        notes: Free-text diligence notes.
    """

    name: str
    buyer_type: str
    description: str
    equity_capacity_millions: float
    target_check_size_min_millions: float
    target_check_size_max_millions: float
    geographic_fit_score: int
    asset_class_fit_score: int
    thesis_alignment_score: int
    leverage_appetite: str
    notes: str


def load_buyers(path: str) -> list[Buyer]:
    """Load the buyer universe from CSV (see ``data/buyers.csv``)."""
    df = pd.read_csv(path)
    return [Buyer(**row) for row in df.to_dict(orient="records")]


DEFAULT_WEIGHTS: dict[str, float] = {
    "geographic_fit": 0.20,
    "asset_class_fit": 0.20,
    "thesis_alignment": 0.30,
    "capacity": 0.30,
}


def capacity_score(buyer: Buyer, implied_deal_size_millions: float) -> float:
    """Score (0-5) of how well the deal size fits the buyer's typical check size.

    A deal inside the buyer's stated range scores a full 5. A deal above
    the buyer's typical maximum, or below its typical minimum, is
    penalized proportionally to how far outside the range it falls --
    too large a deal risks exceeding the buyer's underwriting/committee
    capacity; too small a deal risks being a low priority relative to the
    buyer's other opportunities.
    """
    lo, hi = buyer.target_check_size_min_millions, buyer.target_check_size_max_millions
    if lo <= implied_deal_size_millions <= hi:
        return 5.0
    if implied_deal_size_millions > hi:
        shortfall_pct = (implied_deal_size_millions - hi) / hi
        return max(0.0, 5.0 - shortfall_pct * 10)
    excess_pct = (lo - implied_deal_size_millions) / lo
    return max(0.0, 5.0 - excess_pct * 5)


def score_buyers(
    buyers: list[Buyer],
    implied_deal_size_millions: float,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score and rank the buyer universe against the target deal.

    Args:
        buyers: The buyer universe.
        implied_deal_size_millions: The deal size to screen against --
            typically the football field midpoint equity value (or
            enterprise value, if comparing gross capacity), $ millions.
        weights: Weights for each scoring criterion; must sum to 1.0.
            Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns:
        A DataFrame ranked by weighted score (descending), with each
        component score broken out.
    """
    weights = weights or DEFAULT_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    rows = []
    for b in buyers:
        cap_score = capacity_score(b, implied_deal_size_millions)
        weighted_score = (
            weights["geographic_fit"] * b.geographic_fit_score
            + weights["asset_class_fit"] * b.asset_class_fit_score
            + weights["thesis_alignment"] * b.thesis_alignment_score
            + weights["capacity"] * cap_score
        )
        rows.append(
            {
                "name": b.name,
                "buyer_type": b.buyer_type,
                "geographic_fit_score": b.geographic_fit_score,
                "asset_class_fit_score": b.asset_class_fit_score,
                "thesis_alignment_score": b.thesis_alignment_score,
                "capacity_score": round(cap_score, 2),
                "weighted_score": round(weighted_score, 2),
                "leverage_appetite": b.leverage_appetite,
                "description": b.description,
                "notes": b.notes,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("weighted_score", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda d: range(1, len(d) + 1))
        .set_index("rank")
    )


def top_targets(scored: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Convenience slice of the top-N ranked buyers."""
    return scored.head(n)
