"""
Hyperparameter Analysis Utilities.

Provides tools to analyze how different filter configurations affect
the number of reasoning traces collected.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterSweepResult:
    """Result of a filter parameter sweep."""

    parameter_name: str
    parameter_values: list[float]
    trace_counts: list[int]
    unique_events: list[int]
    unique_forecasters: list[int]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
        return pd.DataFrame(
            {
                self.parameter_name: self.parameter_values,
                "trace_count": self.trace_counts,
                "unique_events": self.unique_events,
                "unique_forecasters": self.unique_forecasters,
            }
        )


def sweep_z_score_threshold(
    brier_df: pd.DataFrame,
    min_z_values: list[float] | None = None,
    min_val: float = 0.15,
    max_mean: float = -1,
) -> FilterSweepResult:
    """
    Sweep the z-score threshold and report the number of traces at each value.

    Args:
        brier_df: DataFrame with Brier scores
        min_z_values: List of min_z values to try (default: [0.5, 1.0, 1.25, 1.5, 1.75, 2.0])
        min_val: Minimum Brier score threshold
        max_mean: Maximum mean Brier score to filter out easy events

    Returns:
        FilterSweepResult with counts at each threshold
    """
    from prophet_hindsight.reasoning.reason_filter import top_z_score_criteria

    if min_z_values is None:
        min_z_values = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

    trace_counts = []
    unique_events = []
    unique_forecasters = []

    for min_z in min_z_values:
        filtered_df = top_z_score_criteria(
            brier_df,
            metric_col="brier_score",
            min_z=min_z,
            max_mean=max_mean,
            min_val=min_val,
        )
        trace_counts.append(len(filtered_df))
        unique_events.append(filtered_df["event_ticker"].nunique() if len(filtered_df) > 0 else 0)
        unique_forecasters.append(
            filtered_df["forecaster"].nunique() if len(filtered_df) > 0 else 0
        )

    return FilterSweepResult(
        parameter_name="min_z",
        parameter_values=min_z_values,
        trace_counts=trace_counts,
        unique_events=unique_events,
        unique_forecasters=unique_forecasters,
    )


def sweep_ambiguous_thresholds(
    brier_df: pd.DataFrame,
    min_val_values: list[float] | None = None,
    max_val: float = 0.15,
    top_k: int = 5,
) -> FilterSweepResult:
    """
    Sweep the ambiguous filter min_val threshold.

    Args:
        brier_df: DataFrame with Brier scores
        min_val_values: List of min_val values to try
        max_val: Maximum Brier score for selection
        top_k: Number of top predictions per submission

    Returns:
        FilterSweepResult with counts at each threshold
    """
    from prophet_hindsight.reasoning.reason_filter import ambiguous_event_criteria

    if min_val_values is None:
        min_val_values = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]

    trace_counts = []
    unique_events = []
    unique_forecasters = []

    for min_val in min_val_values:
        filtered_df = ambiguous_event_criteria(
            brier_df,
            metric_col="brier_score",
            min_val=min_val,
            max_val=max_val,
            top_k=top_k,
        )
        trace_counts.append(len(filtered_df))
        unique_events.append(filtered_df["event_ticker"].nunique() if len(filtered_df) > 0 else 0)
        unique_forecasters.append(
            filtered_df["forecaster"].nunique() if len(filtered_df) > 0 else 0
        )

    return FilterSweepResult(
        parameter_name="min_val",
        parameter_values=min_val_values,
        trace_counts=trace_counts,
        unique_events=unique_events,
        unique_forecasters=unique_forecasters,
    )


def estimate_total_traces(
    brier_df: pd.DataFrame,
    z_score_config: dict | None = None,
    ambiguous_config: dict | None = None,
    n_augmenters: int = 1,
    augment_ratio: float = 1.0,
) -> dict[str, float | int]:
    """
    Estimate the total number of traces that will be generated.

    Args:
        brier_df: DataFrame with Brier scores
        z_score_config: Z-score filter configuration
        ambiguous_config: Ambiguous filter configuration
        n_augmenters: Number of augmenter models
        augment_ratio: Ratio of augmenters per trace (for randomized assignment)

    Returns:
        Dictionary with trace counts at each stage
    """
    from prophet_hindsight.reasoning.reason_filter import (
        ambiguous_event_criteria,
        top_z_score_criteria,
    )

    # Remove market baseline
    brier_df = brier_df[~brier_df["forecaster"].str.startswith("market-baseline")]

    # Z-score filtering
    z_score_df = top_z_score_criteria(
        brier_df,
        metric_col="brier_score",
        **(z_score_config or {}),
    )

    # Ambiguous filtering
    ambiguous_df = ambiguous_event_criteria(
        brier_df,
        metric_col="brier_score",
        **(ambiguous_config or {}),
    )

    # Combined (deduplicated)
    combined = pd.concat([z_score_df, ambiguous_df]).drop_duplicates(
        subset=["forecaster", "event_ticker", "submission_id", "round"]
    )

    # Estimated augmented traces
    avg_augmenters_per_trace = n_augmenters * augment_ratio
    estimated_augmented = int(len(combined) * avg_augmenters_per_trace)

    return {
        "raw_predictions": len(brier_df),
        "z_score_filtered": len(z_score_df),
        "ambiguous_filtered": len(ambiguous_df),
        "combined_filtered": len(combined),
        "unique_events": combined["event_ticker"].nunique(),
        "unique_forecasters": combined["forecaster"].nunique(),
        "n_augmenters": n_augmenters,
        "augment_ratio": augment_ratio,
        "estimated_augmented_traces": estimated_augmented,
    }


def print_filter_analysis(
    brier_df: pd.DataFrame,
    z_score_config: dict | None = None,
    ambiguous_config: dict | None = None,
) -> None:
    """
    Print a comprehensive analysis of filter configurations.

    Args:
        brier_df: DataFrame with Brier scores
        z_score_config: Optional z-score config (uses defaults if None)
        ambiguous_config: Optional ambiguous config (uses defaults if None)
    """
    if z_score_config is None:
        z_score_config = {"min_z": 1.5, "max_mean": -1, "min_val": 0.15}
    if ambiguous_config is None:
        ambiguous_config = {"min_val": 0.25, "max_val": 0.15, "top_k": 5}

    print("=" * 60)
    print("FILTER CONFIGURATION ANALYSIS")
    print("=" * 60)

    # Current configuration estimate
    print("\n📊 Current Configuration Estimate:")
    estimate = estimate_total_traces(brier_df, z_score_config, ambiguous_config)
    for key, value in estimate.items():
        print(f"  {key}: {value}")

    # Z-score sweep
    print("\n📈 Z-Score Threshold Sweep:")
    z_sweep = sweep_z_score_threshold(brier_df, min_val=z_score_config["min_val"])
    print(z_sweep.to_dataframe().to_string(index=False))

    # Ambiguous sweep
    print("\n📉 Ambiguous Min-Val Sweep:")
    amb_sweep = sweep_ambiguous_thresholds(brier_df, max_val=ambiguous_config["max_val"])
    print(amb_sweep.to_dataframe().to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Analyze filter configurations")
    parser.add_argument("--brier-csv", required=True, help="Path to Brier scores CSV")
    args = parser.parse_args()

    brier_df = pd.read_csv(args.brier_csv)
    print_filter_analysis(brier_df)
