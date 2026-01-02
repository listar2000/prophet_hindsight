"""
Filtering Stage - Stage 3

Applies filtering criteria to select high-quality predictions:
- Z-score criteria: predictions significantly better than peers
- Ambiguous event criteria: good predictions on uncertain events
- Absolute score criteria: top performers by raw score

Also augments filtered predictions with sources, context, and event details.
"""

import logging

import pandas as pd

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class FilteringStage(PipelineStage):
    """Apply filtering criteria and augment with additional data."""

    name = "filtering"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that Brier scores are computed."""
        if state.brier_scores_df is None:
            self.logger.error("brier_scores_df is required but not present")
            return False
        # make sure there's no market-baseline predictions in the brier_scores_df
        if "market-baseline" in state.brier_scores_df["forecaster"].values:
            self.logger.error("market-baseline predictions are not allowed in the brier_scores_df")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Apply filtering criteria and augment with additional data.
        """
        from prophet_hindsight.common.db import get_engine
        from prophet_hindsight.reasoning.reason_filter import (
            ambiguous_event_criteria,
            augment_filtered_predictions,
            top_absolute_score_criteria,
            top_z_score_criteria,
        )

        filter_config = self.config.filter
        data_config = self.config.data

        brier_df = state.brier_scores_df.copy()  # type: ignore

        filtered_dfs = []

        # Apply Z-score filtering
        if filter_config.z_score.enabled:
            self.logger.info("Applying z-score filtering...")
            z_score_df = top_z_score_criteria(
                brier_df,
                metric_col=filter_config.metric_col,
                min_z=filter_config.z_score.min_z,
                max_mean=filter_config.z_score.max_mean,
                min_val=filter_config.z_score.min_val,
            )
            state.z_score_filtered_df = z_score_df
            filtered_dfs.append(z_score_df)
            self.logger.info(f"Z-score filter: {len(z_score_df)} predictions")

        # Apply Ambiguous event filtering
        if filter_config.ambiguous.enabled:
            self.logger.info("Applying ambiguous event filtering...")
            ambiguous_df = ambiguous_event_criteria(
                brier_df,
                metric_col=filter_config.metric_col,
                min_val=filter_config.ambiguous.min_val,
                max_val=filter_config.ambiguous.max_val,
                top_k=filter_config.ambiguous.top_k,
            )
            state.ambiguous_filtered_df = ambiguous_df
            filtered_dfs.append(ambiguous_df)
            self.logger.info(f"Ambiguous filter: {len(ambiguous_df)} predictions")

        # Apply Absolute score filtering
        if filter_config.absolute.enabled:
            self.logger.info("Applying absolute score filtering...")
            absolute_df = top_absolute_score_criteria(
                brier_df,
                metric_col=filter_config.metric_col,
                top_k=filter_config.absolute.top_k,
                top_p=filter_config.absolute.top_p,
                min_val=filter_config.absolute.min_val,
            )
            filtered_dfs.append(absolute_df)
            self.logger.info(f"Absolute filter: {len(absolute_df)} predictions")

        # Combine filtered results
        if not filtered_dfs:
            self.logger.warning("No filtering criteria enabled, using all predictions")
            combined_df = brier_df
        else:
            combined_df = pd.concat(filtered_dfs, ignore_index=True)
            # Remove duplicates (same prediction might match multiple criteria)
            combined_df = combined_df.drop_duplicates(
                subset=["forecaster", "event_ticker", "submission_id", "round"], keep="first"
            )

        state.combined_filtered_df = combined_df
        self.logger.info(f"Combined filtered predictions: {len(combined_df)} rows")

        # Augment with sources, prediction context, event details, and market data
        self.logger.info("Augmenting filtered predictions with additional data...")

        engine = get_engine(data_config.db_url)

        # Get submissions path for market data
        submissions_path = data_config.submissions_path
        if submissions_path is None and state.submissions_df is not None:
            # Save submissions temporarily for augmentation
            import os
            import tempfile

            temp_dir = tempfile.mkdtemp()
            submissions_path = os.path.join(temp_dir, "submissions.csv")
            state.submissions_df.to_csv(submissions_path, index=False)

        augmented_df = augment_filtered_predictions(
            combined_df,
            engine=engine,
            submission_df_path=submissions_path,
            filter_contributor_only=data_config.filter_contributor_only,
            filter_category=data_config.filter_category,
        )

        state.augmented_filtered_df = augmented_df
        self.logger.info(f"Augmented filtered predictions: {len(augmented_df)} rows")

        return state

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.brier_scores_df is not None:
            return len(state.brier_scores_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        if state.augmented_filtered_df is not None:
            return len(state.augmented_filtered_df)
        elif state.combined_filtered_df is not None:
            return len(state.combined_filtered_df)
        return 0

    def _get_config_snapshot(self) -> dict:
        filter_config = self.config.filter
        return {
            "metric_col": filter_config.metric_col,
            "z_score_enabled": filter_config.z_score.enabled,
            "z_score_min_z": filter_config.z_score.min_z,
            "ambiguous_enabled": filter_config.ambiguous.enabled,
            "ambiguous_min_val": filter_config.ambiguous.min_val,
            "absolute_enabled": filter_config.absolute.enabled,
        }
