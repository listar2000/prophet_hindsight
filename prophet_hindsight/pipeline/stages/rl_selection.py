"""
RL Selection Stage - RL Pipeline Stage 3

Selects prediction problems for RL training. Unlike SFT filtering which
selects high-quality reasoning traces, RL selection:
1. Deduplicates problems (one per (event_ticker, submission_id) pair)
2. Excludes test set problems
3. Augments with sources, context, and event details

This stage does NOT filter by quality since RL training benefits from
diverse data rather than only high-quality examples.
"""

import logging

import pandas as pd

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class RLSelectionStage(PipelineStage):
    """Select and deduplicate problems for RL training."""

    name = "rl_selection"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that Brier scores are computed."""
        if state.brier_scores_df is None:
            self.logger.error("brier_scores_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Select and deduplicate problems for RL training.

        Unlike SFT filtering:
        - Does not filter by quality metrics
        - Deduplicates to one prediction per (event_ticker, submission_id)
        - Excludes test set problems if specified
        """
        from prophet_hindsight.common.db import get_engine
        from prophet_hindsight.reasoning.reason_filter import (
            augment_filtered_predictions,
        )

        rl_config = self.config.rl_selection
        data_config = self.config.data

        brier_df = state.brier_scores_df.copy()  # type: ignore
        self.logger.info(f"Starting RL selection with {len(brier_df)} predictions")

        # Step 1: Exclude test set problems if specified
        if rl_config.test_set_path:
            brier_df = self._exclude_test_set(brier_df, rl_config.test_set_path)

        # Step 2: Deduplicate to one prediction per (event_ticker, submission_id)
        selected_df = self._deduplicate_problems(brier_df, rl_config.dedup_strategy)
        self.logger.info(f"After deduplication: {len(selected_df)} unique problems")

        # Step 3: Apply max_problems limit if specified
        if rl_config.max_problems > 0 and len(selected_df) > rl_config.max_problems:
            self.logger.info(
                f"Limiting to {rl_config.max_problems} problems (from {len(selected_df)})"
            )
            if rl_config.shuffle:
                selected_df = selected_df.sample(
                    n=rl_config.max_problems, random_state=self.config.run.seed
                )
            else:
                selected_df = selected_df.head(rl_config.max_problems)
        elif rl_config.shuffle:
            selected_df = selected_df.sample(frac=1, random_state=self.config.run.seed)

        # Store the deduplicated selection (before augmentation)
        state.rl_selected_df = selected_df.copy()
        self.logger.info(f"Selected {len(selected_df)} problems for RL training")

        # Step 4: Augment with sources, context, and event details
        self.logger.info("Augmenting selected problems with additional data...")

        engine = get_engine(data_config.db_url)

        # Get submissions path for market data
        submissions_path = data_config.submissions_path
        if submissions_path is None and state.submissions_df is not None:
            import os
            import tempfile

            temp_dir = tempfile.mkdtemp()
            submissions_path = os.path.join(temp_dir, "submissions.csv")
            state.submissions_df.to_csv(submissions_path, index=False)

        # Reuse the augmentation function from SFT pipeline
        augmented_df = augment_filtered_predictions(
            selected_df,
            engine=engine,
            submission_df_path=submissions_path,
            filter_contributor_only=data_config.filter_contributor_only,
            filter_category=data_config.filter_category,
        )

        state.rl_augmented_df = augmented_df
        self.logger.info(f"Augmented RL selection: {len(augmented_df)} rows")

        return state

    def _exclude_test_set(
        self, brier_df: pd.DataFrame, test_set_path: str
    ) -> pd.DataFrame:
        """Exclude problems that are in the test set."""
        try:
            # Load test set and extract identifying columns
            test_df = PipelineState._load_dataframe(test_set_path)

            # Ensure required columns exist
            if (
                "event_ticker" not in test_df.columns
                or "submission_id" not in test_df.columns
            ):
                self.logger.warning(
                    f"Test set at {test_set_path} missing required columns "
                    "(event_ticker, submission_id). Skipping exclusion."
                )
                return brier_df

            # Create exclusion set
            test_keys = set(
                zip(
                    test_df["event_ticker"].astype(str),
                    test_df["submission_id"].astype(str),
                )
            )

            # Filter out test set problems
            brier_df = brier_df.copy()
            brier_df["_key"] = list(
                zip(
                    brier_df["event_ticker"].astype(str),
                    brier_df["submission_id"].astype(str),
                )
            )
            before_len = len(brier_df)
            brier_df = brier_df[~brier_df["_key"].isin(test_keys)]
            brier_df = brier_df.drop(columns=["_key"])

            excluded_count = before_len - len(brier_df)
            self.logger.info(
                f"Excluded {excluded_count} predictions from test set "
                f"({len(test_keys)} test problems)"
            )

            return brier_df

        except Exception as e:
            self.logger.warning(f"Failed to load test set from {test_set_path}: {e}")
            return brier_df

    def _deduplicate_problems(
        self, brier_df: pd.DataFrame, strategy: str
    ) -> pd.DataFrame:
        """
        Deduplicate to one prediction per (event_ticker, submission_id).

        Args:
            brier_df: DataFrame with predictions
            strategy: Deduplication strategy
                - "first": Keep first occurrence
                - "random": Randomly select one
                - "best_brier": Keep prediction with lowest Brier score

        Returns:
            Deduplicated DataFrame
        """
        group_cols = ["event_ticker", "submission_id"]

        if strategy == "first":
            return brier_df.drop_duplicates(subset=group_cols, keep="first")

        elif strategy == "random":
            # Shuffle first, then keep first (effectively random)
            shuffled = brier_df.sample(frac=1, random_state=self.config.run.seed)
            return shuffled.drop_duplicates(subset=group_cols, keep="first")

        elif strategy == "best_brier":
            # Sort by Brier score (ascending) and keep first
            sorted_df = brier_df.sort_values("brier_score", ascending=True)
            return sorted_df.drop_duplicates(subset=group_cols, keep="first")

        else:
            self.logger.warning(f"Unknown dedup strategy '{strategy}', using 'first'")
            return brier_df.drop_duplicates(subset=group_cols, keep="first")

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.brier_scores_df is not None:
            return len(state.brier_scores_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        if state.rl_augmented_df is not None:
            return len(state.rl_augmented_df)
        elif state.rl_selected_df is not None:
            return len(state.rl_selected_df)
        return 0

    def _get_config_snapshot(self) -> dict:
        rl_config = self.config.rl_selection
        return {
            "test_set_path": rl_config.test_set_path,
            "dedup_strategy": rl_config.dedup_strategy,
            "max_problems": rl_config.max_problems,
            "shuffle": rl_config.shuffle,
        }
