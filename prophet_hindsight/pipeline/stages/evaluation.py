"""
Evaluation Stage - Stage 2

Computes evaluation metrics (primarily Brier scores) for all predictions.
Optionally adds market baseline predictions for comparison.
"""

import logging

from prophet_hindsight.common.forecast import ProphetForecasts, uniform_weighting
from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class EvaluationStage(PipelineStage):
    """Compute evaluation metrics for predictions."""

    name = "evaluation"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that predictions and submissions are loaded."""
        if state.predictions_df is None:
            self.logger.error("predictions_df is required but not present")
            return False
        if state.submissions_df is None:
            self.logger.error("submissions_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Compute Brier scores for all predictions.

        Optionally adds market baseline predictions before computing scores.
        """
        from prophet_hindsight.common.forecast import _parse_and_merge_predictions_and_submissions
        from prophet_hindsight.evaluate.algo import compute_brier_score

        self.logger.info("Parsing and merging predictions with submissions...")

        # Create a ProphetForecasts-like merged dataframe
        weight_fn = uniform_weighting()

        merged_df = _parse_and_merge_predictions_and_submissions(
            state.predictions_df[ProphetForecasts.PREDICTION_COLS],  # type: ignore[index]
            state.submissions_df[ProphetForecasts.SUBMISSION_COLS],  # type: ignore[index]
            weight_fn,
        )

        self.logger.info(f"Merged {len(merged_df)} rows")

        # Compute Brier scores
        self.logger.info("Computing Brier scores...")
        state.brier_scores_df = compute_brier_score(merged_df, append=True)

        self.logger.info(f"Obtained Brier scores for {len(state.brier_scores_df)} predictions")
        return state

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.predictions_df is not None:
            return len(state.predictions_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        if state.brier_scores_df is not None:
            return len(state.brier_scores_df)
        return 0

    def _get_config_snapshot(self) -> dict:
        return {"metric": "brier_score"}
