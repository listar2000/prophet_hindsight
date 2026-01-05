"""
Data Loading Stage - Stage 1

Loads raw prediction and submission data from either:
- Supabase database (using db.py functions)
- Existing CSV files
"""

import logging

import pandas as pd

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class DataLoadingStage(PipelineStage):
    """Load raw prediction and submission data."""

    name = "data_loading"

    def validate_inputs(self, state: PipelineState) -> bool:
        """This is the first stage, so no inputs required."""
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Load data from database or files.

        If predictions_path and submissions_path are provided in config,
        loads from files. Otherwise loads from database.
        """
        data_config = self.config.data

        if data_config.predictions_path and data_config.submissions_path:
            # Load from files
            self.logger.info(
                f"Loading predictions from {data_config.predictions_path}"
                f" and submissions from {data_config.submissions_path}"
            )
            predictions_df = PipelineState._load_dataframe(data_config.predictions_path)
            submissions_df = PipelineState._load_dataframe(data_config.submissions_path)
        else:
            # Load from database
            predictions_df, submissions_df = self._load_df_from_database()

        assert (
            predictions_df is not None and submissions_df is not None
        ), "Cannot proceed: predictions_df and submissions_df are required"
        state.predictions_df = predictions_df
        state.submissions_df = submissions_df

        self.logger.info(
            f"Loaded {len(predictions_df)} predictions and {len(submissions_df)} submissions"
        )
        return state

    def _load_df_from_database(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Supabase database."""
        from prophet_hindsight.common.db import (
            get_engine,
            load_data_from_db,
            process_predictions,
        )

        data_config = self.config.data

        if not data_config.db_url:
            raise ValueError(
                "Database URL not provided. Set SUPABASE_DB_URL environment variable or provide data.db_url in config."
            )

        engine = get_engine(data_config.db_url)

        self.logger.info("Loading data from database")
        self.logger.info(
            f"Time range: {data_config.filter_time_after} to {data_config.filter_time_before}"
        )
        self.logger.info(f"Agent only: {data_config.filter_agent_only}")

        # Load raw data
        predictions_df, submissions_df, markets_df, events_df = load_data_from_db(
            engine,
            filter_time_before=data_config.filter_time_before,
            filter_time_after=data_config.filter_time_after,
            filter_agent_only=data_config.filter_agent_only,
        )

        # Process predictions and return
        return process_predictions(predictions_df, submissions_df, markets_df, events_df)

    def _count_output_rows(self, state: PipelineState) -> int:
        return len(state.predictions_df) if state.predictions_df is not None else 0

    def _get_config_snapshot(self) -> dict:
        return {
            "filter_time_before": self.config.data.filter_time_before,
            "filter_time_after": self.config.data.filter_time_after,
            "filter_agent_only": self.config.data.filter_agent_only,
            "predictions_path": self.config.data.predictions_path,
            "submissions_path": self.config.data.submissions_path,
        }
