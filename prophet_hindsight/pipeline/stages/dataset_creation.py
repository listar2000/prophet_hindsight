"""
Dataset Creation Stage - Stage 6

Creates the final HuggingFace Dataset for SFT training.
Combines augmented reasoning traces with predictions and converts to conversation format.
"""

import logging
from pathlib import Path

import pandas as pd

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class DatasetCreationStage(PipelineStage):
    """Create HuggingFace Dataset from augmented data."""

    name = "dataset_creation"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that augmented reasoning traces are available."""
        if not state.augmented_reasoning_df_map:
            self.logger.error("augmented_reasoning_df_map is required but empty")
            return False
        if state.augmented_filtered_df is None:
            self.logger.error("augmented_filtered_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Create HuggingFace Dataset from augmented data.
        """
        import json_repair
        from datasets import Dataset, DatasetDict

        from prophet_trainer.create_dataset import (
            filter_and_replace_rationale,
            message_format,
            pd_train_test_split,
        )

        dataset_config = self.config.output.dataset

        # Combine all augmented reasoning DataFrames
        all_rationales = []
        for model, df in state.augmented_reasoning_df_map.items():
            if df is not None and len(df) > 0:
                all_rationales.append(df)

        # If specified, we merge back the existing augmented reasoning dataframe to the new augmented reasoning dataframe
        if (
            self.config.augment.reasoning.existing_augmented_reasoning_df is not None
            and state._existing_augmented_reasoning_df is not None
        ):
            logger.info(
                f"Merging back {len(state._existing_augmented_reasoning_df)} existing augmented reasoning traces"
            )
            all_rationales.append(state._existing_augmented_reasoning_df)

        if not all_rationales:
            raise ValueError("No augmented reasoning data available")

        combined_rationale_df = pd.concat(all_rationales, ignore_index=True)
        self.logger.info(
            f"Combined {len(combined_rationale_df)} augmented reasoning traces from {len(state.augmented_reasoning_df_map)} models"
        )

        # Filter and replace rationale (remove leakage, fix wording)
        combined_rationale_df = filter_and_replace_rationale(combined_rationale_df)
        self.logger.info(
            f"After removing leakage and replacing: {len(combined_rationale_df)} traces"
        )

        # Get prediction data
        prediction_df = state.augmented_filtered_df.copy()  # type: ignore

        # Deduplicate prediction data
        prediction_df = prediction_df.drop_duplicates(
            subset=["event_ticker", "submission_id", "forecaster"], keep="first"
        )

        # Merge rationale with predictions
        combined_df = combined_rationale_df.merge(
            prediction_df[
                [
                    "event_ticker",
                    "submission_id",
                    "forecaster",
                    "prediction",
                    "market_outcome",
                    "market_data",
                    "sources",
                ]
            ],
            on=["event_ticker", "submission_id", "forecaster"],
            how="left",
        )
        self.logger.info(f"After merging: {len(combined_df)} rows")

        # Parse augmented_rationale if it's a string
        if len(combined_df) > 0 and isinstance(combined_df["augmented_rationale"].iloc[0], str):
            combined_df["augmented_rationale"] = combined_df["augmented_rationale"].apply(
                lambda x: json_repair.loads(x) if pd.notna(x) else None
            )

        # Train/test split
        train_df, test_df = pd_train_test_split(
            combined_df,
            test_size=dataset_config.test_size,
            seed=self.config.run.seed,
        )
        self.logger.info(
            f"Split: {len(train_df)} train, {len(test_df)} test with seed {self.config.run.seed}"
        )

        # Convert to message format
        train_messages = (
            train_df.apply(
                lambda x: message_format(x, conversational=dataset_config.conversational), axis=1
            )
            .dropna()
            .tolist()
        )

        test_messages = (
            test_df.apply(
                lambda x: message_format(x, conversational=dataset_config.conversational), axis=1
            )
            .dropna()
            .tolist()
        )

        self.logger.info(
            f"Converted: {len(train_messages)} train, {len(test_messages)} test messages"
        )

        # Create HuggingFace Dataset
        train_dataset = Dataset.from_list(train_messages)
        test_dataset = Dataset.from_list(test_messages)
        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "test": test_dataset,
            }
        )

        # Save to disk
        run_dir = Path(self.config.get_run_dir())
        dataset_path = run_dir / "dataset"
        dataset_dict.save_to_disk(str(dataset_path), num_proc=dataset_config.n_jobs)
        state.final_dataset_path = str(dataset_path)
        self.logger.info(f"Saved dataset to {dataset_path}")

        # Push to HuggingFace Hub if configured
        if dataset_config.push_to_hub:
            if not dataset_config.repo_id:
                self.logger.warning("push_to_hub is True but repo_id is not set, skipping...")
            else:
                self.logger.info(f"Pushing to HuggingFace Hub: {dataset_config.repo_id}")
                dataset_dict.push_to_hub(
                    dataset_config.repo_id,
                    private=dataset_config.private,
                )
                self.logger.info(
                    f"Successfully pushed to HuggingFace Hub, repo_id: {dataset_config.repo_id}"
                )

        return state

    def _count_input_rows(self, state: PipelineState) -> int:
        return sum(len(df) for df in state.augmented_reasoning_df_map.values())

    def _count_output_rows(self, state: PipelineState) -> int:
        return 0

    def _get_config_snapshot(self) -> dict:
        return {
            "test_size": self.config.output.dataset.test_size,
            "conversational": self.config.output.dataset.conversational,
            "push_to_hub": self.config.output.dataset.push_to_hub,
            "repo_id": self.config.output.dataset.repo_id,
        }
