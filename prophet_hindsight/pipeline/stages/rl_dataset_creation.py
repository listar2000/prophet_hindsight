"""
RL Dataset Creation Stage - RL Pipeline Stage 5

Creates the final HuggingFace Dataset for RL training.
Unlike SFT which includes reasoning traces, RL datasets contain only:
- system prompt
- user prompt (event info, sources, market data)

No assistant response is included since RL training generates its own responses.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from prophet_hindsight.common.utils import unified_json_loads
from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def rl_message_format(row: pd.Series, conversational: bool = True) -> dict:
    """
    Format a row for RL training (prompt-only, no assistant response).

    Unlike SFT message_format, this does NOT include the assistant message
    with reasoning traces and predictions.
    """
    from prophet_trainer.prompt import PredictionPrompts

    event_title = row["title"]
    market_outcome = unified_json_loads(
        row["market_outcome"], dict, raise_on_unknown=False
    )
    if isinstance(market_outcome, dict):
        outcomes_str = ", ".join(list(market_outcome.keys()))
    else:
        outcomes_str = ""

    task_prompt = PredictionPrompts.create_task_prompt().strip()
    user_prompt = PredictionPrompts.create_user_prompt(
        event_title=event_title,
        outcomes_str=outcomes_str,
        sources=row["sources"],
        market_data=row["market_data"],
    ).strip()

    return_dict = {
        "event_ticker": row["event_ticker"],
        "submission_id": row["submission_id"],
        "market_outcome": str(market_outcome),
    }

    if conversational:
        # RL format: only system and user messages, no assistant
        return_dict["messages"] = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # Prompt-only format (no completion)
        return_dict["prompt"] = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_prompt},
        ]

    return return_dict


class RLDatasetCreationStage(PipelineStage):
    """Create HuggingFace Dataset for RL training (prompt-only)."""

    name = "rl_dataset_creation"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that RL augmented data is available."""
        if state.rl_augmented_df is None:
            self.logger.error("rl_augmented_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Create HuggingFace Dataset for RL training.

        Unlike SFT, this creates a prompt-only dataset without assistant responses.
        """
        from datasets import Dataset, DatasetDict

        dataset_config = self.config.output.rl_dataset

        # Get the augmented RL data
        rl_df = state.rl_augmented_df.copy()  # type: ignore
        self.logger.info(f"Creating RL dataset from {len(rl_df)} problems")

        # Parse market_outcome and market_data if they're strings
        if len(rl_df) > 0:
            if isinstance(rl_df["market_outcome"].iloc[0], str):
                rl_df["market_outcome"] = rl_df["market_outcome"].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x
                )
            if isinstance(rl_df["market_data"].iloc[0], str):
                rl_df["market_data"] = rl_df["market_data"].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x
                )

        # Train/test split
        train_df, test_df = self._train_test_split(
            rl_df,
            test_size=dataset_config.test_size,
            seed=self.config.run.seed,
        )
        self.logger.info(
            f"Split: {len(train_df)} train, {len(test_df)} test with seed {self.config.run.seed}"
        )

        # Convert to RL message format (prompt-only)
        train_messages = (
            train_df.apply(
                lambda x: rl_message_format(
                    x, conversational=dataset_config.conversational
                ),
                axis=1,
            )
            .dropna()
            .tolist()
        )

        test_messages = (
            test_df.apply(
                lambda x: rl_message_format(
                    x, conversational=dataset_config.conversational
                ),
                axis=1,
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
        dataset_path = run_dir / "rl_dataset"
        dataset_dict.save_to_disk(str(dataset_path), num_proc=dataset_config.n_jobs)
        state.final_dataset_path = str(dataset_path)
        self.logger.info(f"Saved RL dataset to {dataset_path}")

        # Push to HuggingFace Hub if configured
        if dataset_config.push_to_hub:
            if not dataset_config.repo_id:
                self.logger.warning(
                    "push_to_hub is True but repo_id is not set, skipping..."
                )
            else:
                self.logger.info(
                    f"Pushing to HuggingFace Hub: {dataset_config.repo_id}"
                )
                dataset_dict.push_to_hub(
                    dataset_config.repo_id,
                    private=dataset_config.private,
                )
                self.logger.info(
                    f"Successfully pushed to HuggingFace Hub, repo_id: {dataset_config.repo_id}"
                )

        return state

    def _train_test_split(
        self, df: pd.DataFrame, test_size: float, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Groups by event_ticker to ensure all problems from the same event
        are in the same split.
        """
        test_min_size = int(len(df) * test_size)

        # Group by event_ticker and rank by size
        event_ticker_sizes = df.groupby("event_ticker").size().reset_index(name="count")  # type: ignore

        if seed != -1:
            # Random selection: shuffle event_tickers
            event_ticker_sizes = event_ticker_sizes.sample(
                frac=1, random_state=seed
            ).reset_index(drop=True)
        else:
            # Deterministic selection: sort by size (smallest first)
            event_ticker_sizes = event_ticker_sizes.sort_values(
                "count", ascending=True
            ).reset_index(drop=True)

        # Select event_tickers for test set
        test_event_tickers = set()
        cumulative_size = 0

        for _, row in event_ticker_sizes.iterrows():
            if cumulative_size >= test_min_size:
                break
            test_event_tickers.add(row["event_ticker"])
            cumulative_size += row["count"]

        self.logger.info(
            f"Selected {len(test_event_tickers)} event_tickers for test set "
            f"with {cumulative_size} total rows (target: {test_min_size})"
        )

        # Split by event_ticker
        test_df = df[df["event_ticker"].isin(test_event_tickers)].copy()
        train_df = df[~df["event_ticker"].isin(test_event_tickers)].copy()

        return train_df, test_df

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.rl_augmented_df is not None:
            return len(state.rl_augmented_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        return 0  # HF dataset doesn't have a simple row count

    def _get_config_snapshot(self) -> dict:
        return {
            "test_size": self.config.output.rl_dataset.test_size,
            "conversational": self.config.output.rl_dataset.conversational,
            "push_to_hub": self.config.output.rl_dataset.push_to_hub,
            "repo_id": self.config.output.rl_dataset.repo_id,
        }
