"""
Event Augmentation Stage - Stage 4

Augments event titles to make them more informative and self-contained.
Uses an LLM to transform vague event descriptions into precise questions.
"""

import logging

import pandas as pd

from prophet_hindsight.common.prompts import (
    PromptTemplate,
    get_default_event_augment_prompt,
)
from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class EventAugmentStage(PipelineStage):
    """Augment event titles using LLM."""

    name = "event_augment"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that filtered predictions are available."""
        if state.augmented_filtered_df is None:
            self.logger.error("augmented_filtered_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Augment event titles for all unique events in the filtered predictions.
        """
        from prophet_hindsight.common.judge import LLMJudge
        from prophet_hindsight.event.event_augment import augment_event

        augment_config = self.config.augment.event
        prompts_config = self.config.prompts.event_augment

        if not augment_config.enabled:
            self.logger.info("Event augmentation disabled, skipping...")
            # Create a placeholder DataFrame with original titles
            events_df = state.augmented_filtered_df[["event_ticker", "title"]].drop_duplicates()
            events_df["augmented_title"] = events_df["title"]
            state.augmented_events_df = events_df
            return state

        # Load prompt (from config or default)
        if prompts_config.use_default:
            prompt = get_default_event_augment_prompt()
        else:
            prompt = PromptTemplate.from_yaml(prompts_config.custom_path)

        # Record prompt in state for reproducibility
        state.record_prompt("event_augment", prompt.to_dict())
        self.logger.info(
            f"Using prompt: {prompt.name} v{prompt.version} (hash: {prompt.get_hash()})"
        )

        # Get unique event tickers
        event_tickers = state.augmented_filtered_df["event_ticker"].unique().tolist()
        self.logger.info(f"Augmenting titles for {len(event_tickers)} unique events")

        # Create LLM judge
        judge = LLMJudge(
            model=augment_config.model,
            use_async=True,
            use_openrouter=augment_config.use_openrouter,
            timeout=augment_config.timeout,
        )

        # Augment events with the prompt
        results = augment_event(
            event_tickers=event_tickers,
            judge=judge,
            prompt=prompt,  # Pass the prompt template
            save_path=None,  # We'll save through state management
            demo=False,
        )

        # Convert results to DataFrame
        # Results are list of (event_ticker, raw_title, augmented_title)
        augmented_events_df = pd.DataFrame(
            results, columns=["event_ticker", "title", "augmented_title"]
        )

        # Count successes and failures
        success_count = (augmented_events_df["augmented_title"] != "").sum()
        self.logger.info(f"Successfully augmented {success_count}/{len(event_tickers)} events")

        state.augmented_events_df = augmented_events_df
        return state

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.augmented_filtered_df is not None:
            return state.augmented_filtered_df["event_ticker"].nunique()
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        if state.augmented_events_df is not None:
            return len(state.augmented_events_df)
        return 0

    def _get_config_snapshot(self) -> dict:
        return {
            "enabled": self.config.augment.event.enabled,
            "model": self.config.augment.event.model,
            "batch_size": self.config.augment.event.batch_size,
        }
