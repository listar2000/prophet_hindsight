"""
Reasoning Augmentation Stage - Stage 5

Augments reasoning traces to make them longer, more structured, and more detailed.
Uses LLMs to expand the original short rationales into full reasoning traces.

Supports:
- Multiple augmenter models
- Randomized augmenter assignment (to reduce redundancy)
- Batch processing with checkpointing
- Configurable prompts
"""

import logging

import numpy as np
import pandas as pd

from prophet_hindsight.common.prompts import (
    PromptTemplate,
    get_default_reasoning_augment_prompt,
)
from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class ReasonAugmentStage(PipelineStage):
    """Augment reasoning traces using LLM."""

    name = "reason_augment"

    def validate_inputs(self, state: PipelineState) -> bool:
        """Validate that filtered predictions and augmented events are available."""
        if state.augmented_filtered_df is None:
            self.logger.error("augmented_filtered_df is required but not present")
            return False
        if state.augmented_events_df is None:
            self.logger.error("augmented_events_df is required but not present")
            return False
        return True

    def run(self, state: PipelineState) -> PipelineState:
        """
        Augment reasoning traces for all filtered predictions.
        """
        from prophet_hindsight.common.judge import LLMJudge
        from prophet_hindsight.reasoning.reason_augment import batch_augment_reasoning

        augment_config = self.config.augment.reasoning
        prompts_config = self.config.prompts.reasoning_augment

        if not augment_config.enabled:
            self.logger.info("Reasoning augmentation disabled, skipping...")
            return state

        # Load prompt (from config or default)
        if prompts_config.use_default:
            prompt = get_default_reasoning_augment_prompt()
        else:
            prompt = PromptTemplate.from_yaml(prompts_config.custom_path)

        # Record prompt in state for reproducibility
        state.record_prompt("reasoning_augment", prompt.to_dict())
        self.logger.info(
            f"Using prompt: {prompt.name} v{prompt.version} (hash: {prompt.get_hash()})"
        )

        # Get the predictions to augment
        predictions_df = state.augmented_filtered_df.copy()
        augmented_events_df = state.augmented_events_df.copy()

        self.logger.info(f"Augmenting reasoning for {len(predictions_df)} predictions")

        # Determine which models to use for each prediction
        models = augment_config.models

        if augment_config.randomize:
            # Assign models randomly to each prediction
            predictions_with_models = self._assign_models_randomly(
                predictions_df,
                models,
                augment_config.augment_ratio,
                seed=self.config.run.seed,
            )
        else:
            # Use all models for all predictions
            predictions_with_models = [(predictions_df, model) for model in models]

        # Process each model
        for df, model in predictions_with_models:
            if len(df) == 0:
                self.logger.info(f"No predictions assigned to {model}, skipping...")
                continue

            self.logger.info(f"Augmenting {len(df)} predictions with {model}")

            judge = LLMJudge(
                model=model,
                use_async=True,
                use_openrouter=augment_config.use_openrouter,
                timeout=augment_config.timeout,
            )

            # Use batch processing with the prompt
            augmented_df = batch_augment_reasoning(
                batch_size=augment_config.batch_size,
                judge=judge,
                reasoning_df=df,
                augmented_title_df=augmented_events_df,
                prompt=prompt,  # Pass the prompt template
                save_path=None,  # We'll save through state management
                start_from_batch=0,
            )

            state.augmented_reasoning_dfs[model] = augmented_df

            # Count successes
            if "augmented_rationale" in augmented_df.columns:
                success_count = augmented_df["augmented_rationale"].notna().sum()
                self.logger.info(f"Successfully augmented {success_count}/{len(df)} with {model}")

        return state

    def _assign_models_randomly(
        self,
        predictions_df: pd.DataFrame,
        models: list[str],
        augment_ratio: float,
        seed: int,
    ) -> list[tuple]:
        """
        Randomly assign models to predictions.

        Each prediction is assigned to approximately augment_ratio * len(models) models.

        Returns:
            List of (DataFrame subset, model name) tuples
        """
        np.random.seed(seed)

        n_predictions = len(predictions_df)

        # For each prediction, randomly decide which models to use
        # Each model has augment_ratio probability of being selected
        assignments = {model: [] for model in models}

        for idx in range(n_predictions):
            for model in models:
                if np.random.random() < augment_ratio:
                    assignments[model].append(idx)

        # Create DataFrames for each model
        result = []
        for model in models:
            indices = assignments[model]
            if indices:
                df_subset = predictions_df.iloc[indices].copy()
                result.append((df_subset, model))
            else:
                result.append((pd.DataFrame(), model))

        return result

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.augmented_filtered_df is not None:
            return len(state.augmented_filtered_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        return sum(len(df) for df in state.augmented_reasoning_dfs.values())

    def _get_config_snapshot(self) -> dict:
        return {
            "enabled": self.config.augment.reasoning.enabled,
            "models": self.config.augment.reasoning.models,
            "randomize": self.config.augment.reasoning.randomize,
            "augment_ratio": self.config.augment.reasoning.augment_ratio,
            "batch_size": self.config.augment.reasoning.batch_size,
        }
