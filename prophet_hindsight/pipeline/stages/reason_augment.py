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
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate

from prophet_hindsight.common.prompts import (
    PromptTemplate,
    get_default_reasoning_augment_prompt,
)
from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.state import PipelineState
from prophet_hindsight.reasoning.augment_strategies import AugmentationStrategy

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
            prompt = PromptTemplate.from_yaml(prompts_config.custom_path)  # type: ignore[arg-type]

        # Record prompt in state for reproducibility
        state.record_prompt("reasoning_augment", prompt.to_dict())
        self.logger.info(
            f"Using prompt: {prompt.name} v{prompt.version} (hash: {prompt.get_hash()})"
        )

        # Get the predictions to augment
        predictions_df = state.augmented_filtered_df.copy()  # type: ignore
        augmented_events_df = state.augmented_events_df.copy()  # type: ignore

        self.logger.info(f"Augmenting reasoning for {len(predictions_df)} predictions")

        # Determine which models to use for each prediction
        models = augment_config.models

        # Instantiate the augmentation strategy and assign models to predictions
        augment_strategy: AugmentationStrategy = instantiate(augment_config.strategy)  # type: ignore[arg-type]
        predictions_with_models = augment_strategy.assign(models, predictions_df)

        # Process each model
        for model, df in predictions_with_models.items():
            self.current_model = model

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
                start_from_batch=augment_config.start_from_batch,
                save_partial_callback=self._save_partial_callback,
            )

            if augmented_df is None or len(augmented_df) == 0:
                self.logger.warning(f"No augmented reasoning traces for {model}, skipping...")
                continue

            state.augmented_reasoning_df_map[model] = augmented_df
            # Count successes
            success_count = augmented_df["augmented_rationale"].notna().sum()
            self.logger.info(f"Successfully augmented {success_count}/{len(df)} with {model}")

        return state

    def _save_partial_callback(self, augmented_df: pd.DataFrame, batch_idx: int):
        # Same logic as in the PipelineState._save_dataframe method
        format = self.config.output.checkpoints.format
        safe_name = self.current_model.replace("/", "_").replace(":", "_")
        save_path = (
            Path(self.config.get_run_dir()) / "reason_augment" / f"reasoning_{safe_name}.{format}"
        )

        if format == "parquet":
            augmented_df.to_parquet(save_path, index=False)
        else:
            augmented_df.to_csv(save_path, index=False)

        self.logger.info(
            f"Model {self.current_model}; Batch {batch_idx}: Saved {len(augmented_df)} augmented reasoning traces to {save_path}"
        )

    def _count_input_rows(self, state: PipelineState) -> int:
        if state.augmented_filtered_df is not None:
            return len(state.augmented_filtered_df)
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        return sum(len(df) for df in state.augmented_reasoning_df_map.values())

    def _get_config_snapshot(self) -> dict:
        return {
            "enabled": self.config.augment.reasoning.enabled,
            "models": self.config.augment.reasoning.models,
            "batch_size": self.config.augment.reasoning.batch_size,
            "strategy": self.config.augment.reasoning.strategy,
        }
