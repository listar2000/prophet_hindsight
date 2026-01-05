"""
Augmentation strategies for augmenting reasoning traces.

A strategy is a function that takes in a list of LLMs and the full prediction dataframe, and returns an assignment,
which is a dictionary of model name -> DataFrame subset.
"""

from abc import ABC, abstractmethod

import pandas as pd


class AugmentationStrategy(ABC):
    """Abstract base class for augmentation strategies."""

    @abstractmethod
    def assign(self, models: list[str], predictions_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Assign each LLM a subset of predictions."""
        pass


class RepetitiveStrategy(AugmentationStrategy):
    """The strategy that lets all LLMs augment all predictions."""

    def assign(self, models: list[str], predictions_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Assign all models to all predictions."""
        return {model: predictions_df for model in models}


class UniformRandomStrategy(AugmentationStrategy):
    """The strategy that let each LLM augment a random subset of predictions."""

    overlap: bool
    fraction: float
    seed: int

    def __init__(self, overlap: bool = False, fraction: float = 1.0, seed: int = 2025):
        self.overlap = overlap
        self.fraction = fraction
        self.seed = seed

    def assign(self, models: list[str], predictions_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Assign each LLM a random subset of predictions."""
        assert self.fraction >= 0 and self.fraction <= 1, "Fraction must be between 0 and 1"
        # the num of LLMs times the fraction should be <= 1 when overlap is False
        if not self.overlap:
            assert (
                len(models) * self.fraction <= 1
            ), "Fraction must be <= 1 / len(models) when overlap is False"
        # delegate to the helper functions
        return (
            self._assign_no_overlap(models, predictions_df)
            if not self.overlap
            else self._assign_with_overlap(models, predictions_df)
        )

    def _assign_no_overlap(
        self, models: list[str], predictions_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        shuffled_pred_df = predictions_df.sample(frac=1, random_state=self.seed)
        assignments = {}
        # after randomization via shuffling, we just sequentially assign the predictions to the LLMs
        increment = int(len(shuffled_pred_df) * self.fraction)
        for i, model in enumerate(models):
            beg, end = i * increment, (i + 1) * increment
            assignments[model] = shuffled_pred_df.iloc[beg:end]
        return assignments

    def _assign_with_overlap(
        self, models: list[str], predictions_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        assignments = {}
        for model in models:
            assignments[model] = predictions_df.sample(
                frac=self.fraction, replace=False, random_state=self.seed
            )
        return assignments
