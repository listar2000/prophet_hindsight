"""
SFT Data Curation Pipeline for Predictive Reasoning.

This package provides a unified, configuration-driven pipeline for:
1. Loading raw prediction data from Prophet Arena
2. Evaluating predictions using Brier scores
3. Filtering high-quality predictions
4. Augmenting event titles and reasoning traces
5. Creating HuggingFace datasets for SFT training
"""

from prophet_hindsight.pipeline.config import PipelineConfig
from prophet_hindsight.pipeline.runner import PipelineRunner
from prophet_hindsight.pipeline.state import PipelineState

__all__ = [
    "PipelineConfig",
    "PipelineState",
    "PipelineRunner",
]
