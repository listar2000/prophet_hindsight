"""
Pipeline stages for the SFT/RL Data Curation Pipeline.

Each stage is implemented as a class that inherits from PipelineStage.
Supports both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) pipelines.
"""

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.stages.data_loading import DataLoadingStage
from prophet_hindsight.pipeline.stages.dataset_creation import DatasetCreationStage
from prophet_hindsight.pipeline.stages.evaluation import EvaluationStage
from prophet_hindsight.pipeline.stages.event_augment import EventAugmentStage
from prophet_hindsight.pipeline.stages.filtering import FilteringStage
from prophet_hindsight.pipeline.stages.reason_augment import ReasonAugmentStage
from prophet_hindsight.pipeline.stages.rl_dataset_creation import RLDatasetCreationStage
from prophet_hindsight.pipeline.stages.rl_selection import RLSelectionStage

__all__ = [
    "PipelineStage",
    # Shared stages
    "DataLoadingStage",
    "EvaluationStage",
    "EventAugmentStage",
    # SFT-specific stages
    "FilteringStage",
    "ReasonAugmentStage",
    "DatasetCreationStage",
    # RL-specific stages
    "RLSelectionStage",
    "RLDatasetCreationStage",
]

# Stage registry for SFT pipeline
STAGE_REGISTRY = {
    "data_loading": DataLoadingStage,
    "evaluation": EvaluationStage,
    "filtering": FilteringStage,
    "event_augment": EventAugmentStage,
    "reason_augment": ReasonAugmentStage,
    "dataset_creation": DatasetCreationStage,
}

# Stage registry for RL pipeline
RL_STAGE_REGISTRY = {
    "data_loading": DataLoadingStage,
    "evaluation": EvaluationStage,
    "rl_selection": RLSelectionStage,
    "event_augment": EventAugmentStage,
    "rl_dataset_creation": RLDatasetCreationStage,
}

# Ordered list of stages for SFT pipeline
STAGE_ORDER = [
    "data_loading",
    "evaluation",
    "filtering",
    "event_augment",
    "reason_augment",
    "dataset_creation",
]

# Ordered list of stages for RL pipeline
RL_STAGE_ORDER = [
    "data_loading",
    "evaluation",
    "rl_selection",
    "event_augment",
    "rl_dataset_creation",
]


def get_stage_registry(pipeline_type: str = "sft") -> dict:
    """Get the appropriate stage registry based on pipeline type."""
    if pipeline_type == "rl":
        return RL_STAGE_REGISTRY
    return STAGE_REGISTRY


def get_stage_order(pipeline_type: str = "sft") -> list:
    """Get the appropriate stage order based on pipeline type."""
    if pipeline_type == "rl":
        return RL_STAGE_ORDER
    return STAGE_ORDER
