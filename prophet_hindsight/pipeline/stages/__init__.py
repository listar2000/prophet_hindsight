"""
Pipeline stages for the SFT Data Curation Pipeline.

Each stage is implemented as a class that inherits from PipelineStage.
"""

from prophet_hindsight.pipeline.stages.base import PipelineStage
from prophet_hindsight.pipeline.stages.data_loading import DataLoadingStage
from prophet_hindsight.pipeline.stages.dataset_creation import DatasetCreationStage
from prophet_hindsight.pipeline.stages.evaluation import EvaluationStage
from prophet_hindsight.pipeline.stages.event_augment import EventAugmentStage
from prophet_hindsight.pipeline.stages.filtering import FilteringStage
from prophet_hindsight.pipeline.stages.reason_augment import ReasonAugmentStage

__all__ = [
    "PipelineStage",
    "DataLoadingStage",
    "EvaluationStage",
    "FilteringStage",
    "EventAugmentStage",
    "ReasonAugmentStage",
    "DatasetCreationStage",
]

# Stage registry for easy lookup
STAGE_REGISTRY = {
    "data_loading": DataLoadingStage,
    "evaluation": EvaluationStage,
    "filtering": FilteringStage,
    "event_augment": EventAugmentStage,
    "reason_augment": ReasonAugmentStage,
    "dataset_creation": DatasetCreationStage,
}

# Ordered list of stages for sequential execution
STAGE_ORDER = [
    "data_loading",
    "evaluation",
    "filtering",
    "event_augment",
    "reason_augment",
    "dataset_creation",
]
