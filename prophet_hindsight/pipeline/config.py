"""
Configuration models for the SFT Data Curation Pipeline.

Uses Pydantic for type validation and OmegaConf/Hydra for configuration management.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Literal


class StageType(str, Enum):
    """Enumeration of pipeline stages."""

    DATA_LOADING = "data_loading"
    EVALUATION = "evaluation"
    FILTERING = "filtering"
    EVENT_AUGMENT = "event_augment"
    REASON_AUGMENT = "reason_augment"
    DATASET_CREATION = "dataset_creation"

    @classmethod
    def all_stages(cls) -> list["StageType"]:
        """Return all stages in execution order."""
        return [
            cls.DATA_LOADING,
            cls.EVALUATION,
            cls.FILTERING,
            cls.EVENT_AUGMENT,
            cls.REASON_AUGMENT,
            cls.DATASET_CREATION,
        ]

    @classmethod
    @cached_property
    def get_stage_orders(cls) -> dict["StageType", int]:
        stages = cls.all_stages()
        return {stage: i for i, stage in enumerate(stages)}


@dataclass
class RunConfig:
    """Run-level configuration."""

    name: str | None = None
    seed: int = 42
    resume_from: str | None = None
    new_run_name: str | None = None
    end_at: str | None = None
    skip_stages: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Data loading configuration."""

    db_url: str | None = None
    filter_time_before: str = "2025-10-23 00:00:00"
    filter_time_after: str = "2025-10-10 00:00:00"
    filter_agent_only: bool = True
    filter_contributor_only: list[str] | None = None
    filter_category: list[str] | None = None
    predictions_path: str | None = None
    submissions_path: str | None = None


@dataclass
class ZScoreFilterConfig:
    """Z-score filtering configuration."""

    enabled: bool = True
    min_z: float = 1.5
    max_mean: float = -1
    min_val: float = 0.15


@dataclass
class AmbiguousFilterConfig:
    """Ambiguous event filtering configuration."""

    enabled: bool = True
    min_val: float = 0.25
    max_val: float = 0.15
    top_k: int = 5


@dataclass
class AbsoluteFilterConfig:
    """Absolute score filtering configuration."""

    enabled: bool = False
    top_k: int = -1
    top_p: float = 0.1
    min_val: float = -1


@dataclass
class FilterConfig:
    """Complete filtering configuration."""

    metric_col: str = "brier_score"
    z_score: ZScoreFilterConfig = field(default_factory=ZScoreFilterConfig)
    ambiguous: AmbiguousFilterConfig = field(default_factory=AmbiguousFilterConfig)
    absolute: AbsoluteFilterConfig = field(default_factory=AbsoluteFilterConfig)


@dataclass
class EventAugmentConfig:
    """Event title augmentation configuration."""

    enabled: bool = True
    model: str = "openai/gpt-5-mini"
    use_openrouter: bool = False
    batch_size: int = 100
    timeout: int = 180


@dataclass
class ReasoningAugmentConfig:
    """Reasoning trace augmentation configuration."""

    enabled: bool = True
    models: list[str] = field(default_factory=lambda: ["openai/gpt-5-mini"])
    use_openrouter: bool = False
    batch_size: int = 100
    timeout: int = 200
    strategy: dict = field(default_factory=dict)


@dataclass
class AugmentConfig:
    """Complete augmentation configuration."""

    event: EventAugmentConfig = field(default_factory=EventAugmentConfig)
    reasoning: ReasoningAugmentConfig = field(default_factory=ReasoningAugmentConfig)


@dataclass
class DatasetConfig:
    """Dataset creation configuration."""

    test_size: float = 0.1
    conversational: bool = True
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = True
    n_jobs: int = 8


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    enabled: bool = True
    format: Literal["parquet", "pickle"] = "parquet"


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""

    use_default: bool = True
    custom_path: str | None = None


@dataclass
class PromptsConfig:
    """Complete prompts configuration."""

    reasoning_augment: PromptConfig = field(default_factory=PromptConfig)
    event_augment: PromptConfig = field(default_factory=PromptConfig)
    prediction: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class OutputConfig:
    """Output configuration."""

    base_dir: str = "data/runs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    This is the top-level configuration that contains all sub-configurations.
    It can be constructed from a Hydra DictConfig or programmatically.
    """

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

    @classmethod
    def from_hydra(cls, cfg) -> "PipelineConfig":
        """
        Construct PipelineConfig from a Hydra DictConfig.

        Args:
            cfg: Hydra DictConfig object

        Returns:
            PipelineConfig instance
        """
        from omegaconf import OmegaConf

        # Convert DictConfig to dict and instantiate dataclasses
        cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

        # Safely extract nested configs with defaults
        run_cfg = cfg_dict.get("run", {}) or {}
        data_cfg = cfg_dict.get("data", {}) or {}
        filter_cfg = cfg_dict.get("filter", {}) or {}
        augment_cfg = cfg_dict.get("augment", {}) or {}
        output_cfg = cfg_dict.get("output", {}) or {}

        return cls(
            run=RunConfig(**run_cfg),
            data=DataConfig(**data_cfg),
            filter=FilterConfig(
                metric_col=filter_cfg.get("metric_col", "brier_score"),
                z_score=ZScoreFilterConfig(**filter_cfg.get("z_score", {})),
                ambiguous=AmbiguousFilterConfig(**filter_cfg.get("ambiguous", {})),
                absolute=AbsoluteFilterConfig(**filter_cfg.get("absolute", {})),
            ),
            augment=AugmentConfig(
                event=EventAugmentConfig(**augment_cfg.get("event", {})),
                reasoning=ReasoningAugmentConfig(**augment_cfg.get("reasoning", {})),
            ),
            output=OutputConfig(
                base_dir=output_cfg.get("base_dir", "data/runs"),
                dataset=DatasetConfig(**output_cfg.get("dataset", {})),
                checkpoints=CheckpointConfig(**output_cfg.get("checkpoints", {})),
            ),
            prompts=cls._parse_prompts_config(cfg_dict.get("prompts", {})),
        )

    @classmethod
    def _parse_prompts_config(cls, prompts_cfg: dict) -> PromptsConfig:
        """Parse prompts configuration from dict."""

        def parse_prompt_config(cfg: dict | None) -> PromptConfig:
            if cfg is None:
                return PromptConfig()
            return PromptConfig(
                use_default=cfg.get("use_default", True),
                custom_path=cfg.get("custom_path"),
            )

        return PromptsConfig(
            reasoning_augment=parse_prompt_config(prompts_cfg.get("reasoning_augment")),
            event_augment=parse_prompt_config(prompts_cfg.get("event_augment")),
            prediction=parse_prompt_config(prompts_cfg.get("prediction")),
        )

    def get_run_dir(self) -> str:
        """Get the run directory path."""
        import os
        from datetime import datetime

        run_name = self.run.name
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        return os.path.join(self.output.base_dir, run_name)
