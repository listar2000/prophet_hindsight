"""
Configuration models for the SFT/RL Data Curation Pipeline.

Uses Pydantic for type validation and OmegaConf/Hydra for configuration management.
Supports both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) pipelines.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Literal


class PipelineType(str, Enum):
    """Type of pipeline to run."""

    SFT = "sft"  # Supervised Fine-Tuning: full pipeline with reasoning augmentation
    RL = "rl"  # Reinforcement Learning: prompt-only, no reasoning augmentation


class StageType(str, Enum):
    """Enumeration of pipeline stages (SFT pipeline)."""

    DATA_LOADING = "data_loading"
    EVALUATION = "evaluation"
    FILTERING = "filtering"
    EVENT_AUGMENT = "event_augment"
    REASON_AUGMENT = "reason_augment"
    DATASET_CREATION = "dataset_creation"

    @classmethod
    def all_stages(cls) -> list["StageType"]:
        """Return all stages in execution order (SFT)."""
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


class RLStageType(str, Enum):
    """Enumeration of pipeline stages for RL pipeline."""

    DATA_LOADING = "data_loading"
    EVALUATION = "evaluation"
    RL_SELECTION = "rl_selection"  # Replaces filtering for RL
    EVENT_AUGMENT = "event_augment"
    RL_DATASET_CREATION = "rl_dataset_creation"  # RL-specific dataset creation

    @classmethod
    def all_stages(cls) -> list["RLStageType"]:
        """Return all stages in execution order (RL)."""
        return [
            cls.DATA_LOADING,
            cls.EVALUATION,
            cls.RL_SELECTION,
            cls.EVENT_AUGMENT,
            cls.RL_DATASET_CREATION,
        ]

    @classmethod
    @cached_property
    def get_stage_orders(cls) -> dict["RLStageType", int]:
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
    # Pipeline type: "sft" or "rl"
    pipeline_type: str = "sft"


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
    """Complete filtering configuration (SFT pipeline)."""

    metric_col: str = "brier_score"
    z_score: ZScoreFilterConfig = field(default_factory=ZScoreFilterConfig)
    ambiguous: AmbiguousFilterConfig = field(default_factory=AmbiguousFilterConfig)
    absolute: AbsoluteFilterConfig = field(default_factory=AbsoluteFilterConfig)


@dataclass
class RLSelectionConfig:
    """RL data selection configuration.

    Unlike SFT filtering which selects high-quality traces, RL selection:
    - Deduplicates problems (one per (event_ticker, submission_id) pair)
    - Excludes test set problems
    - Does NOT filter by quality (we want more diverse data for RL)
    """

    # Path to test set CSV with event_ticker and submission_id columns to exclude
    test_set_path: str | None = None
    # Strategy for selecting which prediction to keep when multiple exist
    # Options: "first", "random", "best_brier" (lowest Brier score)
    dedup_strategy: str = "random"
    # Maximum number of problems to select (-1 = no limit)
    max_problems: int = -1
    # Whether to shuffle the final selection (useful for RL training)
    shuffle: bool = True


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
    start_from_batch: int = 0
    timeout: int = 200
    existing_augmented_reasoning_df: str | None = None
    strategy: dict = field(default_factory=dict)


@dataclass
class AugmentConfig:
    """Complete augmentation configuration."""

    event: EventAugmentConfig = field(default_factory=EventAugmentConfig)
    reasoning: ReasoningAugmentConfig = field(default_factory=ReasoningAugmentConfig)


@dataclass
class DatasetConfig:
    """Dataset creation configuration (SFT pipeline)."""

    test_size: float = 0.1
    conversational: bool = True
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = True
    n_jobs: int = 8


@dataclass
class RLDatasetConfig:
    """Dataset creation configuration for RL pipeline.

    Unlike SFT which includes reasoning traces, RL datasets contain only:
    - system prompt
    - user prompt (event info, sources, market data)
    No assistant response is included.
    """

    # Test/train split size (for validation during RL training)
    test_size: float = 0.1
    # Whether to use conversational format (messages list)
    conversational: bool = True
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = True
    n_jobs: int = 8


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    enabled: bool = True
    format: Literal["parquet", "csv"] = "parquet"
    # whether to skip saving the original raw data
    skip_raw_data: bool = False


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
    # RL-specific dataset config
    rl_dataset: RLDatasetConfig = field(default_factory=RLDatasetConfig)


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    This is the top-level configuration that contains all sub-configurations.
    It can be constructed from a Hydra DictConfig or programmatically.
    Supports both SFT and RL pipelines via run.pipeline_type.
    """

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # SFT-specific filtering config
    filter: FilterConfig = field(default_factory=FilterConfig)
    # RL-specific selection config
    rl_selection: RLSelectionConfig = field(default_factory=RLSelectionConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

    def is_rl_pipeline(self) -> bool:
        """Check if this is an RL pipeline configuration."""
        return self.run.pipeline_type == "rl"

    def is_sft_pipeline(self) -> bool:
        """Check if this is an SFT pipeline configuration."""
        return self.run.pipeline_type == "sft"

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
        rl_selection_cfg = cfg_dict.get("rl_selection", {}) or {}
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
            rl_selection=RLSelectionConfig(**rl_selection_cfg),
            augment=AugmentConfig(
                event=EventAugmentConfig(**augment_cfg.get("event", {})),
                reasoning=ReasoningAugmentConfig(**augment_cfg.get("reasoning", {})),
            ),
            output=OutputConfig(
                base_dir=output_cfg.get("base_dir", "data/runs"),
                dataset=DatasetConfig(**output_cfg.get("dataset", {})),
                checkpoints=CheckpointConfig(**output_cfg.get("checkpoints", {})),
                rl_dataset=RLDatasetConfig(**output_cfg.get("rl_dataset", {})),
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
