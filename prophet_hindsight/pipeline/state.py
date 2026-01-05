"""
Pipeline state management.

The PipelineState class tracks all intermediate artifacts between pipeline stages,
enabling checkpoint/resume functionality and observability.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping of stage names to their output DataFrame attributes
# This determines which attributes are saved in which stage subdirectory
STAGE_OUTPUTS: dict[str, list[str]] = {
    "data_loading": ["predictions_df", "submissions_df"],
    "evaluation": ["brier_scores_df"],
    "filtering": [
        "z_score_filtered_df",
        "ambiguous_filtered_df",
        "combined_filtered_df",
        "augmented_filtered_df",
    ],
    "event_augment": ["augmented_events_df"],
    "reason_augment": ["augmented_reasoning_df_map"],
    "dataset_creation": ["final_dataset_path"],
}

# Flat list of all DataFrame attributes (excluding special ones like augmented_reasoning_df_map)
DF_ATTRS = [
    "predictions_df",
    "submissions_df",
    "brier_scores_df",
    "z_score_filtered_df",
    "ambiguous_filtered_df",
    "combined_filtered_df",
    "augmented_filtered_df",
    "augmented_events_df",
]


@dataclass
class StageMetadata:
    """Metadata for a completed pipeline stage."""

    stage_name: str
    started_at: str
    completed_at: str
    duration_seconds: float
    input_rows: int = 0
    output_rows: int = 0
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "config_snapshot": self.config_snapshot,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageMetadata":
        """Construct from dictionary."""
        return cls(**data)


@dataclass
class PipelineState:
    """
    Holds all intermediate data between pipeline stages.

    This class is the central state container for the pipeline. It tracks:
    - DataFrames produced by each stage
    - Metadata about stage execution
    - Prompts used in each stage
    - Errors and warnings

    The state can be serialized to disk and loaded to resume a pipeline run.
    """

    # Run identification
    run_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Stage 1: Data Loading
    predictions_df: pd.DataFrame | None = None
    submissions_df: pd.DataFrame | None = None

    # Stage 2: Evaluation
    brier_scores_df: pd.DataFrame | None = None

    # Stage 3: Filtering
    z_score_filtered_df: pd.DataFrame | None = None
    ambiguous_filtered_df: pd.DataFrame | None = None
    combined_filtered_df: pd.DataFrame | None = None
    augmented_filtered_df: pd.DataFrame | None = None  # With sources, context, etc.

    # Stage 4: Event Augmentation
    augmented_events_df: pd.DataFrame | None = None

    # Stage 5: Reasoning Augmentation
    # Maps augmenter model name -> augmented reasoning DataFrame
    augmented_reasoning_df_map: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Stage 6: Dataset Creation
    # Note: HuggingFace DatasetDict is not stored here, only the path
    final_dataset_path: str | None = None

    # Prompts used in this run (for reproducibility)
    # Maps prompt name -> prompt dict (serialized PromptTemplate)
    prompts_used: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    stage_history: list[StageMetadata] = field(default_factory=list)
    current_stage: str | None = None

    # Session tracking (not persisted)
    # Tracks which stages have been run in the current session for smart saving
    _stages_run_in_session: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        # Ensure _stages_run_in_session is always a set
        if not isinstance(self._stages_run_in_session, set):
            self._stages_run_in_session = set()

    def mark_stage_run(self, stage_name: str) -> None:
        """Mark a stage as run in the current session."""
        self._stages_run_in_session.add(stage_name)

    def get_stages_run_in_session(self) -> set[str]:
        """Get the set of stages run in the current session."""
        return self._stages_run_in_session

    def get_last_completed_stage(self) -> str | None:
        """Get the name of the last successfully completed stage."""
        if not self.stage_history:
            return None
        return self.stage_history[-1].stage_name

    def add_stage_metadata(self, metadata: StageMetadata) -> None:
        """
        Add or update metadata for a completed stage.

        If the stage was already run (exists in stage_history), its metadata
        is replaced with the new metadata. Otherwise, it's appended.
        """
        # Mark this stage as run in the current session
        self.mark_stage_run(metadata.stage_name)

        # Check if this stage already exists in history
        existing_idx = None
        for i, existing in enumerate(self.stage_history):
            if existing.stage_name == metadata.stage_name:
                existing_idx = i
                break

        if existing_idx is not None:
            # Replace existing metadata for this stage
            self.stage_history[existing_idx] = metadata
            logger.debug(f"Replaced stage metadata for {metadata.stage_name}")
        else:
            # Append new stage metadata
            self.stage_history.append(metadata)
            logger.debug(f"Added stage metadata for {metadata.stage_name}")

    def record_prompt(self, name: str, prompt_dict: dict[str, Any]) -> None:
        """
        Record a prompt used in this run.

        Args:
            name: Name/identifier for the prompt
            prompt_dict: Serialized prompt template (from PromptTemplate.to_dict())
        """
        self.prompts_used[name] = prompt_dict

    def save(
        self,
        output_dir: Path,
        format: str = "parquet",
        skip_raw_data: bool = False,
        stages_to_save: set[str] | None = None,
    ) -> None:
        """
        Save pipeline state to disk with stage-based directory organization.

        Args:
            output_dir: Directory to save state files
            format: Format for DataFrames ("parquet" or "csv")
            skip_raw_data: Whether to skip saving the original raw data
            stages_to_save: Optional set of stage names to save. If None, saves
                           only stages that were run in the current session.
                           If empty set, saves nothing. Use {"all"} to force save all.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which stages to save
        if stages_to_save is None:
            # Default: save only stages run in current session
            stages_to_save = self._stages_run_in_session
        elif "all" in stages_to_save:
            # Force save all stages that have data
            stages_to_save = set(STAGE_OUTPUTS.keys())

        logger.info(f"Saving state for stages: {stages_to_save}")

        # Save metadata (always)
        self._save_metadata(output_dir)

        # Save prompts (always, if any)
        if self.prompts_used:
            with open(output_dir / "prompts_used.json", "w") as f:
                json.dump(self.prompts_used, f, indent=2)

        # Save DataFrames organized by stage
        for stage_name in stages_to_save:
            if stage_name not in STAGE_OUTPUTS:
                continue
            elif stage_name == "data_loading" and skip_raw_data:
                logger.info("Skipping saving raw data after data loading stage")
                continue

            stage_dir = output_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)

            for attr in STAGE_OUTPUTS[stage_name]:
                if attr == "augmented_reasoning_df_map":
                    # Special handling for the map of reasoning DataFrames
                    for model_name, df in self.augmented_reasoning_df_map.items():
                        safe_name = model_name.replace("/", "_").replace(":", "_")
                        self._save_dataframe(
                            df, stage_dir / f"reasoning_{safe_name}.{format}", format
                        )
                elif attr == "final_dataset_path":
                    # Dataset path is just stored in metadata, not as a file here
                    pass
                else:
                    # Regular DataFrame attribute
                    df = getattr(self, attr)
                    if df is not None:
                        logger.info(f"Saving {attr} to {stage_dir}")
                        try:
                            self._save_dataframe(df, stage_dir / f"{attr}.{format}", format)
                        except Exception as e:
                            logger.error(f"Error saving dataframe {attr}: {e}")
                            logger.info(f"Columns of dataframe {attr}: {df.columns}")
                            raise e

        logger.info(f"Saved pipeline state to {output_dir}")

    def _save_metadata(self, output_dir: Path) -> None:
        """Save metadata.json with run information."""
        metadata = {
            "run_name": self.run_name,
            "created_at": self.created_at,
            "current_stage": self.current_stage,
            "stage_history": [s.to_dict() for s in self.stage_history],
            "final_dataset_path": self.final_dataset_path,
            "augmented_reasoning_models": list(self.augmented_reasoning_df_map.keys()),
            "prompts_used": self.prompts_used,
            # Track which stages have data (for loading)
            "stages_with_data": self._get_stages_with_data(),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_stages_with_data(self) -> list[str]:
        """Get list of stages that have data saved."""
        stages = []
        for stage_name, attrs in STAGE_OUTPUTS.items():
            has_data = False
            for attr in attrs:
                if attr == "augmented_reasoning_df_map":
                    if self.augmented_reasoning_df_map:
                        has_data = True
                elif attr == "final_dataset_path":
                    if self.final_dataset_path:
                        has_data = True
                else:
                    if getattr(self, attr) is not None:
                        has_data = True
            if has_data:
                stages.append(stage_name)
        return stages

    @classmethod
    def load(cls, output_dir: Path) -> "PipelineState":
        """
        Load pipeline state from disk.

        Supports both new stage-based directory structure and legacy flat structure.

        Args:
            output_dir: Directory containing state files

        Returns:
            Loaded PipelineState instance
        """
        output_dir = Path(output_dir)

        # Load metadata
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        state = cls(
            run_name=metadata["run_name"],
            created_at=metadata["created_at"],
            current_stage=metadata.get("current_stage"),
            final_dataset_path=metadata.get("final_dataset_path"),
            stage_history=[StageMetadata.from_dict(s) for s in metadata.get("stage_history", [])],
            prompts_used=metadata.get("prompts_used", {}),
        )

        # Detect format from existing files
        state._load_from_stage_dirs(output_dir)

        logger.info(f"Loaded pipeline state from {output_dir}")
        return state

    def _load_from_stage_dirs(self, output_dir: Path) -> None:
        """Load DataFrames from stage-based directory structure."""
        for stage_name, attrs in STAGE_OUTPUTS.items():
            stage_dir = output_dir / stage_name
            if not stage_dir.exists():
                continue

            for attr in attrs:
                if attr == "augmented_reasoning_df_map":
                    # Load all reasoning_* files
                    for path in stage_dir.glob("reasoning_*.*"):
                        # Extract model name from filename
                        model_name = (
                            path.stem.replace("reasoning_", "").replace(
                                "_", "/", 1
                            )  # Restore first slash
                        )
                        self.augmented_reasoning_df_map[model_name] = self._load_dataframe(path)
                elif attr == "final_dataset_path":
                    # Already loaded from metadata
                    pass
                else:
                    match = list(stage_dir.glob(f"{attr}.*"))
                    if match:
                        path = match[0]
                        setattr(self, attr, self._load_dataframe(path))

    @staticmethod
    def _save_dataframe(df: pd.DataFrame, path: Path | str, format: str | None = None) -> None:
        if format is None:
            format = str(path).split(".")[-1]
        if format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    @staticmethod
    def _load_dataframe(path: Path | str, format: str | None = None) -> pd.DataFrame:
        if format is None:
            format = str(path).split(".")[-1]
        if format == "parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)

    def summary(self) -> str:
        """Generate a human-readable summary of the pipeline state."""
        lines = [
            f"Pipeline State: {self.run_name}",
            f"Created: {self.created_at}",
            f"Current Stage: {self.current_stage or 'Not started'}",
            "",
            "Data Summary:",
        ]

        if self.predictions_df is not None:
            lines.append(f"  - Predictions: {len(self.predictions_df)} rows")
        if self.submissions_df is not None:
            lines.append(f"  - Submissions: {len(self.submissions_df)} rows")
        if self.brier_scores_df is not None:
            lines.append(f"  - Brier Scores: {len(self.brier_scores_df)} rows")
        if self.combined_filtered_df is not None:
            lines.append(f"  - Filtered Predictions: {len(self.combined_filtered_df)} rows")
        if self.augmented_events_df is not None:
            lines.append(f"  - Augmented Events: {len(self.augmented_events_df)} rows")
        if self.augmented_reasoning_df_map:
            total = sum(len(df) for df in self.augmented_reasoning_df_map.values())
            lines.append(
                f"  - Augmented Reasonings: {total} rows across {len(self.augmented_reasoning_df_map)} models"
            )

        if self.stage_history:
            lines.extend(["", "Stage History:"])
            for stage in self.stage_history:
                lines.append(
                    f"  - {stage.stage_name}: {stage.duration_seconds:.1f}s ({stage.input_rows} -> {stage.output_rows} rows)"
                )

        if self._stages_run_in_session:
            lines.extend(["", f"Stages run this session: {self._stages_run_in_session}"])

        return "\n".join(lines)
