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

    def get_last_completed_stage(self) -> str | None:
        """Get the name of the last successfully completed stage."""
        if not self.stage_history:
            return None
        return self.stage_history[-1].stage_name

    def add_stage_metadata(self, metadata: StageMetadata) -> None:
        """Add metadata for a completed stage."""
        self.stage_history.append(metadata)

    def record_prompt(self, name: str, prompt_dict: dict[str, Any]) -> None:
        """
        Record a prompt used in this run.

        Args:
            name: Name/identifier for the prompt
            prompt_dict: Serialized prompt template (from PromptTemplate.to_dict())
        """
        self.prompts_used[name] = prompt_dict

    def save(self, output_dir: Path, format: str = "parquet") -> None:
        """
        Save pipeline state to disk.

        Args:
            output_dir: Directory to save state files
            format: Format for DataFrames ("parquet" or "pickle")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "run_name": self.run_name,
            "created_at": self.created_at,
            "current_stage": self.current_stage,
            "stage_history": [s.to_dict() for s in self.stage_history],
            "final_dataset_path": self.final_dataset_path,
            "augmented_reasoning_models": list(self.augmented_reasoning_df_map.keys()),
            "prompts_used": self.prompts_used,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save prompts to a separate file for easy inspection
        if self.prompts_used:
            with open(output_dir / "prompts_used.json", "w") as f:
                json.dump(self.prompts_used, f, indent=2)

        # Save DataFrames
        df_attrs = [
            "predictions_df",
            "submissions_df",
            "brier_scores_df",
            "z_score_filtered_df",
            "ambiguous_filtered_df",
            "combined_filtered_df",
            "augmented_filtered_df",
            "augmented_events_df",
        ]

        for attr in df_attrs:
            df = getattr(self, attr)
            if df is not None:
                self._save_dataframe(df, output_dir / f"{attr}.{format}", format)

        # Save augmented reasoning DataFrames
        for model_name, df in self.augmented_reasoning_df_map.items():
            safe_name = model_name.replace("/", "_").replace(":", "_")
            self._save_dataframe(df, output_dir / f"reasoning_{safe_name}.{format}", format)

        logger.info(f"Saved pipeline state to {output_dir}")

    @classmethod
    def load(cls, output_dir: Path) -> "PipelineState":
        """
        Load pipeline state from disk.

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
        format = "parquet" if (output_dir / "predictions_df.parquet").exists() else "pickle"

        # Load DataFrames
        df_attrs = [
            "predictions_df",
            "submissions_df",
            "brier_scores_df",
            "z_score_filtered_df",
            "ambiguous_filtered_df",
            "combined_filtered_df",
            "augmented_filtered_df",
            "augmented_events_df",
        ]

        for attr in df_attrs:
            path = output_dir / f"{attr}.{format}"
            if path.exists():
                setattr(state, attr, cls._load_dataframe(path, format))

        # Load augmented reasoning DataFrames
        for model_name in metadata.get("augmented_reasoning_models", []):
            safe_name = model_name.replace("/", "_").replace(":", "_")
            path = output_dir / f"reasoning_{safe_name}.{format}"
            if path.exists():
                state.augmented_reasoning_df_map[model_name] = cls._load_dataframe(path, format)

        logger.info(f"Loaded pipeline state from {output_dir}")
        return state

    @staticmethod
    def _save_dataframe(df: pd.DataFrame, path: Path, format: str) -> None:
        """Save a DataFrame to disk."""
        if format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_pickle(path)

    @staticmethod
    def _load_dataframe(path: Path, format: str) -> pd.DataFrame:
        """Load a DataFrame from disk."""
        if format == "parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_pickle(path)

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

        return "\n".join(lines)
