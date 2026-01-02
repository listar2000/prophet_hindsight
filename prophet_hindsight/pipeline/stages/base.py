"""
Abstract base class for pipeline stages.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from prophet_hindsight.pipeline.config import PipelineConfig
from prophet_hindsight.pipeline.state import PipelineState, StageMetadata

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.

    Each stage must implement:
    - name: A unique identifier for the stage
    - run(): The main execution logic
    - validate_inputs(): Check that required inputs are present
    """

    name: str = "base_stage"

    def __init__(self, config: PipelineConfig):
        """
        Initialize the stage with configuration.

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the stage and return updated state.

        Args:
            state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        pass

    @abstractmethod
    def validate_inputs(self, state: PipelineState) -> bool:
        """
        Validate that required inputs are present in state.

        Args:
            state: Current pipeline state

        Returns:
            True if inputs are valid, False otherwise
        """
        pass

    def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute the stage with timing and error handling.

        This method wraps the run() method with:
        - Input validation
        - Timing
        - Error handling
        - Metadata recording

        Args:
            state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        self.logger.info(f"Starting stage: {self.name}")
        state.current_stage = self.name

        # Validate inputs
        if not self.validate_inputs(state):
            raise ValueError(f"Input validation failed for stage: {self.name}")

        # Get input row count for metadata
        input_rows = self._count_input_rows(state)

        # Execute with timing
        started_at = datetime.now()

        try:
            state = self.run(state)
        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            raise

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        # Get output row count
        output_rows = self._count_output_rows(state)

        # Record metadata
        metadata = StageMetadata(
            stage_name=self.name,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration,
            input_rows=input_rows,
            output_rows=output_rows,
            config_snapshot=self._get_config_snapshot(),
        )
        state.add_stage_metadata(metadata)

        self.logger.info(
            f"Completed stage: {self.name} in {duration:.1f}s ({input_rows} -> {output_rows} rows)"
        )
        return state

    def _count_input_rows(self, state: PipelineState) -> int:
        """Count input rows for this stage. Override in subclasses."""
        return 0

    def _count_output_rows(self, state: PipelineState) -> int:
        """Count output rows from this stage. Override in subclasses."""
        return 0

    def _get_config_snapshot(self) -> dict:
        """Get relevant config for this stage. Override in subclasses."""
        return {}
