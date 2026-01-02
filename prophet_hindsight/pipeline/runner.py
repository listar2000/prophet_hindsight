"""
Pipeline Runner - Orchestrates the execution of all pipeline stages.

Supports:
- Full pipeline execution
- Resume from checkpoint
- Stage-specific runs
- Comprehensive logging
"""

import logging
from datetime import datetime
from pathlib import Path

from prophet_hindsight.pipeline.config import PipelineConfig
from prophet_hindsight.pipeline.stages import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    PipelineStage,
)
from prophet_hindsight.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Orchestrates the execution of all pipeline stages.

    Usage:
        config = PipelineConfig.from_hydra(cfg)
        runner = PipelineRunner(config)
        state = runner.run()
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline runner.

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.stages: list[PipelineStage] = self._create_stages()

        # Setup logging
        self._setup_logging()

    def _create_stages(self) -> list[PipelineStage]:
        """Create stage instances in execution order."""
        stages = []
        for stage_name in STAGE_ORDER:
            stage_class = STAGE_REGISTRY[stage_name]
            stages.append(stage_class(self.config))
        return stages

    def _setup_logging(self) -> None:
        """Setup logging to console and file."""
        run_dir = Path(self.config.get_run_dir())
        run_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = run_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        # Configure root logger for pipeline
        pipeline_logger = logging.getLogger("prophet_hindsight.pipeline")
        pipeline_logger.setLevel(logging.DEBUG)
        pipeline_logger.addHandler(file_handler)
        pipeline_logger.addHandler(console_handler)

    def run(self, state: PipelineState | None = None) -> PipelineState:
        """
        Execute the pipeline.

        Args:
            state: Optional existing state to resume from

        Returns:
            Final pipeline state
        """
        run_config = self.config.run
        run_dir = Path(self.config.get_run_dir())

        # Initialize or load state
        if state is None:
            if run_config.resume_from:
                # Resume from checkpoint
                state = self._load_checkpoint(run_dir)
                logger.info(f"Resuming from checkpoint: {state.get_last_completed_stage()}")
            else:
                # Start fresh
                state = PipelineState(
                    run_name=run_config.name or datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                logger.info(f"Starting new pipeline run: {state.run_name}")

        # Determine stages to run
        stages_to_run = self._get_stages_to_run(state)

        if not stages_to_run:
            logger.info("No stages to run")
            return state

        logger.info(f"Running stages: {[s.name for s in stages_to_run]}")

        # Execute stages
        for stage in stages_to_run:
            try:
                state = stage.execute(state)

                # Save checkpoint after each stage
                if self.config.output.checkpoints.enabled:
                    self._save_checkpoint(state, run_dir)

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                # Save state even on failure for debugging
                self._save_checkpoint(state, run_dir)
                raise

        logger.info("Pipeline completed successfully")
        logger.info(state.summary())

        return state

    def _get_stages_to_run(self, state: PipelineState) -> list[PipelineStage]:
        """
        Determine which stages to run based on config and state.

        Considers:
        - resume_from: Start from this stage
        - end_at: Stop after this stage
        - skip_stages: Skip these stages
        - Last completed stage in state
        """
        run_config = self.config.run

        # Find start index
        start_idx = 0
        if run_config.resume_from:
            # Start from specified stage
            try:
                start_idx = STAGE_ORDER.index(run_config.resume_from)
            except ValueError:
                raise ValueError(f"Unknown stage: {run_config.resume_from}")
        elif state.get_last_completed_stage():
            # Resume from next stage after last completed
            last_stage = state.get_last_completed_stage()
            try:
                start_idx = STAGE_ORDER.index(last_stage) + 1
            except ValueError:
                start_idx = 0

        # Find end index
        end_idx = len(STAGE_ORDER)
        if run_config.end_at:
            try:
                end_idx = STAGE_ORDER.index(run_config.end_at) + 1
            except ValueError:
                raise ValueError(f"Unknown stage: {run_config.end_at}")

        # Get stages in range
        stages_to_run = []
        for i in range(start_idx, end_idx):
            stage = self.stages[i]
            if stage.name not in run_config.skip_stages:
                stages_to_run.append(stage)

        return stages_to_run

    def _save_checkpoint(self, state: PipelineState, run_dir: Path) -> None:
        """Save pipeline state checkpoint."""
        checkpoint_format = self.config.output.checkpoints.format
        state.save(run_dir, format=checkpoint_format)
        logger.debug(f"Saved checkpoint to {run_dir}")

    def _load_checkpoint(self, run_dir: Path) -> PipelineState:
        """Load pipeline state from checkpoint."""
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")
        return PipelineState.load(run_dir)

    def run_stage(self, stage_name: str, state: PipelineState | None = None) -> PipelineState:
        """
        Run a single stage.

        Args:
            stage_name: Name of the stage to run
            state: Optional existing state

        Returns:
            Updated pipeline state
        """
        if stage_name not in STAGE_REGISTRY:
            raise ValueError(f"Unknown stage: {stage_name}")

        # Initialize state if needed
        if state is None:
            state = PipelineState(
                run_name=self.config.run.name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        # Find and execute stage
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.execute(state)

        raise ValueError(f"Stage not found: {stage_name}")
