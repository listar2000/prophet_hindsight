"""
Main entry point for the SFT Data Curation Pipeline.

Usage:
    # Run full pipeline with default config
    python -m prophet_hindsight.pipeline.run_pipeline
    
    # Run with custom config overrides
    python -m prophet_hindsight.pipeline.run_pipeline \\
        run.name=my_run \\
        data.filter_category='["Sports"]' \\
        filter.z_score.min_z=1.25
    
    # Resume from a specific stage
    python -m prophet_hindsight.pipeline.run_pipeline \\
        run.resume_from=reason_augment \\
        run.name=existing_run
        run.new_run_name=existing_run_2
    
    # Run only specific stages
    python -m prophet_hindsight.pipeline.run_pipeline \\
        run.end_at=filtering
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from prophet_hindsight.pipeline.config import PipelineConfig
from prophet_hindsight.pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("Pipeline Configuration:")

    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(cfg_str)

    # Convert Hydra config to PipelineConfig
    config = PipelineConfig.from_hydra(cfg)

    # Create and run pipeline
    runner = PipelineRunner(config)
    state = runner.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(state.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
