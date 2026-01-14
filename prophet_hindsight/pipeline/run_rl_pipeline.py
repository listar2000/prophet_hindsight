"""
Main entry point for the RL Data Curation Pipeline.

The RL pipeline differs from SFT in that:
- It does not filter by quality (uses all available data)
- It deduplicates to one problem per (event_ticker, submission_id)
- It excludes test set problems if specified
- It creates prompt-only datasets (no assistant responses)

Usage:
    # Run full RL pipeline with default config
    python -m prophet_hindsight.pipeline.run_rl_pipeline
    
    # Run with custom config overrides
    python -m prophet_hindsight.pipeline.run_rl_pipeline \\
        run.name=my_rl_run \\
        data.filter_category='["Sports"]' \\
        rl_selection.test_set_path="data/test_set.csv"
    
    # Resume from a specific stage
    python -m prophet_hindsight.pipeline.run_rl_pipeline \\
        run.resume_from=rl_selection \\
        run.name=existing_run \\
        run.new_run_name=existing_run_2
    
    # Limit number of problems
    python -m prophet_hindsight.pipeline.run_rl_pipeline \\
        rl_selection.max_problems=5000
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from prophet_hindsight.pipeline.config import PipelineConfig
from prophet_hindsight.pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config_rl")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the RL pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("RL Pipeline Configuration:")

    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(cfg_str)

    # Convert Hydra config to PipelineConfig
    config = PipelineConfig.from_hydra(cfg)

    # Verify this is an RL pipeline
    if not config.is_rl_pipeline():
        logger.warning(
            f"Expected RL pipeline but got pipeline_type={config.run.pipeline_type}. "
            "Forcing pipeline_type to 'rl'."
        )
        config.run.pipeline_type = "rl"

    # Create and run pipeline
    runner = PipelineRunner(config)
    state = runner.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("RL PIPELINE COMPLETED")
    print("=" * 60)
    print(state.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
