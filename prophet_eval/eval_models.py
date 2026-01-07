import subprocess
import signal
import time
import asyncio
import os
import yaml
import pandas as pd
from pathlib import Path

from sglang.utils import wait_for_server

from prophet_eval.eval import load_hf_dataset, batch_eval


MODEL_MAPPING = {
    "Qwen3-8B": {
        "model_dir": ".cache/torchtune/Qwen3-8B",
        "dataset": "listar2000/full_augmented_sft",
        "split": "test"
    }, 
    "Qwen3-8B-SFT-v1": {
        "model_dir": ".cache/torchtune/Qwen3-8B-Prophet-Forecast-SFT-2-Epochs/epoch_0",
        "dataset": "listar2000/full_augmented_sft",
        "split": "test"
    }, 
    "Qwen3-8B-SFT-v2": {
        "model_dir": ".cache/torchtune/Qwen3-8B-Prophet-Forecast-SFT-2-Epochs/epoch_1",
        "dataset": "listar2000/full_augmented_sft",
        "split": "test"
    }, 
    "Qwen3-8B-SFT-TOP-Z": {
        "model_dir": ".cache/torchtune/Qwen3-8B-Prophet-Forecast-SFT-Top-Z/epoch_0",
        "dataset": "listar2000/top_z_only_augmented_sft",
        "split": "test"
    }, 
}

# Path to server config
SERVER_CONFIG_PATH = Path(__file__).parent / "server_args.yaml"


def load_server_config() -> dict:
    """Load the server configuration from server_args.yaml"""
    with open(SERVER_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_server_command(model_path: str) -> list[str]:
    """Build the command to launch the sglang server using config file"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--config", str(SERVER_CONFIG_PATH),
        "--model-path", model_path,  # Override model path from config
    ]
    return cmd


def launch_server(model_path: str) -> subprocess.Popen:
    """
    Launch the sglang server as a subprocess and wait for it to be ready.
    
    Returns:
        The subprocess.Popen object for the server process
    """
    config = load_server_config()
    cmd = build_server_command(model_path)
    print(f"Launching server with command:\n{' '.join(cmd)}\n")
    
    # Start the server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create a new process group for clean termination
    )
    
    port = config.get("port", 30000)
    server_url = f"http://localhost:{port}"
    
    print(f"Waiting for server to be ready at {server_url}...")
    try:
        wait_for_server(server_url, timeout=600)  # 10 minute timeout for model loading
        print(f"Server is ready at {server_url}")
    except Exception as e:
        print(f"Server failed to start: {e}")
        terminate_server(process)
        raise
    
    return process


def terminate_server(process: subprocess.Popen, timeout: int = 30):
    """
    Gracefully terminate the server process and all child processes.
    
    Args:
        process: The subprocess.Popen object for the server
        timeout: Seconds to wait for graceful shutdown before force killing
    """
    if process.poll() is not None:
        print("Server process already terminated.")
        return
    
    print("Terminating server...")
    
    try:
        # Send SIGTERM to the entire process group for graceful shutdown
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        # Wait for graceful termination
        process.wait(timeout=timeout)
        print("Server terminated gracefully.")
    except subprocess.TimeoutExpired:
        print("Graceful shutdown timed out, force killing...")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        print("Server force killed.")
    except Exception as e:
        print(f"Error terminating server: {e}")
        # Try force kill as last resort
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass


async def evaluate_model(
    model_name: str,
    model_config: dict,
    output_dir: str = "evals",
    debug_mode: bool = False,
):
    """
    Evaluate a single model: launch server, run evaluation, save results, terminate server.
    
    Args:
        model_name: Name of the model (used for output file naming)
        model_config: Config dict with model_dir, dataset, split
        output_dir: Directory to save evaluation results
        debug_mode: If True, only evaluate first 10 samples
    """
    model_dir = model_config["model_dir"]
    dataset_name = model_config["dataset"]
    split = model_config["split"]
    
    # Create output filename (replace spaces with hyphens, lowercase)
    output_filename = model_name.lower().replace(" ", "-") + ".csv"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"Dataset: {dataset_name} (split: {split})")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Launch the server
    server_process = launch_server(model_dir)
    
    try:
        # Load the dataset
        ds = load_hf_dataset(dataset_name, split)
        df = pd.DataFrame(ds)
        
        if debug_mode:
            print("DEBUG MODE: Only evaluating first 10 samples")
            df = df.head(10)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation - use model_dir as the model name for the API
        await batch_eval(model_dir, df, output_path, result_col="prediction")
        
        print(f"\nEvaluation complete for {model_name}. Results saved to {output_path}")
        
    finally:
        # Always terminate the server, even if evaluation fails
        terminate_server(server_process)
        
        # Give some time for GPU memory to be released
        print("Waiting for GPU memory to be released...")
        time.sleep(10)


async def evaluate_all_models(
    models: dict | None = None,
    output_dir: str = "evals",
    debug_mode: bool = False,
):
    """
    Evaluate all models in the MODEL_MAPPING (or a subset).
    
    Args:
        models: Optional dict of models to evaluate. If None, uses MODEL_MAPPING
        output_dir: Directory to save evaluation results
        debug_mode: If True, only evaluate first 10 samples per model
    """
    if models is None:
        models = MODEL_MAPPING
    
    total_models = len(models)
    
    for idx, (model_name, model_config) in enumerate(models.items(), 1):
        print(f"\n{'#'*60}")
        print(f"# Model {idx}/{total_models}: {model_name}")
        print(f"{'#'*60}")
        
        try:
            await evaluate_model(
                model_name=model_name,
                model_config=model_config,
                output_dir=output_dir,
                debug_mode=debug_mode,
            )
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            print("Continuing with next model...\n")
            continue
    
    print(f"\n{'='*60}")
    print("All model evaluations complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multiple trained models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model names to evaluate (if not provided, evaluates all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals",
        help="Directory to save evaluation results (default: evals)"
    )
    parser.add_argument(
        "--debug-mode", "--dm",
        action="store_true",
        help="Run in debug mode (only evaluate first 10 samples per model)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, config in MODEL_MAPPING.items():
            print(f"  - {name}")
            print(f"      model_dir: {config['model_dir']}")
            print(f"      dataset: {config['dataset']}")
            print(f"      split: {config['split']}")
            # check whether the model_dir exists
            if not os.path.exists(config['model_dir']):
                print(f"      [WARNING] model_dir does not exist: {config['model_dir']}")
        exit(0)
    
    # Filter models if specific ones are requested
    if args.models:
        selected_models = {}
        for model_name in args.models:
            if model_name not in MODEL_MAPPING:
                print(f"Warning: Model '{model_name}' not found in MODEL_MAPPING, skipping...")
                continue
            selected_models[model_name] = MODEL_MAPPING[model_name]
        
        if not selected_models:
            print("No valid models selected. Use --list-models to see available models.")
            exit(1)
    else:
        selected_models = None  # Will use all models
    
    asyncio.run(evaluate_all_models(
        models=selected_models,
        output_dir=args.output_dir,
        debug_mode=args.debug_mode,
    ))
