import datasets
import argparse
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm

from openai import AsyncOpenAI
import os


DEFAULT_PORT = 30000
SAVE_COLS = ["event_ticker", "submission_id", "market_outcome"]

client = None
use_openrouter_flag = False


def load_hf_dataset(dataset_name: str, split: str = "test") -> datasets.Dataset:
    ds = datasets.load_dataset(dataset_name, split=split)
    assert "messages" in ds.column_names, "Dataset must have a 'messages' column"
    assert all(col in ds.column_names for col in SAVE_COLS), "Dataset must have the following columns: " + ", ".join(SAVE_COLS)
    print(f"Loaded {len(ds)} rows from {dataset_name} split {split} with columns:\n{ds.column_names}\n")
    return ds


async def get_response(model_name: str, task_idx: int, messages: list[dict]):
    # ref: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    if use_openrouter_flag:
        extra_kwargs = {"reasoning": {"effort": "medium"}}

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
        max_tokens=8192,
        extra_body=extra_kwargs,
    )
    return task_idx, response.choices[0].message
    

async def batch_eval(model_name: str, df: pd.DataFrame, save_path: str, result_col: str = "prediction"):
    results, tasks = [], []
    for i, row in df.iterrows():
        messages = row["messages"]
        assert len(messages) == 3 and messages[-1]["role"] == "assistant", "Last message must be the assistant message"
        # remove the last message to get prediction
        messages = messages[:-1]
        task = asyncio.create_task(get_response(model_name, i, messages))
        tasks.append(task)
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        task_idx, response = await future
        og_row = df.iloc[task_idx]
        results.append({
            "event_ticker": og_row["event_ticker"],
            "submission_id": og_row["submission_id"],
            "market_outcome": og_row["market_outcome"],
            result_col: {
                "content": response.content,
                "reasoning": response.reasoning_content if not use_openrouter_flag else getattr(response, "reasoning", None),
            },
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    return results_df


if __name__ == "__main__":
    """
    Example usage (sglang local):
    python -m prophet_eval.eval \
        --model-name .cache/torchtune/Qwen3-8B-Prophet-Forecast-SFT-2-Epochs/epoch_0 \
        --dataset-name listar2000/full_augmented_sft \
        --split test \
        --save-path evals/qwen3-8b-sft-epoch-0/full-in-2-epochs.csv \
        --result-col prediction
    
    Example usage (openrouter):
    python -m prophet_eval.eval \
        --model-name x-ai/grok-4 \
        --dataset-name listar2000/full_augmented_sft \
        --split test \
        --save-path evals/grok-4/full.csv \
        --result-col prediction \
        --use-openrouter \
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--dataset-name", type=str, default="listar2000/top_z_only_augmented_sft")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--debug-mode", "--dm", action="store_true", help="Run in debug mode")
    parser.add_argument("--result-col", type=str, default="prediction", help="Column name to save the results to")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the results to")
    parser.add_argument("--use-openrouter", action="store_true", help="Use OpenRouter to evaluate the model")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to use for the local sglang/vllm server")
    args = parser.parse_args()

    # handle the case where we use use openrouter for inference
    if args.use_openrouter:
        if os.getenv("OPENROUTER_API_KEY") is None:
            api_key = input("Enter your OpenRouter API key: ")
            os.environ["OPENROUTER_API_KEY"] = api_key
        use_openrouter_flag = True
        client = AsyncOpenAI(base_url=f"https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    else:  # use local sglang server
        client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="None")

    ds = load_hf_dataset(args.dataset_name, args.split)
    df = pd.DataFrame(ds)
    if args.debug_mode:
        df = df.head(10)

    # make dir if the save path doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    asyncio.run(batch_eval(args.model_name, df, args.save_path, args.result_col))