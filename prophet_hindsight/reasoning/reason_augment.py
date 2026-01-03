import asyncio
import logging

import json_repair
import pandas as pd
import tqdm
from pydantic import BaseModel

from prophet_hindsight.common.judge import LLMJudge, MessageBuilder
from prophet_hindsight.common.prompts import PromptTemplate, get_default_reasoning_augment_prompt

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["prediction", "title", "market_data", "sources", "event_ticker"]


class ReasonAugmentResponse(BaseModel):
    source_analysis: str
    market_analysis: str
    rationale: str


def _format_event_info(df_row: pd.Series) -> str:
    market_data = json_repair.loads(df_row["market_data"])
    if isinstance(market_data, dict):
        potential_outcomes_str = list(market_data.keys())
    else:
        potential_outcomes_str = []
    return f"Title: {df_row['title']}\nPotential Outcomes: {potential_outcomes_str}"


def _parse_rationale_and_prediction(raw_prediction: str) -> tuple[str, str]:
    raw_json = json_repair.loads(raw_prediction)
    if isinstance(raw_json, dict):
        return raw_json.get("rationale", ""), raw_json.get("probabilities", "")
    return "", ""


def augment_reasoning(
    judge: LLMJudge,
    reasoning_df: pd.DataFrame,
    augmented_title_df: pd.DataFrame | None = None,
    prompt: PromptTemplate | None = None,
    save_path: str | None = None,
    demo: bool = False,
    timeout: int = 200,
) -> pd.DataFrame:
    """
    Augment the reasoning trace for each row in the dataframe.

    Args:
        judge: LLMJudge instance for making LLM calls
        reasoning_df: DataFrame containing predictions to augment
        augmented_title_df: Optional DataFrame with augmented event titles
        prompt: Optional PromptTemplate to use (defaults to built-in prompt)
        save_path: Optional path to save results
        demo: If True, only process first 10 rows
        timeout: Timeout for async judge calls

    Returns:
        DataFrame with augmented reasoning traces
    """
    # Use default prompt if none provided
    if prompt is None:
        prompt = get_default_reasoning_augment_prompt()

    if demo:
        logger.info("Augmenting reasoning for demo (first 10 rows)")
        reasoning_df = reasoning_df.head(10)
    else:
        logger.info(f"Augmenting reasoning for all {len(reasoning_df)} rows")

    # check for required columns
    if not all(col in reasoning_df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(f"Reasoning dataframe must contain columns: {REQUIRED_COLUMNS}")

    if augmented_title_df is not None:
        # create a merged df so we can get the augmented title for each row in reasoning df by matching on event_ticker
        merged_df = reasoning_df.merge(
            augmented_title_df[["event_ticker", "augmented_title"]], on="event_ticker", how="left"
        )

        is_augmented_title_valid = merged_df["augmented_title"].notna() & (
            merged_df["augmented_title"] != "nan"
        )
        merged_df.loc[is_augmented_title_valid, "title"] = merged_df.loc[
            is_augmented_title_valid, "augmented_title"
        ]
    else:
        merged_df = reasoning_df

    # iteratively construct the prompts
    user_prompts, augmented_reasonings = [], []
    for _, row in merged_df.iterrows():
        event_info = _format_event_info(row)
        sources_data = json_repair.loads(row["sources"])
        if isinstance(sources_data, list):
            sources = "\n".join([str(source) for source in sources_data])
        else:
            sources = str(sources_data)
        market_data = row["market_data"]
        vanilla_rationale, llm_prediction = _parse_rationale_and_prediction(row["prediction"])

        augmented_reasoning_dict = {
            "event_ticker": row["event_ticker"],
            "submission_id": row["submission_id"],
            "title": row["title"],
            "forecaster": row["forecaster"],
            "augmenter": judge.model,
            "original_rationale": vanilla_rationale,
        }
        augmented_reasonings.append(augmented_reasoning_dict)

        # Use the prompt template to format the user prompt
        user_prompt = prompt.format(
            event_info=event_info,
            sources=sources,
            market_data=market_data,
            vanilla_rationale=vanilla_rationale,
            llm_prediction=llm_prediction,
        )
        user_prompts.append(user_prompt)

    completed_results, cancelled_ids = asyncio.run(
        judge.async_judge(
            prompts=user_prompts,
            builder=MessageBuilder(system_prompt=prompt.system_prompt),
            structure=ReasonAugmentResponse,
            ids=list(range(len(user_prompts))),
            timeout=timeout,
        )
    )

    if cancelled_ids:
        logger.warning(f"Failed to get responses for {len(cancelled_ids)} prompts: {cancelled_ids}")

    for i, augmented_reasoning_dict in enumerate(augmented_reasonings):
        if i in completed_results:
            response = completed_results[i]
            augmented_reasoning_dict["augmented_rationale"] = {
                "source_analysis": response.source_analysis,
                "market_analysis": response.market_analysis,
                "rationale": response.rationale,
            }
        else:
            # Mark as failed if not in completed results
            logger.warning(
                f"Skipping augmentation for prompt {i} (event_ticker: {augmented_reasoning_dict.get('event_ticker', 'unknown')})"
            )
            augmented_reasoning_dict["augmented_rationale"] = None

    augmented_reasonings_df = pd.DataFrame(augmented_reasonings)

    if save_path is not None:
        augmented_reasonings_df.to_csv(save_path, index=False)
        # augmented_reasonings_df.to_json(save_path.replace("csv", "json"), orient="index", indent=2)

    return augmented_reasonings_df


def batch_augment_reasoning(
    batch_size: int,
    judge: LLMJudge,
    reasoning_df: pd.DataFrame,
    augmented_title_df: pd.DataFrame | None = None,
    prompt: PromptTemplate | None = None,
    save_path: str | None = None,
    start_from_batch: int = 0,
    timeout: int = 200,
) -> pd.DataFrame | None:
    """
    Batch augment the reasoning for the dataframe. This helps avoid being rate limited by the LLM.

    Args:
        batch_size: Number of rows to process per batch
        judge: LLMJudge instance for making LLM calls
        reasoning_df: DataFrame containing predictions to augment
        augmented_title_df: Optional DataFrame with augmented event titles
        prompt: Optional PromptTemplate to use (defaults to built-in prompt)
        save_path: Optional path to save intermediate results
        start_from_batch: Batch number to start from (for resuming)
        timeout: Timeout for async judge calls

    Returns:
        DataFrame with augmented reasoning traces
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    # Use default prompt if none provided
    if prompt is None:
        prompt = get_default_reasoning_augment_prompt()

    total_size = len(reasoning_df)
    total_batches = (total_size + batch_size - 1) // batch_size  # Ceiling division

    current_df = None
    for i in tqdm.trange(total_batches, desc="Augmenting reasoning batches"):
        if i < start_from_batch:
            continue
        logger.info(f"Processing batch {i+1} of {total_batches}")
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_size)
        batch_df = reasoning_df.iloc[start_idx:end_idx]

        # Process batch with prompt
        batch_result_df = augment_reasoning(
            judge,
            batch_df,
            augmented_title_df,
            prompt=prompt,
            save_path=None,
            demo=False,
            timeout=timeout,
        )

        if current_df is None:
            current_df = batch_result_df
        else:
            current_df = pd.concat([current_df, batch_result_df])

        # Save intermediate results after each batch
        if save_path is not None:
            current_df.to_csv(save_path, index=False)
            # try:
            #     # Reset indices for proper JSON serialization
            #     current_df.index = pd.RangeIndex(start=0, stop=len(current_df))
            #     current_df.to_json(save_path.replace("csv", "json"), orient="index", indent=2)
            # except Exception as e:
            #     logger.error(f"Error saving json for batch {i+1} of {total_batches}: {e}")

    return current_df


if __name__ == "__main__":
    reasoning_df = pd.read_csv("data/raw/full_data/reasoning/filtered_predictions.csv")
    augmented_title_df = pd.read_csv("data/raw/full_data/augmented_event_titles.csv")

    judge = LLMJudge(model="openai/gpt-5-mini", use_async=True, use_openrouter=False)

    # Use default prompt (can also load custom prompt from YAML)
    prompt = get_default_reasoning_augment_prompt()
    print(f"Using prompt: {prompt.name} v{prompt.version}")

    # Example: augment_reasoning with explicit prompt
    # augment_reasoning(judge, reasoning_df, augmented_title_df, prompt=prompt,
    #     save_path="data/raw/full_data/reasoning/demo/gpt-5-mini.csv", demo=True)

    batch_augment_reasoning(
        batch_size=100,
        judge=judge,
        reasoning_df=reasoning_df,
        augmented_title_df=augmented_title_df,
        prompt=prompt,
        save_path="data/raw/full_data/reasoning/gpt-5-mini/augmented_filtered_reasonings.csv",
        start_from_batch=0,
    )
