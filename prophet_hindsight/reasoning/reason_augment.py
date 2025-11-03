from prophet_hindsight.reasoning.prompt import REASONING_AUGMENT_SYSTEM_PROMPT, REASONING_AUGMENT_USER_PROMPT
from prophet_hindsight.common.judge import LLMJudge, MessageBuilder
from pydantic import BaseModel
import pandas as pd
import json_repair
import logging
import asyncio

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ['prediction', 'title', 'market_data', 'sources', 'event_ticker']


class ReasonAugmentResponse(BaseModel):
    source_analysis: str
    market_analysis: str
    rationale: str


def _format_event_info(df_row: pd.Series) -> str:
    potential_outcomes_str = list(json_repair.loads(df_row['market_data']).keys())
    return f"Title: {df_row['title']}\nPotential Outcomes: {potential_outcomes_str}"


def _parse_rationale_and_prediction(raw_prediction: str) -> tuple[str, str]:
    raw_json = json_repair.loads(raw_prediction)
    return raw_json['rationale'], raw_json['probabilities']


def augment_reasoning(judge: LLMJudge, reasoning_df: pd.DataFrame, augmented_title_df: pd.DataFrame = None, save_path: str = None, demo: bool = False) -> pd.DataFrame:
    """
    Augment the reasoning trace for each row in the dataframe.
    """
    if demo:
        logger.info("Augmenting reasoning for demo (first 10 rows)")
        reasoning_df = reasoning_df.head(1)
    else:
        logger.info(f"Augmenting reasoning for all {len(reasoning_df)} rows")

    # check for required columns
    if not all(col in reasoning_df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(f"Reasoning dataframe must contain columns: {REQUIRED_COLUMNS}")
    
    if augmented_title_df is not None:
        # create a merged df so we can get the augmented title for each row in reasoning df by matching on event_ticker
        merged_df = reasoning_df.merge(augmented_title_df[['event_ticker', 'augmented_title']], on='event_ticker', how='left')

        is_augmented_title_valid = merged_df['augmented_title'].notna() & (merged_df['augmented_title'] != "nan")
        merged_df.loc[is_augmented_title_valid, 'title'] = merged_df.loc[is_augmented_title_valid, 'augmented_title']
    else:
        merged_df = reasoning_df

    # iteratively construct the prompts
    prompts, augmented_reasonings = [], []
    for _, row in merged_df.iterrows():
        event_info = _format_event_info(row)
        sources = json_repair.loads(row['sources'])
        sources = "\n".join([str(source) for source in sources])
        market_data = row['market_data']
        vanilla_rationale, llm_prediction = _parse_rationale_and_prediction(row['prediction'])

        augmented_reasoning_dict = {
            'event_ticker': row['event_ticker'],
            'submission_id': row['submission_id'],
            'title': row['title'],
            'forecaster': row['forecaster'],
            'augmenter': judge.model,
            'original_rationale': vanilla_rationale,
        }
        augmented_reasonings.append(augmented_reasoning_dict)

        prompt = REASONING_AUGMENT_USER_PROMPT.format(event_info=event_info, sources=sources, market_data=market_data, vanilla_rationale=vanilla_rationale, llm_prediction=llm_prediction)
        prompts.append(prompt)

    async_responses = asyncio.run(judge.async_judge(
        prompts=prompts,
        builder=MessageBuilder(system_prompt=REASONING_AUGMENT_SYSTEM_PROMPT),
        structure=ReasonAugmentResponse,
    ))

    for i, response in enumerate(async_responses):
        augmented_reasonings[i]['augmented_rationale'] = {
            'source_analysis': response.source_analysis,
            'market_analysis': response.market_analysis,
            'rationale': response.rationale,
        }

    augmented_reasonings_df = pd.DataFrame(augmented_reasonings)

    if save_path is not None:
        augmented_reasonings_df.to_csv(save_path, index=False)
        augmented_reasonings_df.to_json(save_path.replace("csv", "json"), orient="index", indent=2)

    return augmented_reasonings_df


def batch_augment_reasoning(batch_size: int, judge: LLMJudge, reasoning_df: pd.DataFrame, augmented_title_df: pd.DataFrame = None, save_path: str = None, start_from_batch: int = 0) -> pd.DataFrame:
    """
    Batch augment the reasoning for the dataframe. This helps avoid being rate limited by the LLM.
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0") 
    
    total_size = len(reasoning_df)
    total_batches = total_size // batch_size + 1

    current_df = None
    for i in range(total_batches):
        if i < start_from_batch - 1:
            continue
        print(f"Processing batch {i+1} of {total_batches}")
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_size)
        batch_df = reasoning_df.iloc[start_idx:end_idx]
        # we hijack this operation to avoid overwritting the result
        batch_result_df = augment_reasoning(judge, batch_df, augmented_title_df, save_path=None, demo=False)
        if current_df is None:
            current_df = batch_result_df
        else:
            current_df = pd.concat([current_df, batch_result_df])

        # save the current df
        if save_path is not None:
            current_df.to_csv(save_path, index=False)
            # set the indices of the current df to be from start_idx to end_idx
            try:
                # reset the indices to 0 to len - 1
                current_df.index = pd.RangeIndex(start=0, stop=len(current_df))
                current_df.to_json(save_path.replace("csv", "json"), orient="index", indent=2)
            except Exception as e:
                logger.error(f"Error saving json for batch {i+1} of {total_batches}: {e}")

    return current_df


def csv_to_json_helper(csv_path: str, json_path: str = None) -> None:
    if not json_path:
        json_path = csv_path.replace("csv", "json")
    
    df = pd.read_csv(csv_path)
    df['augmented_rationale'] = df['augmented_rationale'].astype(str).apply(json_repair.loads)
    df.to_json(json_path, orient="index", indent=2)


if __name__ == "__main__":
    # reasoning_df = pd.read_csv("data/raw/full_data/reasoning/filtered_predictions.csv")
    # augmented_title_df = pd.read_csv("data/raw/full_data/augmented_event_titles.csv")

    # judge = LLMJudge(model="openai/gpt-5-mini", use_async=True, use_openrouter=False)

    # batch_augment_reasoning(100, judge, reasoning_df, augmented_title_df, \
    #     save_path="data/raw/full_data/reasoning/gpt-5-mini/augmented_filtered_reasonings_2.csv", start_from_batch=2)

    csv_to_json_helper("data/raw/full_data/reasoning/gpt-5-mini/augmented_filtered_reasonings.csv")