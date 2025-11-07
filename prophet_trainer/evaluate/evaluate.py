import pandas as pd
from datasets import load_dataset
import json_repair
import numpy as np


DATASET_NAME = "listar2000/sports_augmented_sft_v2"
SUBMISSION_FILE_PATH = "data/raw/full_data/submissions.csv"


def get_market_outcomes_for_eval_data(
    dataset_name: str = None,
    dataset_split: str = "test",
    max_pred_single_event: int = 5,
    submission_file_path: str = None,
    return_submission_ids: bool = False,
    save_path: str = None
) -> pd.DataFrame: 
    if dataset_name is None:
        dataset_name = DATASET_NAME
    if submission_file_path is None:
        submission_file_path = SUBMISSION_FILE_PATH

    dataset = load_dataset(dataset_name)[dataset_split]
    df = dataset.to_pandas()
    # apply the same filtering logic as when we generated the evaluation prompts
    df = df.groupby("event_ticker").head(max_pred_single_event)
    # return the event tickers and submission ids
    df = df.reset_index(drop=True)[["event_ticker", "submission_id"]]
    print(f"Filtered {len(df)} rows")

    submission_df = pd.read_csv(submission_file_path)
    # we merge on the `submission_id` column and take in the `market_outcome` column
    merged_df = df.merge(submission_df[["submission_id", "market_outcome", "market_data"]], on="submission_id", how="left")
    print(f"Merged {len(merged_df)} rows")

    merged_df["market_outcome"] = merged_df["market_outcome"].apply(json_repair.loads)
    if return_submission_ids:
        merged_df = merged_df[["event_ticker", "submission_id", "market_outcome", "market_data"]]
    else:
        merged_df = merged_df[["market_outcome", "market_data"]]

    if save_path is not None:
        merged_df.to_csv(save_path, index=False)
    return merged_df


def parse_raw_prediction(raw_prediction_str: str) -> dict:
    begin_str = "<probabilities>"
    begin_str_idx = raw_prediction_str.find(begin_str)
    if begin_str_idx == -1:
        return None
    prediction_str = raw_prediction_str[begin_str_idx + len(begin_str):]
    end_str = "</probabilities>"
    end_str_idx = prediction_str.find(end_str)
    if end_str_idx == -1:
        prediction_str = prediction_str.strip()
    else:
        prediction_str = prediction_str[:end_str_idx].strip()
    return json_repair.loads(prediction_str)["probabilities"]

    
def _calculate_brier_score(row: pd.Series) -> float:
    prediction = parse_raw_prediction(row["content"])
    if not prediction:
        return float('nan')

    if isinstance(row["market_outcome"], str):
        market_outcome = json_repair.loads(row["market_outcome"])
    else:
        market_outcome = row["market_outcome"]

    try:
        market_names = [outcome for outcome in market_outcome.keys()]
        prediction_scores = np.array([prediction[market_name] for market_name in market_names]).astype(float)
        market_outcome_scores = np.array([market_outcome[market_name] for market_name in market_names]).astype(float)
        brier_score = np.mean((prediction_scores - market_outcome_scores) ** 2)
    except Exception as e:
        print(f"Error calculating brier score: {e}")
        print("prediction keys: ", prediction.keys())
        print("market outcome keys: ", market_outcome.keys())
        return float('nan')
    return brier_score


def evaluate_with_outcomes_and_outputs(market_outcomes_df: pd.DataFrame, model_outputs_df: pd.DataFrame) -> pd.DataFrame:
    assert len(market_outcomes_df) == len(model_outputs_df)
    # directly do row by row concatenation
    concatenated_df = pd.concat([market_outcomes_df, model_outputs_df], axis=1)
    concatenated_df["brier_score"] = concatenated_df.apply(_calculate_brier_score, axis=1)    
    # count how many valid brier scores we have
    valid_brier_scores = concatenated_df["brier_score"].notna().sum()
    print(f"Number of valid brier scores: {valid_brier_scores}, out of {len(concatenated_df)}")

    # getting summary statistics. No.1: average brier score
    average_brier_score = concatenated_df["brier_score"].mean()
    print(f"Average brier score: {average_brier_score}")

    # No.2: average brier score after first normalizing within each event_ticker
    # which means that we take the average brier score within each event_ticker group first, then take the average of the grouped scores
    grouped_brier_scores = concatenated_df.groupby("event_ticker")["brier_score"].mean()
    normalized_average_brier_score = grouped_brier_scores.mean()
    print(f"Normalized average brier score (averaged by event_ticker): {normalized_average_brier_score}")
    
    # Also print the number of unique events
    num_unique_events = concatenated_df["event_ticker"].nunique()
    print(f"Number of unique event_tickers: {num_unique_events}")

    return concatenated_df


def _calculate_market_brier_score(row: pd.Series) -> np.ndarray:
    market_data = json_repair.loads(row["market_data"])
    market_prediction = {market_name: market_data[market_name]["yes_ask"] / 100.0 for market_name in market_data.keys()}
    if isinstance(row["market_outcome"], str):
        market_outcome = json_repair.loads(row["market_outcome"])
    else:
        market_outcome = row["market_outcome"]
    
    try:
        market_prediction_scores = np.array([market_prediction[market_name] for market_name in market_outcome.keys()]).astype(float)
        market_outcome_scores = np.array([market_outcome[market_name] for market_name in market_outcome.keys()]).astype(float)
        brier_score = np.mean((market_prediction_scores - market_outcome_scores) ** 2)
    except Exception as e:
        print(f"Error calculating market brier score: {e}")
        print("market prediction keys: ", market_prediction.keys())
        print("market outcome keys: ", market_outcome.keys())
        return float('nan')
    return brier_score


def evaluate_market_baseline(market_outcomes_df: pd.DataFrame) -> pd.DataFrame:
    market_outcomes_df["brier_score"] = market_outcomes_df.apply(_calculate_market_brier_score, axis=1)
    valid_brier_scores = market_outcomes_df["brier_score"].notna().sum()
    print(f"Number of valid market brier scores: {valid_brier_scores}, out of {len(market_outcomes_df)}")
    average_brier_score = market_outcomes_df["brier_score"].mean()
    print(f"Average market brier score: {average_brier_score}")
    grouped_brier_scores = market_outcomes_df.groupby("event_ticker")["brier_score"].mean()
    normalized_average_brier_score = grouped_brier_scores.mean()
    print(f"Normalized average market brier score (averaged by event_ticker): {normalized_average_brier_score}")
    num_unique_events = market_outcomes_df["event_ticker"].nunique()
    print(f"Number of unique event_tickers: {num_unique_events}")
    return market_outcomes_df


if __name__ == "__main__":
    # df = get_market_outcomes_for_eval_data(
    #     dataset_name=DATASET_NAME,
    #     dataset_split="test",
    #     max_pred_single_event=5,
    #     return_submission_ids=True,
    #     save_path="data/evals/market_outcomes.csv"
    # )
    # print(df.head())
    market_outcomes_df = pd.read_csv("data/evals/market_outcomes.csv")
    model_outputs_df = pd.read_csv("data/evals/qwen3-8b.csv")
    evaluate_with_outcomes_and_outputs(
        market_outcomes_df=market_outcomes_df,
        model_outputs_df=model_outputs_df
    )
    evaluate_market_baseline(market_outcomes_df=market_outcomes_df)