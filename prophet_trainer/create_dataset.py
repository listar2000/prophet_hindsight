"""
Convert the list of csv files (with the augmented reasoning traces) into a HuggingFace Dataset (Conversation Style) for SFT.
"""

import json
import logging
import os

import json_repair
import pandas as pd
from datasets import Dataset, DatasetDict

from prophet_hindsight.common.utils import unified_json_loads
from prophet_trainer.prompt import PredictionPrompts

logger = logging.getLogger(__name__)


def concatenate_csv_files(
    csv_files: list[str], exist_strict: bool = True, save_path: str = None
) -> pd.DataFrame:
    """
    Concatenate the list of csv files into a single pandas DataFrame.
    """
    # Step 1: make sure all the files exist, or throw an error
    df_list = []
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            if exist_strict:
                raise FileNotFoundError(f"File {csv_file} does not exist")
            else:
                continue
        df_list.append(pd.read_csv(csv_file))

    # Step 2: concatenate the files into a single pandas DataFrame
    concatenated_df = pd.concat(df_list)
    logger.info(f"Concatenated {len(concatenated_df)} rows from {len(csv_files)} files")

    if save_path is not None:
        concatenated_df.to_parquet(save_path, index=False)
        logger.info(f"Saved concatenated DataFrame to {save_path}")

    return concatenated_df


def message_format(row: pd.Series, conversational: bool = True) -> dict:
    event_title = row["title"]
    market_outcome = unified_json_loads(row["market_outcome"], dict, raise_on_unknown=False)
    if isinstance(market_outcome, dict):
        outcomes_str = ", ".join(list(market_outcome.keys()))
    else:
        outcomes_str = ""
    task_prompt = PredictionPrompts.create_task_prompt().strip()
    user_prompt = PredictionPrompts.create_user_prompt(
        event_title=event_title,
        outcomes_str=outcomes_str,
        sources=row["sources"],
        market_data=row["market_data"],
    ).strip()

    augmented_rationale_dict = row["augmented_rationale"]

    for key in ["source_analysis", "market_analysis", "rationale"]:
        if key not in augmented_rationale_dict:
            logger.warning(
                f"Key {key} not found in augmented_rationale_dict for row {row['event_ticker']}_{row['submission_id']}"
            )
            logger.warning(f"Augmented rationale dict: \n{augmented_rationale_dict}")
            return {}

    prediction = unified_json_loads(row["prediction"], dict, raise_on_unknown=False)
    if isinstance(prediction, dict):
        probabilities = prediction.get("probabilities", {})
    else:
        probabilities = {}
    probabilities_dict = {item["market"]: item["probability"] for item in probabilities}
    probabilities_str = json.dumps(probabilities_dict, indent=2).strip()

    augmented_rationale_str = (
        f"## Source Analysis\n\n{augmented_rationale_dict['source_analysis']}\n\n"
        + f"## Market Analysis\n\n{augmented_rationale_dict['market_analysis']}\n\n"
        + f"## Rationale\n\n{augmented_rationale_dict['rationale']}"
    )

    augmented_rationale_str = (
        f"<think>\n{augmented_rationale_str}\n</think>\n"
        + f"<probabilities>\n{probabilities_str}\n</probabilities>"
    )

    return_dict = {
        "event_ticker": row["event_ticker"],
        "submission_id": row["submission_id"],
        "market_outcome": str(market_outcome),
    }

    if conversational:
        return_dict["messages"] = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": augmented_rationale_str},
        ]
    else:
        return_dict["prompt"] = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return_dict["completion"] = [
            {"role": "assistant", "content": augmented_rationale_str},
        ]

    return return_dict


REPLACE_WORDS = [
    "vanilla rationale",
    "original rationale",
    "vanilla reasoning",
    "original reasoning",
]


def filter_and_replace_rationale(
    rationale_df: pd.DataFrame, replace_word: str = "given context"
) -> pd.DataFrame:
    """
    Post-processing steps for the rationale dataframe:
    1. Remove any rows with NaN values in augmented_rationale.
    2. We filter out any augmented rationale that mentions "leakage" in any field.
    3. We replace any word in the REPLACE_WORDS list with the replace_word.
    """
    # Remove rows with NaN values
    filtered_df = rationale_df.dropna(subset=["augmented_rationale"]).copy()
    # Take a string version of that row
    augmented_rationale_str = filtered_df["augmented_rationale"].apply(str)

    # Filter out rows containing "leakage"
    filtered_df = filtered_df[~augmented_rationale_str.str.contains("leakage", na=False)]

    # Replace words
    def apply_replace_word(row: pd.Series) -> pd.Series:
        # change the "augmented_rationale" field of row to a dictionary if it is a string
        if isinstance(row["augmented_rationale"], str):
            row["augmented_rationale"] = json_repair.loads(row["augmented_rationale"])
        for key in row["augmented_rationale"]:
            for word in REPLACE_WORDS:
                row["augmented_rationale"][key] = row["augmented_rationale"][key].replace(
                    word, replace_word
                )
        return row

    filtered_df = filtered_df.apply(apply_replace_word, axis=1)
    return filtered_df


def pd_train_test_split(
    df: pd.DataFrame, test_size: float = 0.1, seed: int = -1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A pandas-level train/test split (before converting to HuggingFace Dataset)
    The logic is to group by event_ticker first and find a "smallest" subset of event_tickers that contains the test_size proportion of the data.

    If seed is provided (i.e. not -1), instead of going from event_ticker with smallest size, we simply take random event_ticker
    until we have the test_size proportion of the data.
    """
    test_min_size = int(len(df) * test_size)

    # step 1: group by and rank the event_tickers and (optionally) rank by row size if no seed is provided
    event_ticker_sizes = df.groupby("event_ticker").size().reset_index(name="count")  # type: ignore

    if seed != -1:
        # Random selection: shuffle event_tickers
        event_ticker_sizes = event_ticker_sizes.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )
    else:
        # Deterministic selection: sort by size (smallest first)
        event_ticker_sizes = event_ticker_sizes.sort_values("count", ascending=True).reset_index(
            drop=True
        )

    # step 2: maintain a small loop that keeps adding event_tickers to a set
    test_event_tickers = set()
    cumulative_size = 0

    for _, row in event_ticker_sizes.iterrows():
        if cumulative_size >= test_min_size:
            break
        test_event_tickers.add(row["event_ticker"])
        cumulative_size += row["count"]

    logger.info(
        f"Selected {len(test_event_tickers)} event_tickers for test set with {cumulative_size} total rows (target: {test_min_size})"
    )

    # step 3: simply take the train & test sets by the event_ticker in the set or not.
    test_df = df[df["event_ticker"].isin(test_event_tickers)].copy()
    train_df = df[~df["event_ticker"].isin(test_event_tickers)].copy()

    return train_df, test_df


def create_augmented_rationale_sft_dataset(
    rationale_csv_files: list[str],
    prediction_csv_files: list[str],
    save_path: str = None,
    test_size: float = 0.1,
    conversational: bool = False,
    seed: int = 42,
    push_to_hub: bool = False,
    repo_id: str = None,
    private: bool = True,
    n_jobs: int = -1,
) -> DatasetDict:
    """
    Create a HuggingFace Dataset from augmented rationale CSV files.

    Args:
        rationale_csv_files: List of paths to rationale CSV files
        prediction_csv_files: List of paths to prediction CSV files
        save_path: Optional path to save the dataset locally (as Arrow format)
        test_size: Proportion of data to use for test split (default: 0.1)
        conversational: Whether to use conversational format (default: True)
        seed: Random seed for train/test split (default: 42)
        push_to_hub: Whether to push the dataset to HuggingFace Hub (default: False)
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name")
        private: Whether the Hub dataset should be private (default: True)
        n_jobs: Number of parallel jobs for processing (-1 uses all cores, default: -1)

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    combined_rationale_df = concatenate_csv_files(rationale_csv_files, exist_strict=True)
    combined_prediction_df = concatenate_csv_files(prediction_csv_files, exist_strict=True)

    # Step 1: Filter and replace rationale
    combined_rationale_df = filter_and_replace_rationale(combined_rationale_df)
    logger.info(f"After filtering: {len(combined_rationale_df)} rows")

    # Step 1.5: Deduplicate prediction dataframe (keep first occurrence of each key)
    combined_prediction_df = combined_prediction_df.drop_duplicates(
        subset=["event_ticker", "submission_id", "forecaster"], keep="first"
    )
    logger.info(
        f"Deduplicated prediction dataframe: {len(combined_prediction_df)} unique (event_ticker, submission_id, forecaster) triplets"
    )

    # Step 2: Add important columns (e.g. outcomes, market_data, sources) through merging
    combined_df = combined_rationale_df.merge(
        combined_prediction_df[
            [
                "event_ticker",
                "submission_id",
                "forecaster",
                "prediction",
                "market_outcome",
                "market_data",
                "sources",
            ]
        ],
        on=["event_ticker", "submission_id", "forecaster"],
        how="left",
    )
    logger.info(f"After merging: {len(combined_df)} rows")

    # Step 3: Parse augmented_rationale if it's a string (JSON)
    if isinstance(combined_df["augmented_rationale"].iloc[0], str):
        combined_df["augmented_rationale"] = combined_df["augmented_rationale"].apply(
            json_repair.loads
        )

    # Step 4: Pandas train/test split and convert to message format
    train_df, test_df = pd_train_test_split(combined_df, test_size=test_size, seed=seed)
    logger.info(
        f"Converting {len(train_df)} train rows and {len(test_df)} test rows to message format..."
    )

    # Apply message_format and convert to list, filtering out None values
    train_messages = (
        train_df.apply(lambda x: message_format(x, conversational=conversational), axis=1)
        .dropna()
        .tolist()
    )
    test_messages = (
        test_df.apply(lambda x: message_format(x, conversational=conversational), axis=1)
        .dropna()
        .tolist()
    )

    # Step 5: Create HuggingFace Dataset from list of dicts
    train_dataset = Dataset.from_list(train_messages)
    test_dataset = Dataset.from_list(test_messages)
    combined_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    logger.info(
        f"Created train dataset with {len(train_dataset)} examples and test dataset with {len(test_dataset)} examples"
    )

    # Step 7: Save to disk if path provided
    if save_path is not None:
        combined_dataset.save_to_disk(save_path, num_proc=n_jobs)
        logger.info(f"Saved dataset to {save_path}")

    # Step 8: Push to HuggingFace Hub if requested
    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        combined_dataset.push_to_hub(repo_id, private=private)
        logger.info(f"Successfully pushed dataset to {repo_id}")

    return combined_dataset


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rationale_csv_files = [
        "./data/raw/full_data/reasoning/gpt-5-mini/augmented_filtered_reasonings.csv",
        "./data/raw/full_data/reasoning/gpt-5-mini/augmented_hard_reasonings.csv",
        "./data/raw/full_data/reasoning/gemini-2.5-flash/augmented_filtered_reasonings.csv",
        "./data/raw/full_data/reasoning/gemini-2.5-flash/augmented_hard_reasonings.csv",
    ]

    prediction_csv_files = [
        "./data/raw/full_data/reasoning/filtered_predictions.csv",
        "./data/raw/full_data/reasoning/hard_predictions.csv",
    ]

    create_augmented_rationale_sft_dataset(
        rationale_csv_files,
        prediction_csv_files,
        save_path="./data/augmented_sft_conversational",
        test_size=0.1,
        conversational=True,
        seed=123,
        private=False,
        n_jobs=8,
        push_to_hub=True,
        repo_id="listar2000/sports_augmented_sft_1210",
    )
