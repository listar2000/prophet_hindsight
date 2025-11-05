"""
Convert the list of csv files (with the augmented reasoning traces) into a HuggingFace Dataset (Conversation Style) for SFT.
"""
from prophet_trainer.prompt import PredictionPrompts

import pandas as pd
import os
from datasets import Dataset, DatasetDict
import logging
import json_repair
from tqdm import tqdm
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)


def concatenate_csv_files(csv_files: list[str], exist_strict: bool = True, save_path: str = None) -> pd.DataFrame:
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


def message_format(row: pd.Series) -> dict:
    event_title = row["title"]
    outcomes = list(json_repair.loads(row["market_outcome"]).keys())
    task_prompt = PredictionPrompts.create_task_prompt(event_title, outcomes)
    user_prompt = PredictionPrompts.create_user_prompt(row["sources"], row["market_data"])

    augmented_rationale_dict = row["augmented_rationale"]
    
    for key in ["source_analysis", "market_analysis", "rationale"]:
        if key not in augmented_rationale_dict:
            logger.warning(f"Key {key} not found in augmented_rationale_dict for row {row['event_ticker']}_{row['submission_id']}")
            logger.warning(f"Augmented rationale dict: \n{augmented_rationale_dict}")
            return None

    augmented_rationale_str = f"<source_analysis>\n{augmented_rationale_dict['source_analysis']}</source_analysis>" + \
        f"<market_analysis>\n{augmented_rationale_dict['market_analysis']}</market_analysis>" + \
        f"<rationale>\n{augmented_rationale_dict['rationale']}</rationale>"

    return {
        "event_ticker": row["event_ticker"],
        "submission_id": row["submission_id"],
        "messages": [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": augmented_rationale_str},
        ]
    }


REPLACE_WORDS = ["vanilla rationale", "original rationale", "vanilla reasoning", "original reasoning"]

def filter_and_replace_rationale(rationale_df: pd.DataFrame, replace_word: str = "given context") -> pd.DataFrame:
    """
    Post-processing steps for the rationale dataframe:
    1. Remove any rows with NaN values in augmented_rationale.
    2. We filter out any augmented rationale that mentions "leakage" in any field.
    3. We replace any word in the REPLACE_WORDS list with the replace_word.
    """
    # Remove rows with NaN values
    filtered_df = rationale_df.dropna(subset=["augmented_rationale"]).copy()
    
    # Filter out rows containing "leakage"
    filtered_df = filtered_df[~filtered_df["augmented_rationale"].str.contains("leakage", na=False)]
    
    # Replace words
    for word in REPLACE_WORDS:
        filtered_df["augmented_rationale"] = filtered_df["augmented_rationale"].str.replace(word, replace_word, regex=False)
    return filtered_df


def create_augmented_rationale_sft_dataset(
    rationale_csv_files: list[str], 
    prediction_csv_files: list[str],
    save_path: str = None,
    test_size: float = 0.1,
    seed: int = 42,
    push_to_hub: bool = False,
    repo_id: str = None,
    private: bool = True,
    n_jobs: int = -1
) -> DatasetDict:
    """
    Create a HuggingFace Dataset from augmented rationale CSV files.
    
    Args:
        rationale_csv_files: List of paths to rationale CSV files
        prediction_csv_files: List of paths to prediction CSV files
        save_path: Optional path to save the dataset locally (as Arrow format)
        test_size: Proportion of data to use for test split (default: 0.1)
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
    combined_prediction_df = combined_prediction_df.drop_duplicates(subset=["event_ticker", "submission_id"], keep="first")
    logger.info(f"Deduplicated prediction dataframe: {len(combined_prediction_df)} unique (event_ticker, submission_id) pairs")

    # Step 2: Add important columns (e.g. outcomes, market_data, sources) through merging
    combined_df = combined_rationale_df.merge(
        combined_prediction_df[["event_ticker", "submission_id", "market_outcome", "market_data", "sources"]], 
        on=["event_ticker", "submission_id"], 
        how="left"
    )
    logger.info(f"After merging: {len(combined_df)} rows")
    
    # Step 3: Parse augmented_rationale if it's a string (JSON)
    if isinstance(combined_df["augmented_rationale"].iloc[0], str):
        combined_df["augmented_rationale"] = combined_df["augmented_rationale"].apply(json_repair.loads)
    
    # Step 4: Convert to message format (parallel processing)
    logger.info(f"Converting {len(combined_df)} rows to message format using {n_jobs if n_jobs > 0 else 'all available'} cores...")
    
    # Convert dataframe rows to list of Series for parallel processing
    rows = [row for _, row in combined_df.iterrows()]
    
    # Process in parallel with progress bar
    messages_list = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(message_format)(row) for row in tqdm(rows, desc="Processing rows")
    )
    
    # Step 5: Create HuggingFace Dataset
    dataset = Dataset.from_list([message for message in messages_list if message is not None])
    logger.info(f"Created dataset with {len(dataset)} examples (removed {len(messages_list) - len(dataset)} rows with None values)")
    
    # Step 6: Split into train/test
    dataset_dict = dataset.train_test_split(test_size=test_size, seed=seed)
    logger.info(f"Split dataset into train ({len(dataset_dict['train'])}) and test ({len(dataset_dict['test'])})")
    
    # Step 7: Save to disk if path provided
    if save_path is not None:
        dataset_dict.save_to_disk(save_path, num_proc=n_jobs)
        logger.info(f"Saved dataset to {save_path}")
    
    # Step 8: Push to HuggingFace Hub if requested
    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        dataset_dict.push_to_hub(repo_id, private=private)
        logger.info(f"Successfully pushed dataset to {repo_id}")
    
    return dataset_dict


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    rationale_csv_files = [
        "./data/raw/full_data/reasoning/gpt-5-mini/augmented_filtered_reasonings.csv",
        "./data/raw/full_data/reasoning/gpt-5-mini/augmented_hard_reasonings.csv",
        "./data/raw/full_data/reasoning/gemini-2.5-flash/augmented_filtered_reasonings.csv",
        "./data/raw/full_data/reasoning/gemini-2.5-flash/augmented_hard_reasonings.csv"
    ]

    prediction_csv_files = [
        "./data/raw/full_data/reasoning/filtered_predictions.csv",
        "./data/raw/full_data/reasoning/hard_predictions.csv",
    ]

    create_augmented_rationale_sft_dataset(
        rationale_csv_files, 
        prediction_csv_files, 
        save_path="./data/train",
        test_size=0.1,
        seed=42,
        private=False,
        n_jobs=8, 
        push_to_hub=True, 
        repo_id="listar2000/sports_augmented_sft"
    )