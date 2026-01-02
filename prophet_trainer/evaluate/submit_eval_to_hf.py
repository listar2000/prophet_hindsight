import pandas as pd
from datasets import Dataset, DatasetDict

sft_df = pd.read_csv("data/evals/qwen3-8b-sft.csv")
raw_df = pd.read_csv("data/evals/qwen3-8b.csv")
market_outcomes_df = pd.read_csv("data/evals/market_outcomes.csv")

# we take the event_ticker and submission_id columns from the market_outcomes_df and concatenate (row by row)
# with the sft_df and raw_df, respectively
sft_concat_df = pd.concat([market_outcomes_df[["event_ticker", "submission_id"]], sft_df], axis=1)
raw_concat_df = pd.concat([market_outcomes_df[["event_ticker", "submission_id"]], raw_df], axis=1)

# Load the CSV files
sft_dataset = Dataset.from_pandas(sft_concat_df)
raw_dataset = Dataset.from_pandas(raw_concat_df)

# Create a DatasetDict with both splits
dataset_dict = DatasetDict({"sft": sft_dataset, "raw": raw_dataset})

# Push to HuggingFace Hub (replace 'your-username/your-repo-name' with your actual repo)
dataset_dict.push_to_hub("listar2000/qwen3-8b-model-outputs")
