<div align="center" id="sglangtop">
<img src="./assets/prophet_hindsight_logo.png" alt="logo" width="200" margin="10px"></img>
</div>

### Prophet Hindsight: post-evaluation analysis and data augmentation for 🔮 Prophet Arena

---

## Current Repo Structure

- `common/`: common utilities such as database connection, standardized LLM judge (via the `instructor` library)

- `evaluate/`: a next-generation evaluation framework that will eventually replace the `pm_rank` library. Currently still in development

- `event/`: utilities for event data augmentation, such as augmenting the event title to be more informative and unambiguous.

- `reasoning/`: utilities for reasoning data filtering and augmentation. Currently supports filtering out high-quality raw reasoning traces and augmenting them using LLM augmenters.


## Installation

We recommend using `uv` as the package manager.

```bash
uv sync
```

Or to do regular installation with `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Use Cases

### 1. Fetch data from the database

_This is similar to `retrieve_csv.py` in the `pm_rank` library. But doing so in a much more robust and maintainable way (up to potential issues within the database design)._

```python
from prophet_hindsight.common.forecast import ProphetForecasts, uniform_weighting
import os

forecasts = ProphetForecasts.from_database(
    filter_time_before="2025-10-23 00:00:00",
    filter_time_after="2025-01-01 00:00:00",
    filter_agent_only=False,
    weight_fn=uniform_weighting(),
    exclude_forecasters=None,
    database_url=os.getenv("SUPABASE_DB_URL"),
    save_to_directory=True,
    data_dir="data/raw/full_data"
)
```

By specifying the `save_to_directory` flag, the data will be saved to a directory in the `data/raw/` folder. The directory name will be the time range of the data fetched. Two files will be saved: `predictions.csv` and `submissions.csv`.

Later, you can reuse these saved files to initialize a `ProphetForecasts` object:

```python
forecasts = ProphetForecasts.from_directory(data_dir="data/raw/full_data")
```

### 2. Evaluate the forecasts

_This is similar to `evaluate.py` in the `pm_rank.nightly` module. In the future we might completely move towards `Prophet Hindsight` for evaluation as well._

```python
from prophet_hindsight.evaluate.algo import compute_brier_score

# obtain the brier score
brier_score = compute_brier_score(forecasts.data)

# or directly append the brier score to the forecasts dataframe
compute_brier_score(forecasts.data, append=True)
```

Please check the `prophet_hindsight.evaluate.algo` module for more evaluation metrics and a comprehensive use cases under the `if __name__ == "__main__":` block.

### 3. Obtain the augmented titles for a given list of event tickers

```python
from prophet_hindsight.event.event_augment import augment_event
from prophet_hindsight.common.llm import LLMJudge

# demo: get first 10 event tickers from the forecasts dataframe
event_tickers = forecasts.data["event_ticker"].unique().tolist()[:10]

# use the gpt-5-mini for LLM augmenter with the OpenAI client (instead of OpenRouter)
# Note: this requires you to create a `.env` file to store the `OPENAI_API_KEY`
judge = LLMJudge(model="openai/gpt-5-mini", use_async=True, use_openrouter=False)
results = augment_event(event_tickers, judge, save_path=None)

print(results)
```

### 4. Filter out high-quality reasoning traces and augment them

**Step 1**: Filter out high-quality reasoning traces

The `prophet_hindsight.reasoning.reason_filter` module contains a set of pre-defined criteria functions for filtering out high-quality reasoning traces. For instance, the `top_z_score_criteria` function filters out the reasoning traces that have a z-score (for Brier score) greater than 1.5 and (optionally) with absolute Brier score greater than 0.15.

```python
from prophet_hindsight.reasoning.reason_filter import top_z_score_criteria

# Assume that you already have run evaluation and saved the Brier score csv to this path
brier_df = pd.read_csv("data/raw/full_data/evals/brier_score.csv")

filtered_df = top_z_score_criteria(brier_df, metric_col="brier_score", min_z=1.5, min_val=0.15)
```

**Step 2**: Augment the filtered reasoning traces

```python
from prophet_hindsight.reasoning.reason_augment import augment_reasoning, batch_augment_reasoning
from prophet_hindsight.common.llm import LLMJudge

judge = LLMJudge(model="openai/gpt-5-mini", use_async=True, use_openrouter=False)

# Optionally, you can provide the augmented titles for each event to help the LLM augmenter understand the event better
augmented_df = augment_reasoning(judge, filtered_df, augmented_title_df=augmented_title_df, save_path=None)

# If the number of reasoning traces to augment is too large, you can batch augment them to avoid hitting rate limits
# Here, batch size is 100 and we start from batch 0
batched_augmented_df = batch_augment_reasoning(100, judge, filtered_df, augmented_title_df=augmented_title_df, save_path=None)
```