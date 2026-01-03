"""
The different criteria function for filtering what is a good prediction versus a bad prediction.
"""

import json
import logging

import json_repair
import pandas as pd
from sqlalchemy import Engine, text

from prophet_hindsight.common.db import get_engine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def top_absolute_score_criteria(
    results_df: pd.DataFrame,
    metric_col: str = "brier_score",
    top_k: int = -1,
    top_p: float = 0.1,
    min_val: float = -1,
):
    """
    Group by submission_id and for each submission, select the best prediction by comparing among the LLM forecasters.
    We offer two options of selection (default to `top_p = 0.1`):
        - `top_k`: select the top k predictions by the metric (turn off when `top_k < 1`)
        - `top_p`: select the top p% predictions by the metric (turn off when `top_p < 1`)
    We then filter out the submissions that do not meet the criteria.

    Additionally, we can specify a minimum value for the metric to select.
    """
    top_k = int(top_k)
    use_top_k = top_k > 0

    if not use_top_k:
        assert top_p > 0 and top_p < 1, "top_p must be between 0 and 1"

    ascending = metric_col == "brier_score"  # Brier score is lower is better

    # group by submission_id and filter within each group
    filtered_groups = []

    for _, group in results_df.groupby("submission_id"):
        # Sort by metric (ascending for metrics where lower is better)
        sorted_group = group.sort_values(by=metric_col, ascending=ascending)

        # Select top k or top p%
        if use_top_k:
            selected = sorted_group.head(top_k)
        else:
            # Calculate number of predictions to select (at least 1)
            num_to_select = max(1, int(len(sorted_group) * top_p))
            selected = sorted_group.head(num_to_select)

        # Filter by minimum/maximum value if specified
        if min_val > 0:
            if ascending:
                # For metrics where lower is better, filter by maximum allowed value
                selected = selected[selected[metric_col] <= min_val]
            else:
                # For metrics where higher is better, filter by minimum required value
                selected = selected[selected[metric_col] >= min_val]

        # Only add if there are remaining predictions after filtering
        if len(selected) > 0:
            filtered_groups.append(selected)

    # Combine all filtered groups
    if len(filtered_groups) == 0:
        logger.warning("No submissions met the criteria")
        # Return empty DataFrame with same columns as input
        return pd.DataFrame(columns=results_df.columns)

    filtered_df = pd.concat(filtered_groups, ignore_index=True)
    return filtered_df


def top_z_score_criteria(
    results_df: pd.DataFrame,
    metric_col: str = "brier_score",
    min_z: float = 1.5,
    max_mean: float = -1,
    min_val: float = -1,
):
    """
    For each submission i, compute the z-score for each forecaster j as the (score_ij - mean(score_i)) / std(score_i)
    We then select the submissions that have a z-score greater than `min_z` and (optionally) with absolute score greater than `min_val`.
    We also filter out the submission where everyone can do well, i.e. the mean score is greater than `max_mean`.

    For metrics where lower is better (e.g., brier_score), we select z-scores < -min_z (i.e., scores significantly below the mean).
    For metrics where higher is better, we select z-scores > min_z (i.e., scores significantly above the mean).

    Additionally, we can specify a minimum value for the metric to select.
    """
    ascending = metric_col == "brier_score"  # Brier score is lower is better

    # group by submission_id and filter within each group
    filtered_groups = []

    for _, group in results_df.groupby("submission_id"):
        # Skip if group has less than 2 items (z-score needs variance)
        if len(group) < 2:
            continue

        # Skip if there exists any forecaster who predicts more than once within this submission
        if len(group["forecaster"].unique()) != len(group):
            continue

        # Compute z-scores for this submission
        mean_score = group[metric_col].mean()
        std_score = group[metric_col].std()

        # Check if the mean score is too good, i.e. the submission is "too easy" for all forecasters
        if max_mean > 0:
            unqualified_mean = mean_score < max_mean if ascending else mean_score > max_mean
        else:
            unqualified_mean = False

        # Skip if std is 0 (all scores are the same)
        if std_score == 0 or unqualified_mean:
            continue

        group = group.copy()
        group["z_score"] = (group[metric_col] - mean_score) / std_score

        # Select based on z-score threshold
        if ascending:
            # For metrics where lower is better, select z-scores < -min_z
            selected = group[group["z_score"] < -min_z]
        else:
            # For metrics where higher is better, select z-scores > min_z
            selected = group[group["z_score"] > min_z]

        # Filter by minimum/maximum value if specified
        if min_val > 0:
            if ascending:
                # For metrics where lower is better, filter by maximum allowed value
                selected = selected[selected[metric_col] <= min_val]
            else:
                # For metrics where higher is better, filter by minimum required value
                selected = selected[selected[metric_col] >= min_val]

        # Drop the z_score column before adding to results
        if len(selected) > 0:
            selected = selected.drop(columns=["z_score"])
            filtered_groups.append(selected)

    # Combine all filtered groups
    if len(filtered_groups) == 0:
        logger.warning("No submissions met the criteria")
        # Return empty DataFrame with same columns as input
        return pd.DataFrame(columns=results_df.columns)

    filtered_df = pd.concat(filtered_groups, ignore_index=True)
    return filtered_df


def ambiguous_event_criteria(
    results_df: pd.DataFrame,
    metric_col: str = "brier_score",
    min_val: float = 0.25,
    max_val: float = 0.15,
    top_k: int = -1,
):
    """
    Filter by the following criteria:
    - We first group by `submission_id`. We only retain groups where all the predictions within it have Brier score > max_val.
    - Then for these groups, we leave only the predictions with Brier score < min_val.
    """
    ascending = metric_col == "brier_score"  # Brier score is lower is better
    if ascending:
        assert min_val > max_val, "min_val must be greater than max_val for ascending metrics"
        # manually swap so later logic can be consistent
        min_val, max_val = max_val, min_val
    else:
        assert min_val < max_val, "min_val must be less than max_val for descending metrics"

    # Step 1: Group by submission_id and keep only groups where ALL predictions have metric > min_val
    grouped = results_df.groupby("submission_id")
    valid_groups = grouped.filter(lambda x: (x[metric_col] > min_val).all())

    # Step 2: Within those valid groups, keep only predictions with metric < max_val (and the top k predictions if provided)
    # Otherwise, we keep all predictions whose metric is < max_val.
    filtered_results = valid_groups[valid_groups[metric_col] < max_val]

    if top_k > 0:
        # Sort by metric (ascending for metrics where lower is better) and take top k per group
        filtered_results = (
            filtered_results.sort_values(metric_col, ascending=ascending)
            .groupby("submission_id", as_index=False)
            .head(top_k)
        )

    return filtered_results


def _augment_results_with_sources(
    results_df: pd.DataFrame,
    engine: Engine,
    submission_id_col: str = "submission_id",
    filter_contributor_only: list[str] | None = None,
) -> pd.DataFrame:
    """
    Augment each row of `results_df` with a JSON list of sources under column `sources`.
    The JSON is a Python list[dict] like [{"url": "...", "summary": "...", "contributor": "...", "title": "..."}, ...].

    If `filter_contributor_only` is provided, our final dataframe

    Args:
        results_df: existing dataframe with a `submission_id` column.
        engine: SQLAlchemy engine connected to the Supabase Postgres.
        submission_id_col: name of the submission id column in the dataframe.
        filter_contributor_only: list of contributors to filter by. If provided, our final dataframe will only contain submissions from these contributors.
    """
    if submission_id_col not in results_df.columns:
        raise ValueError(f"DataFrame must contain column '{submission_id_col}'")

    # Reduce to the unique IDs we actually need to fetch
    ids = results_df[submission_id_col].dropna().unique().tolist()
    assert len(ids) > 0, "No submission IDs to fetch sources for"

    # Using tuple binding which works better with SQLAlchemy
    params = {f"id_{i}": str(id_val) for i, id_val in enumerate(ids)}

    sql = f"""
    WITH ids AS (
      SELECT DISTINCT id::uuid AS submission_id
      FROM (VALUES {','.join([f"(:id_{i})" for i in range(len(ids))])}) AS t(id)
    ),
    src_agg AS (
      SELECT
        ss.user_submission_id AS submission_id,
        jsonb_agg(
          DISTINCT jsonb_build_object(
            'url', s.url,
            'summary', s.summary,
            'contributor', s.contributor,
            'title', s.title
          )
        ) AS sources
      FROM submission_source_usage ss
      JOIN source s ON s.id = ss.source_id
      JOIN ids i ON i.submission_id = ss.user_submission_id
      GROUP BY ss.user_submission_id
    )
    SELECT
      i.submission_id,
      COALESCE(src_agg.sources, '[]'::jsonb) AS sources
    FROM ids i
    LEFT JOIN src_agg ON src_agg.submission_id = i.submission_id;
    """
    # Pull the aggregated sources once
    src_df = pd.read_sql(text(sql), engine, params=params)

    # If `filter_contributor_only` is provided, filter the src_df to only include submissions from these contributors
    if filter_contributor_only:

        def _contain_contributor(source_list: list) -> bool:
            # only need to look at the first contributor as we assume all contributors are the same
            if len(source_list) == 0:
                return False
            return source_list[0]["contributor"] in filter_contributor_only

        src_df = src_df[src_df["sources"].apply(_contain_contributor)]

    # Now remove the contributor key from each source in the sources list
    src_df["sources"] = src_df["sources"].apply(
        lambda x: [{k: v for k, v in source.items() if k != "contributor"} for source in x]
    )

    # Convert both columns to string for merging to avoid UUID type mismatch
    src_df["submission_id"] = src_df["submission_id"].astype(str)
    results_df_copy = results_df.copy()
    results_df_copy[submission_id_col] = results_df_copy[submission_id_col].astype(str)

    # Merge them back to your results_df; rows with the same submission_id all get the same sources array
    out = results_df_copy.merge(
        src_df, how="inner", left_on=submission_id_col, right_on="submission_id"
    )

    # For submission_ids with no sources, ensure empty list rather than NaN
    out["sources"] = out["sources"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def _augment_results_with_prediction_context(
    results_df: pd.DataFrame,
    engine,
    submission_id_col: str = "submission_id",
    predictor_col: str = "forecaster",
) -> pd.DataFrame:
    """
    Augment each row of `results_df` with the prediction JSON from the `prediction` table.
    Matches on both submission_id AND predictor_name to get the specific prediction for each forecaster.

    Args:
        results_df: existing dataframe with submission_id and predictor columns.
        engine: SQLAlchemy engine connected to the Supabase Postgres.
        submission_id_col: name of the submission id column in the dataframe.
        predictor_col: name of the predictor/forecaster column in the dataframe.
    """
    from sqlalchemy import text

    if submission_id_col not in results_df.columns:
        raise ValueError(f"DataFrame must contain column '{submission_id_col}'")
    if predictor_col not in results_df.columns:
        raise ValueError(f"DataFrame must contain column '{predictor_col}'")

    # Get unique (submission_id, predictor_name) pairs
    pairs = results_df[[submission_id_col, predictor_col]].drop_duplicates().dropna()
    assert len(pairs) > 0, "No submission-predictor pairs to fetch predictions for"

    # Create parameters for each pair
    params = {}
    for i, (_, row) in enumerate(pairs.iterrows()):
        params[f"sub_id_{i}"] = str(row[submission_id_col])
        params[f"pred_name_{i}"] = str(row[predictor_col])

    # Build VALUES clause for the pairs
    values_clause = ",".join([f"(:sub_id_{i}, :pred_name_{i})" for i in range(len(pairs))])

    sql = f"""
    WITH pairs AS (
      SELECT 
        sub_id::uuid AS submission_id,
        pred_name::text AS predictor_name
      FROM (VALUES {values_clause}) AS t(sub_id, pred_name)
    )
    SELECT
      p.submission_id::text AS submission_id,
      p.predictor_name,
      p.prediction
    FROM prediction p
    JOIN pairs pr ON pr.submission_id = p.submission_id 
                  AND pr.predictor_name = p.predictor_name
    WHERE p.prediction IS NOT NULL;
    """

    # Fetch predictions
    pred_df = pd.read_sql(text(sql), engine, params=params)

    # Convert to string for merging to avoid type mismatch
    pred_df["submission_id"] = pred_df["submission_id"].astype(str)
    pred_df["predictor_name"] = pred_df["predictor_name"].astype(str)

    results_df_copy = results_df.copy()
    results_df_copy[submission_id_col] = results_df_copy[submission_id_col].astype(str)
    results_df_copy[predictor_col] = results_df_copy[predictor_col].astype(str)

    # Merge on both submission_id and predictor_name
    out = results_df_copy.merge(
        pred_df,
        how="left",
        left_on=[submission_id_col, predictor_col],
        right_on=["submission_id", "predictor_name"],
    )

    # Clean up duplicate columns if names differ
    if submission_id_col != "submission_id":
        out.drop(columns=["submission_id"], inplace=True)
    if predictor_col != "predictor_name":
        out.drop(columns=["predictor_name"], inplace=True)

    # 1. Drop all rows where the "prediction" column is not an instance of dict
    # 2. Turn the dictionary into a string by Json dumps
    out = out[out["prediction"].apply(lambda x: isinstance(x, dict) or isinstance(x, str))]
    out["prediction"] = out["prediction"].apply(
        lambda x: json.dumps(x) if isinstance(x, dict) else x
    )
    return out


def _augment_results_with_event_details(
    results_df: pd.DataFrame,
    engine,
    event_ticker_col: str = "event_ticker",
    filter_category: list[str] | None = None,
) -> pd.DataFrame:
    """
    Augment each row of `results_df` with the event title from the `event` table.
    Matches on event_ticker to get the specific event title.

    If `filter_category` is provided, we will only retain the rows where the event category is in the list.
    """
    if event_ticker_col not in results_df.columns:
        raise ValueError(f"DataFrame must contain column '{event_ticker_col}'")

    # Get unique event_tickers
    event_tickers = results_df[event_ticker_col].dropna().unique().tolist()
    assert len(event_tickers) > 0, "No event tickers to fetch titles for"

    # Using tuple binding which works better with SQLAlchemy
    params = {
        f"event_ticker_{i}": str(event_ticker) for i, event_ticker in enumerate(event_tickers)
    }
    sql = f"""
    SELECT event_ticker, title, category, rules FROM event WHERE event_ticker IN ({','.join([f"(:event_ticker_{i})" for i in range(len(event_tickers))])})
    """
    event_df = pd.read_sql(text(sql), engine, params=params)

    # Convert to string for merging to avoid type mismatch
    event_df["event_ticker"] = event_df["event_ticker"].astype(str)
    results_df_copy = results_df.copy()
    results_df_copy[event_ticker_col] = results_df_copy[event_ticker_col].astype(str)

    # Merge on event_ticker
    out = results_df_copy.merge(
        event_df, how="left", left_on=event_ticker_col, right_on="event_ticker"
    )

    # Clean up duplicate columns if names differ
    if event_ticker_col != "event_ticker":
        out.drop(columns=["event_ticker"], inplace=True)

    # Filter by category if provided
    if filter_category:
        out = out[out["category"].isin(filter_category)]

    # For rows with no details, ensure empty string rather than NaN
    out["title"] = out["title"].apply(lambda x: x if isinstance(x, str) else "")
    out["category"] = out["category"].apply(lambda x: x if isinstance(x, str) else "")
    out["rules"] = out["rules"].apply(lambda x: x if isinstance(x, str) else "")
    return out


def _augment_results_with_market_data(
    results_df: pd.DataFrame,
    submission_df: pd.DataFrame | str,
    submission_id_col: str = "submission_id",
) -> pd.DataFrame:
    if isinstance(submission_df, str):
        # load it from the csv
        submission_df = pd.read_csv(submission_df)

    # simply merge the `market_outcome` column into the results_df (merging by the submission_id key)
    if submission_id_col != "submission_id":
        results_df["submission_id"] = results_df[submission_id_col]

    results_df_copy = results_df.copy()
    out = results_df_copy.merge(
        submission_df[["submission_id", "market_outcome", "market_data"]],
        how="left",
        left_on=submission_id_col,
        right_on="submission_id",
    )

    out["market_outcome"] = out["market_outcome"].apply(json_repair.loads)
    out["market_data"] = out["market_data"].apply(json_repair.loads)

    if submission_id_col != "submission_id":
        out.drop(columns=["submission_id"], inplace=True)
    return out


def augment_filtered_predictions(
    filtered_df: pd.DataFrame,
    engine: Engine | None = None,
    submission_df_path: str | None = None,
    filter_contributor_only: list[str] | None = None,
    filter_category: list[str] | None = None,
) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()

    augmented_df = _augment_results_with_sources(
        filtered_df, engine, filter_contributor_only=filter_contributor_only
    )
    augmented_df = _augment_results_with_prediction_context(augmented_df, engine)
    augmented_df = _augment_results_with_event_details(
        augmented_df, engine, filter_category=filter_category
    )
    if submission_df_path is not None:
        submission_df = pd.read_csv(submission_df_path)
        augmented_df = _augment_results_with_market_data(augmented_df, submission_df)
    return augmented_df


if __name__ == "__main__":
    import os

    brier_df = pd.read_csv("data/raw/after_cleanup/evals/brier_score_with_mb.csv")
    old_len = len(brier_df)

    # filter out all the rows where the forecaster starts with "agent-"
    # brier_df = brier_df[~brier_df["forecaster"].str.startswith("agent-")]
    # filtered_df = top_z_score_criteria(brier_df, metric_col="brier_score", min_z=1.25, min_val=0.15)
    filtered_df = ambiguous_event_criteria(
        brier_df, metric_col="brier_score", min_val=0.2, max_val=0.15, top_k=5
    )

    print(f"Filtered out {old_len - len(filtered_df)} rows")

    augmented_df = augment_filtered_predictions(
        filtered_df,
        submission_df_path="data/raw/after_cleanup/submissions.csv",
        filter_contributor_only=["o3", "gpt-4o"],
        filter_category=["Sports"],
    )

    print(
        f"Final number of rows: {len(augmented_df)}\n"
        f"Number of unique forecasters: {len(augmented_df['forecaster'].unique())}\n"
        f"Number of unique submissions: {len(augmented_df['submission_id'].unique())}\n"
        f"Number of unique events: {len(augmented_df['event_ticker'].unique())}"
    )

    save_dir = "data/raw/after_cleanup/reasoning"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    augmented_df.to_csv(save_dir + "/hard_predictions.csv", index=False)
    augmented_df.to_json(save_dir + "/hard_predictions.json", orient="index", indent=2)
