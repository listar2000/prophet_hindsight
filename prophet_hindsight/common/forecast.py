"""
Adapted from the `pm_rank` packages
"""
import pandas as pd
import numpy as np
import json
from typing import Literal
import os

WeightingStrategy = Literal['uniform', 'first_n', 'last_n', 'exponential']

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def uniform_weighting():
    # give each forecast a weight of 1
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def first_n_weighting(n = 1, group_col: list[str] = ['forecaster', 'event_ticker'], time_col: str = 'round'):
    # give the first n forecasts a weight of 1, and the rest directly filtered out
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.sort_values(by=time_col, ascending=True)
        forecasts = forecasts.groupby(group_col).head(n)
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def last_n_weighting(n = 1, group_col: list[str] = ['forecaster', 'event_ticker'], time_col: str = 'round'):
    # give the last n forecasts a weight of 1, and the rest directly filtered out
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.sort_values(by=time_col, ascending=True)
        forecasts = forecasts.groupby(group_col).tail(n)
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def exponential_weighting(lambda_ = 0.1, time_col: str = 'time_rank'):
    # give the forecasts a weight of e^(-lambda * relative_time), where relative_time is the positional distance from the most recent forecast
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.copy()
        forecasts['weight'] = np.exp(-lambda_ * forecasts[time_col])
        return forecasts
    return weight_fn


def _turn_market_data_to_odds(market_data: dict) -> tuple[np.ndarray, np.ndarray]:
    # sort the list to ensure market consistency
    markets = sorted(list(market_data.keys()))
    yes_asks = np.array([market_data[mkt]['yes_ask'] / 100.0 for mkt in markets])
    no_asks = np.array([market_data[mkt]['no_ask'] / 100.0 for mkt in markets])
    return yes_asks, no_asks


def _simplify_prediction(prediction: dict) -> tuple[np.ndarray, str]:
    # Extract rationale before reassigning prediction
    rationale = prediction.get('rationale', '')
    # Convert probabilities to a dict
    prediction_dict = {item['market']: item['probability'] for item in prediction['probabilities']}
    # Return sorted probabilities as array and rationale as string
    return np.array([prediction_dict[mkt] for mkt in sorted(list(prediction_dict.keys()))]), str(rationale)


def _simplify_market_outcome(market_outcome: dict) -> np.ndarray:
    return np.array([market_outcome[mkt] for mkt in sorted(list(market_outcome.keys()))])


def _parse_and_merge_predictions_and_submissions(predictions_df: pd.DataFrame, submissions_df: pd.DataFrame, weight_fn) -> pd.DataFrame:
    predictions_df = predictions_df.copy()
    submissions_df = submissions_df.copy()
    # Parse JSON columns
    predictions_df['prediction'], predictions_df['rationale'] = zip(*predictions_df['prediction'].apply(json.loads).apply(_simplify_prediction))
    submissions_df['market_outcome'] = submissions_df['market_outcome'].apply(json.loads).apply(_simplify_market_outcome)
    submissions_df['market_data'] = submissions_df['market_data'].apply(json.loads)

    # Convert the `market_data` in submissions_df to a list of odds & no_odds
    submissions_df['odds'], submissions_df['no_odds'] = zip(*submissions_df['market_data'].apply(_turn_market_data_to_odds))

    # Merge predictions with submissions for the odds and no_odds columns
    merged = predictions_df.merge(
        submissions_df[['event_ticker', 'round', 'odds', 'no_odds', 'snapshot_time', 'close_time', 'market_outcome']],
        on=['event_ticker', 'round'],
        how='inner'
    )

    # We leave only rows where the `odds`, `prediction`, `market_outcome` columns have the same length
    odds_len, prediction_len, market_outcome_len = merged['odds'].apply(len), merged['prediction'].apply(len), merged['market_outcome'].apply(len)
    merged = merged[(odds_len == prediction_len) & (odds_len == market_outcome_len)]

    # Add `relative_round` column
    merged['time_rank'] = merged.groupby(['forecaster', 'event_ticker'])['round'].rank(ascending=False) - 1

    # Apply the weighting function
    merged = weight_fn(merged)

    logger.info(f"Loaded {len(merged)} rows")
    return merged


class ProphetForecasts:
    """
    The class holding the data useful for downstream analysis.
    
    The main data is stored in the `forecasts` DataFrame with the following columns:
    - forecaster: Name of the forecasting agent
    - event_ticker: Unique identifier for the event
    - submission_id: Unique identifier for the submission
    - round: Submission round number
    - prediction: Array of probability predictions (np.ndarray)
    - rationale: Reasoning behind the prediction
    - market_outcome: Actual outcome values (np.ndarray)
    - odds: Market odds, yes ask (np.ndarray)
    - no_odds: Market odds, no ask (np.ndarray)
    - snapshot_time: When the snapshot was taken
    - close_time: When the market closed
    - time_rank: Relative time ranking (0 = most recent)
    - weight: Weight assigned to this forecast
    
    Example:
        >>> forecasts = ProphetForecasts.from_directory("data/raw/...")
        >>> df = forecasts.forecasts
        >>> # Access columns
        >>> agents = df[ProphetForecasts.COL_FORECASTER]
        >>> # Or use directly
        >>> filtered = df[df['weight'] > 0.5]
    """
    
    # Column name constants for IDE autocomplete
    COL_FORECASTER = 'forecaster'
    COL_EVENT_TICKER = 'event_ticker'
    COL_SUBMISSION_ID = 'submission_id'
    COL_ROUND = 'round'
    COL_PREDICTION = 'prediction'
    COL_RATIONALE = 'rationale'
    COL_MARKET_OUTCOME = 'market_outcome'
    COL_ODDS = 'odds'
    COL_NO_ODDS = 'no_odds'
    COL_SNAPSHOT_TIME = 'snapshot_time'
    COL_CLOSE_TIME = 'close_time'
    COL_TIME_RANK = 'time_rank'
    COL_WEIGHT = 'weight'
    
    # Data source column lists
    PREDICTION_COLS = ['forecaster', 'event_ticker', 'round', 'prediction', 'submission_id']
    SUBMISSION_COLS = ['event_ticker', 'round', 'market_data', 'market_outcome', 'snapshot_time', 'close_time']

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)
    
    def describe_schema(self) -> pd.DataFrame:
        """
        Return a DataFrame describing the schema of the forecasts DataFrame.
        
        Returns:
            DataFrame with columns: Column, Description, Dtype, Non-Null, Null
        
        Example:
            >>> forecasts = ProphetForecasts.from_directory("data/raw/...")
            >>> schema = forecasts.describe_schema()
            >>> print(schema)
        """
        descriptions = {
            self.COL_FORECASTER: 'Name of the forecasting agent',
            self.COL_EVENT_TICKER: 'Unique identifier for the event',
            self.COL_SUBMISSION_ID: 'Unique identifier for the submission',
            self.COL_ROUND: 'Submission round number',
            self.COL_PREDICTION: 'Array of probability predictions',
            self.COL_RATIONALE: 'Reasoning behind the prediction',
            self.COL_MARKET_OUTCOME: 'Actual outcome values',
            self.COL_ODDS: 'Market odds (yes ask)',
            self.COL_NO_ODDS: 'Market odds (no ask)',
            self.COL_SNAPSHOT_TIME: 'When the snapshot was taken',
            self.COL_CLOSE_TIME: 'When the market closed',
            self.COL_TIME_RANK: 'Relative time ranking (0 = most recent)',
            self.COL_WEIGHT: 'Weight assigned to this forecast'
        }
        
        present_cols = [col for col in descriptions.keys() if col in self.data.columns]
        
        return pd.DataFrame({
            'Column': present_cols,
            'Description': [descriptions[col] for col in present_cols],
            'Dtype': [str(self.data[col].dtype) for col in present_cols],
            'Non-Null': [self.data[col].notna().sum() for col in present_cols],
            'Null': [self.data[col].isna().sum() for col in present_cols]
        })

    def __init__(self, raw_predictions_df: pd.DataFrame, raw_submissions_df: pd.DataFrame, merged_forecasts_df: pd.DataFrame, exclude_forecasters: list[str] = None):
        self.raw_predictions_df = raw_predictions_df
        self.raw_submissions_df = raw_submissions_df

        prev_len = len(merged_forecasts_df)
        if exclude_forecasters is not None:
            merged_forecasts_df = merged_forecasts_df[~merged_forecasts_df['forecaster'].isin(exclude_forecasters)]
            print(f"Filtered out {prev_len - len(merged_forecasts_df)} forecasts. Remaining {len(merged_forecasts_df)}.")
        self.data = merged_forecasts_df

    @classmethod
    def from_database(cls, 
            filter_time_before: str = None, 
            filter_time_after: str = None, 
            filter_agent_only: bool = False,
            weight_fn = uniform_weighting(),
            exclude_forecasters: list[str] = None,
            database_url: str = None,
            save_to_directory: bool = False,
            data_dir: str = None
        ) -> 'ProphetForecasts':

        from prophet_hindsight.common.db import (
            load_data_from_db, 
            process_predictions, 
            DATABASE_URL, 
            FILTER_TIME_BEFORE, 
            FILTER_TIME_AFTER, 
            FILTER_AGENT_ONLY,
            get_mm_dd,
            get_engine
        )

        filter_time_before = FILTER_TIME_BEFORE if filter_time_before is None else filter_time_before
        filter_time_after = FILTER_TIME_AFTER if filter_time_after is None else filter_time_after
        filter_agent_only = FILTER_AGENT_ONLY if filter_agent_only is None else filter_agent_only

        engine = get_engine(db_url=DATABASE_URL if database_url is None else database_url)
        logger.info(f"Loading forecasts from database")

        predictions_df, submissions_df, markets_df, events_df = load_data_from_db(engine, filter_time_before, filter_time_after, filter_agent_only)
        predictions_output_df, submissions_output_df = process_predictions(predictions_df, submissions_df, markets_df, events_df)

        if save_to_directory:
            if data_dir is None:
                data_dir = f"data/raw/data_from_{get_mm_dd(filter_time_before)}_to_{get_mm_dd(filter_time_after)}"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            predictions_output_df.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)
            submissions_output_df.to_csv(os.path.join(data_dir, "submissions.csv"), index=False)
            logger.info(f"Saved predictions and submissions to {data_dir}")

        merged = _parse_and_merge_predictions_and_submissions(predictions_output_df[cls.PREDICTION_COLS], submissions_output_df[cls.SUBMISSION_COLS], weight_fn)
        return cls(predictions_output_df, submissions_output_df, merged, exclude_forecasters)


    @classmethod
    def from_directory(cls, data_dir: str, weight_fn = uniform_weighting(), exclude_forecasters: list[str] = None):
        logger.info(f"Loading forecasts from {data_dir}")
        logger.info(f"Weighting function: {weight_fn}")

        # make sure the data_dir exists and is a directory
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Load CSVs
        predictions_df = pd.read_csv(os.path.join(data_dir, "predictions.csv"))
        submissions_df = pd.read_csv(os.path.join(data_dir, "submissions.csv"))
        
        merged = _parse_and_merge_predictions_and_submissions(predictions_df[cls.PREDICTION_COLS], submissions_df[cls.SUBMISSION_COLS], weight_fn)
        return cls(predictions_df, submissions_df, merged, exclude_forecasters)


if __name__ == "__main__":
    ProphetForecasts.from_database(
        filter_time_before="2025-10-23 00:00:00",
        filter_time_after="2025-01-01 00:00:00",
        filter_agent_only=False,
        weight_fn=uniform_weighting(),
        exclude_forecasters=None,
        database_url=None,
        save_to_directory=True,
        data_dir="data/raw/full_data"
    )
