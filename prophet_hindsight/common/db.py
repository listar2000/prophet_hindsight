import pandas as pd
import json
from sqlalchemy import create_engine, Engine
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Connect to Supabase Postgres
DATABASE_URL = os.getenv("SUPABASE_DB_URL")
# yyyy-mm-dd hh:mm:ss used to include only predictions before this time
FILTER_TIME_BEFORE = f"2025-10-23 00:00:00"
FILTER_TIME_AFTER = f"2025-10-10 00:00:00"
# If this is turn on, we only consider predictor_name that starts with "agent-", if off, we consider all predictor_name
FILTER_AGENT_ONLY = True

_ENGINE_REGISTRY = {}


def get_engine(db_url: str = DATABASE_URL) -> Engine:
    global _ENGINE_REGISTRY
    if db_url not in _ENGINE_REGISTRY:
        _ENGINE_REGISTRY[db_url] = create_engine(db_url)
    return _ENGINE_REGISTRY[db_url]


def get_supabase_client() -> "Client":
    global _ENGINE_REGISTRY
    
    if "supabase" not in _ENGINE_REGISTRY:
        from supabase import create_client, Client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        _ENGINE_REGISTRY["supabase"] = supabase

    return _ENGINE_REGISTRY["supabase"]


def load_prediction_data_from_db(
    engine, 
    filter_time_before: str = FILTER_TIME_BEFORE, 
    filter_time_after: str = FILTER_TIME_AFTER, 
    filter_agent_only: bool = FILTER_AGENT_ONLY
) -> pd.DataFrame:
    """
    Load prediction data from the database with optional filtering.
    
    Args:
        engine: SQLAlchemy engine for database connection
        filter_time_before: Upper bound timestamp for created_at
        filter_time_after: Lower bound timestamp for created_at
        filter_agent_only: If True, only load predictions from predictors starting with 'agent-'
    
    Returns:
        DataFrame with columns: id, event_ticker, predictor_name, prediction, submission_id
    """
    prediction_filter = f"""
        WHERE created_at < '{filter_time_before}'::timestamp 
        AND created_at > '{filter_time_after}'::timestamp
    """
    
    if filter_agent_only:
        logger.info("Filtering predictions: only loading predictor_name starting with 'agent-'")
        prediction_filter += " AND predictor_name LIKE 'agent-%%'"
    
    predictions_df = pd.read_sql(f"""
        SELECT id, event_ticker, predictor_name, prediction, submission_id
        FROM prediction
        {prediction_filter}
    """, engine)
    
    logger.info(f"Loaded {len(predictions_df)} predictions")
    return predictions_df


def load_submission_data_from_db(
    engine, 
    filter_time_before: str = FILTER_TIME_BEFORE, 
    filter_time_after: str = FILTER_TIME_AFTER
) -> pd.DataFrame:
    """
    Load user submission data from the database with time filtering.
    
    Args:
        engine: SQLAlchemy engine for database connection
        filter_time_before: Upper bound timestamp for snapshot_time
        filter_time_after: Lower bound timestamp for snapshot_time
    
    Returns:
        DataFrame with columns: id, submission, snapshot_time, created_at
    """
    submission_filter = f"""
        WHERE snapshot_time < '{filter_time_before}'::timestamp 
        AND snapshot_time > '{filter_time_after}'::timestamp
    """
    
    submissions_df = pd.read_sql(f"""
        SELECT id, submission, snapshot_time, created_at
        FROM user_submission
        {submission_filter}
    """, engine)
    
    logger.info(f"Loaded {len(submissions_df)} user submissions")
    return submissions_df


def load_market_data_from_db(engine, sql_query: str = None) -> pd.DataFrame:
    """
    Load market data from the database (no filtering applied).
    
    Args:
        engine: SQLAlchemy engine for database connection
    
    Returns:
        DataFrame with columns: event_ticker, market_title, created_at,
                               yes_ask, yes_bid, no_ask, no_bid, liquidity
    """
    if sql_query is None:
        sql_query = """
            SELECT event_ticker, market_title, created_at,
                   yes_ask, yes_bid, no_ask, no_bid, liquidity
            FROM market
        """
    markets_df = pd.read_sql(sql_query, engine)
    
    logger.info(f"Loaded {len(markets_df)} markets")
    return markets_df


def load_event_data_from_db(engine, sql_query: str = None) -> pd.DataFrame:
    """
    Load event data from the database, filtering to only events with outcomes.
    
    Args:
        engine: SQLAlchemy engine for database connection
    
    Returns:
        DataFrame with columns: event_ticker, markets, market_outcome, category, close_time
    """
    if sql_query is None:
        sql_query = """
            SELECT event_ticker, title, markets, market_outcome, category, close_time
            FROM event
            WHERE market_outcome IS NOT NULL
        """
    events_df = pd.read_sql(sql_query, engine)
    
    logger.info(f"Loaded {len(events_df)} events with outcomes")
    return events_df


def load_data_from_db(
    engine, 
    filter_time_before: str = FILTER_TIME_BEFORE, 
    filter_time_after: str = FILTER_TIME_AFTER, 
    filter_agent_only: bool = FILTER_AGENT_ONLY
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required data from database using modular helper functions.
    
    Args:
        engine: SQLAlchemy engine for database connection
        filter_time_before: Upper bound timestamp for filtering
        filter_time_after: Lower bound timestamp for filtering
        filter_agent_only: If True, only load predictions from agents
    
    Returns:
        Tuple of (predictions_df, submissions_df, markets_df, events_df)
    """
    logger.info("Loading data from database...")
    logger.info(f"Filtering data created before: {filter_time_before} and after: {filter_time_after}")
    
    # Load all data using modular functions
    predictions_df = load_prediction_data_from_db(
        engine, filter_time_before, filter_time_after, filter_agent_only
    )
    
    submissions_df = load_submission_data_from_db(
        engine, filter_time_before, filter_time_after
    )
    
    markets_df = load_market_data_from_db(engine)
    
    events_df = load_event_data_from_db(engine)
    
    return predictions_df, submissions_df, markets_df, events_df


def validate_market_row(row: pd.Series, use_bid_for_odds: bool = False) -> bool:
    """
    Validate that a market row has all required fields for downstream processing.
    Returns True if valid, False if it would cause warnings/fallbacks.
    
    This mirrors the logic in ProphetArenaChallengeLoader._calculate_implied_probs_for_problem
    """
    # Check if yes_ask/no_ask exists and is valid
    if pd.isna(row.get('yes_ask')) or row.get('yes_ask', 0) <= 0:
        return False
    if pd.isna(row.get('no_ask')) or row.get('no_ask', 0) <= 0:
        return False
    
    # Check liquidity requirement
    if pd.isna(row.get('liquidity')) or row.get('liquidity', 0) < 100:
        return False
    
    # If using bid for odds, check that bid exists too
    if use_bid_for_odds:
        if pd.isna(row.get('yes_bid')) or pd.isna(row.get('no_bid')):
            return False
    
    return True


def process_predictions(predictions_df: pd.DataFrame, submissions_df: pd.DataFrame, 
                       markets_df: pd.DataFrame, events_df: pd.DataFrame,
                       use_bid_for_odds: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process and merge prediction data using efficient DataFrame operations.
    Returns two dataframes: predictions and augmented_submissions.
    """
    logger.info("Processing data...")
    
    # Filter to only valid event tickers
    valid_event_tickers = set(events_df['event_ticker'].unique())
    initial_pred_count = len(predictions_df)
    predictions_df = predictions_df[predictions_df['event_ticker'].isin(valid_event_tickers)].copy()
    logger.info(f"Filtered to {len(predictions_df)} predictions with valid event tickers "
                f"(removed {initial_pred_count - len(predictions_df)})")
    
    # Step 1: Build augmented submissions CSV first
    logger.info("Building augmented submissions...")
    
    # Filter markets to only valid ones
    markets_df['is_valid'] = markets_df.apply(lambda row: validate_market_row(row, use_bid_for_odds), axis=1)
    initial_market_count = len(markets_df)
    markets_df = markets_df[markets_df['is_valid']].drop(columns=['is_valid'])
    logger.info(f"Filtered to {len(markets_df)} valid market rows (removed {initial_market_count - len(markets_df)})")
    
    # Get canonical market order for each event_ticker
    # We'll use the first submission's market order as canonical
    logger.info("Determining canonical market order for each event_ticker...")
    market_order_map = {}
    for event_ticker in valid_event_tickers:
        event_markets = markets_df[markets_df['event_ticker'] == event_ticker].sort_values('created_at')
        if len(event_markets) > 0:
            # Get unique market titles in order of first appearance
            first_snapshot = event_markets['created_at'].min()
            first_markets = event_markets[event_markets['created_at'] == first_snapshot]
            # Sort by some consistent field to ensure deterministic order
            market_titles = sorted(first_markets['market_title'].unique())
            market_order_map[event_ticker] = market_titles
    
    # Group markets by (event_ticker, snapshot_time) to create augmented submissions
    def create_market_dict(group, canonical_order):
        """Create ordered market dictionary from market rows."""
        market_dict = {}
        for market_title in canonical_order:
            # Find the row for this market_title
            rows = group[group['market_title'] == market_title]
            if len(rows) > 0:
                row = rows.iloc[0]
                market_dict[market_title] = {
                    'yes_ask': float(row['yes_ask']) if pd.notna(row['yes_ask']) else None,
                    'yes_bid': float(row['yes_bid']) if pd.notna(row['yes_bid']) else None,
                    'no_ask': float(row['no_ask']) if pd.notna(row['no_ask']) else None,
                    'no_bid': float(row['no_bid']) if pd.notna(row['no_bid']) else None,
                    'liquidity': float(row['liquidity']) if pd.notna(row['liquidity']) else None,
                }
        return market_dict
    
    # Group markets and create augmented submissions
    submissions_list = []
    for (event_ticker, snapshot_time), group in markets_df.groupby(['event_ticker', 'created_at']):
        if event_ticker not in market_order_map:
            continue
        canonical_order = market_order_map[event_ticker]
        market_dict = create_market_dict(group, canonical_order)
        
        # Only include if we have all markets
        if len(market_dict) == len(canonical_order):
            submissions_list.append({
                'event_ticker': event_ticker,
                'snapshot_time': snapshot_time,
                'market_dict': market_dict,
            })
    
    augmented_submissions_df = pd.DataFrame(submissions_list)
    logger.info(f"Created {len(augmented_submissions_df)} augmented submissions")
    
    # Add round: rank by timestamp within each event_ticker
    augmented_submissions_df = augmented_submissions_df.sort_values(['event_ticker', 'snapshot_time'])
    augmented_submissions_df['round'] = augmented_submissions_df.groupby('event_ticker').cumcount()
    logger.info(f"Added round (max: {augmented_submissions_df['round'].max()})")
    
    # Merge with events to get market_outcome and category
    events_df_indexed = events_df.set_index('event_ticker')
    augmented_submissions_df = augmented_submissions_df.merge(
        events_df_indexed[['title', 'market_outcome', 'category', 'close_time']],
        left_on='event_ticker',
        right_index=True,
        how='inner'
    )
    
    # Merge with submissions to get submission_id
    submissions_df['snapshot_key'] = submissions_df['snapshot_time']
    augmented_submissions_df['snapshot_key'] = augmented_submissions_df['snapshot_time']
    augmented_submissions_df = augmented_submissions_df.merge(
        submissions_df[['id', 'snapshot_key']],
        on='snapshot_key',
        how='left'
    )
    augmented_submissions_df['submission_id'] = augmented_submissions_df['id'].fillna('').astype(str)
    augmented_submissions_df = augmented_submissions_df.drop(columns=['id', 'snapshot_key'])
    
    # Step 2: Build predictions CSV using augmented submissions
    logger.info("Building predictions CSV...")
    
    # Merge predictions with submissions to get snapshot_time
    predictions_df = predictions_df.merge(
        submissions_df[['id', 'snapshot_time']],
        left_on='submission_id',
        right_on='id',
        how='inner',
        suffixes=('', '_submission')
    )
    logger.info(f"After merging submissions: {len(predictions_df)} rows")
    
    # Merge with augmented_submissions to get round
    predictions_df = predictions_df.merge(
        augmented_submissions_df[['event_ticker', 'snapshot_time', 'round', 'title']],
        on=['event_ticker', 'snapshot_time'],
        how='inner'
    )
    logger.info(f"After merging with augmented submissions: {len(predictions_df)} rows")
    
    # Merge with events to get category
    predictions_df = predictions_df.merge(
        events_df_indexed[['title', 'market_outcome', 'category']],
        left_on='event_ticker',
        right_index=True,
        how='inner'
    )
    
    # Create predictions output dataframe
    logger.info("Creating predictions output dataframe...")
    predictions_output = pd.DataFrame({
        'prediction_id': predictions_df['id'].astype(str),
        'submission_id': predictions_df['submission_id'].astype(str),
        'forecaster': predictions_df['predictor_name'],
        'event_ticker': predictions_df['event_ticker'],
        'title': predictions_df['title'],
        'round': predictions_df['round'].astype(int),
        'prediction': predictions_df['prediction'].apply(json.dumps),
    })
    
    # Create augmented submissions output dataframe
    logger.info("Creating augmented submissions output dataframe...")
    submissions_output = pd.DataFrame({
        'submission_id': augmented_submissions_df['submission_id'],
        'event_ticker': augmented_submissions_df['event_ticker'],
        'round': augmented_submissions_df['round'].astype(int),
        'snapshot_time': augmented_submissions_df['snapshot_time'].apply(
            lambda x: x.isoformat() if pd.notna(x) and hasattr(x, 'isoformat') else str(x)
        ),
        'close_time': augmented_submissions_df['close_time'].apply(
            lambda x: x.isoformat() if pd.notna(x) and hasattr(x, 'isoformat') else str(x)
        ),
        'market_data': augmented_submissions_df['market_dict'].apply(json.dumps),
        'market_outcome': augmented_submissions_df['market_outcome'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        ),
        'category': augmented_submissions_df['category'],
    })
    
    return predictions_output, submissions_output


def get_mm_dd(date_str: str) -> str:
    date_str = date_str.split(' ')[0]
    return f"{date_str.split('-')[1]}_{date_str.split('-')[2]}"
