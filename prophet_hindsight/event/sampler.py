from collections import Counter

import pandas as pd
from sqlalchemy import text

from prophet_hindsight.common.db import get_engine


# get the category distribution (percentage) of all events
def get_category_distribution(events_df: pd.DataFrame, normalize: bool = True) -> dict[str, float]:
    category_dist = events_df["category"].value_counts(normalize=normalize).to_dict()
    # remove any item with value < 0.01
    category_dist = {k: v for k, v in category_dist.items() if v >= 0.01}
    return category_dist


def sample_data_according_to_category_dist(
    df: pd.DataFrame,
    category_distribution: dict[str, float],
    total_budget: int = 1200,
    max_submissions_per_event: int = 5,
) -> pd.DataFrame:
    # set the category to "Other" if it is not in the category_distribution
    df["category"] = df["category"].apply(lambda x: x if x in category_distribution else "Other")

    budget_by_category = {k: int(v * total_budget) for k, v in category_distribution.items()}
    # let the remaining budget be "Other" category
    budget_by_category["Other"] = total_budget - sum(budget_by_category.values())

    sampled_rows = []
    sampled_counts, event_counts = Counter(), Counter()
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # iterate over the rows to sample according to the budget
    for _, row in df.iterrows():
        category = row["category"]
        event_ticker = row["event_ticker"]
        # use the counter to record the number of rows sampled for each category
        if (
            sampled_counts[category] < budget_by_category[category]
            and event_counts[event_ticker] < max_submissions_per_event
        ):
            sampled_counts[category] += 1
            event_counts[event_ticker] += 1
            if sampled_counts[category] == budget_by_category[category]:
                print(f"Reached the budget for category {category}")
            if event_counts[event_ticker] == max_submissions_per_event:
                print(f"Reached the max submissions per event for event {event_ticker}")
            sampled_rows.append(row)

    sampled_df = pd.DataFrame(sampled_rows)
    return sampled_df


def augment_sampled_data(
    sampled_df: pd.DataFrame, augmented_title_df: pd.DataFrame
) -> pd.DataFrame:
    from prophet_hindsight.event.event_augment import get_event_details

    # get event_tickers
    event_tickers = sampled_df["event_ticker"].unique().tolist()
    event_details = get_event_details(event_tickers, batch_size=100)

    # Step 1: merge the "augmented_title" column with the sampled dataframe
    sampled_df = sampled_df.merge(
        augmented_title_df[["event_ticker", "augmented_title"]], on="event_ticker", how="left"
    )

    # Step 2: take the "rules" from the event_details dictionary and make it as a column in the sampled dataframe
    sampled_df["rules"] = sampled_df["event_ticker"].apply(lambda x: event_details[x][1])
    return sampled_df


def augment_sampled_data_with_sources(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the sampled dataframe with the sources of the submissions.
    """
    import json

    # get the submission ids from the sampled dataframe
    submission_ids = sampled_df["submission_id"].dropna().unique().tolist()
    assert len(submission_ids) > 0, "No submission IDs to fetch sources for"
    print(f"Fetching sources for {len(submission_ids)} submission IDs")
    # using tuple binding which works better with SQLAlchemy
    params = {f"id_{i}": str(submission_id) for i, submission_id in enumerate(submission_ids)}
    engine = get_engine()

    sql = f"""
    WITH ids AS (
      SELECT DISTINCT id::uuid AS submission_id
      FROM (VALUES {','.join([f"(:id_{i})" for i in range(len(submission_ids))])}) AS t(id)
    ),
    src_agg AS (
      SELECT
        ss.user_submission_id AS submission_id,
        jsonb_agg(
          DISTINCT jsonb_build_object(
            'summary', s.summary,
            'source_id', s.id,
            'ranking', s.popularity_score,
            'title', s.title,
            'url', s.url
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
    sources_df = pd.read_sql(text(sql), engine, params=params)
    print(f"Fetched {len(sources_df)} sources")
    print(sources_df.head())

    # Convert submission_id to string to ensure merge works properly
    sources_df["submission_id"] = sources_df["submission_id"].astype(str)
    sampled_df["submission_id"] = sampled_df["submission_id"].astype(str)

    # Convert the sources column to JSON string to properly save to CSV
    sources_df["sources"] = sources_df["sources"].apply(
        lambda x: json.dumps(x) if x is not None else "[]"
    )

    print(f"\nBefore merge - sampled_df submission_id type: {sampled_df['submission_id'].dtype}")
    print(f"Before merge - sources_df submission_id type: {sources_df['submission_id'].dtype}")
    print(f"Sample submission_id from sampled_df: {sampled_df['submission_id'].iloc[0]}")
    print(f"Sample submission_id from sources_df: {sources_df['submission_id'].iloc[0]}")

    # get the sources back to the sampled dataframe
    sampled_df = sampled_df.merge(
        sources_df[["submission_id", "sources"]], on="submission_id", how="left"
    )

    print(f"\nAfter merge - sources column null count: {sampled_df['sources'].isna().sum()}")
    print(f"After merge - sources column with data: {(~sampled_df['sources'].isna()).sum()}")

    return sampled_df


if __name__ == "__main__":
    data_file = "data/raw/after_cleanup/submissions.csv"
    df = pd.read_csv(data_file)
    # drop the "weight" column
    df = df.drop(columns=["round"])
    # add the "markets" column (by parsing the "market_outcome" column)
    # df["markets"] = df["market_outcome"].apply(lambda x: list(json.loads(x).keys()))
    category_distribution = get_category_distribution(df)
    print(category_distribution)
    # sampled_df = sample_data_according_to_category_dist(df, category_distribution, total_budget=1200, max_submissions_per_event=3)
    # # save the sampled dataframe
    # sampled_df.to_csv("data/rebuttal/sampled_submissions.csv", index=False)
    # # check how many unqiue event_tickers are in the sampled dataframe
    # print(f"Number of unique event_tickers in the sampled dataframe: {len(sampled_df['event_ticker'].unique())}")

    # sampled_df = pd.read_csv("data/rebuttal/sampled_submissions_augmented.csv")
    # augmented_title_df = pd.read_csv("data/rebuttal/augmented_event_titles.csv")
    # sampled_df = augment_sampled_data(sampled_df, augmented_title_df)
    # sampled_df.to_csv("data/rebuttal/sampled_submissions_augmented.csv", index=False)

    # get a category distribution of the augmented sampled dataframe
    # category_distribution = get_category_distribution(sampled_df, normalize=False)
    # print(category_distribution)

    # sampled_df = augment_sampled_data_with_sources(sampled_df)
    # sampled_df.to_csv("data/rebuttal/sampled_submissions_augmented_with_sources.csv", index=False)
