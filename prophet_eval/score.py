import pandas as pd
import numpy as np
import json
import json_repair
import re
import os


def calculate_brier_score(predictions: dict, outcomes: dict) -> float:
    if len(predictions) != len(outcomes):
        print("Warning: predictions and outcomes have different lengths")
    sorted_keys = sorted(outcomes.keys())
    predictions = np.array([float(predictions[k]) if k in predictions else 0.0 for k in sorted_keys])
    outcomes = np.array([int(outcomes[k]) for k in sorted_keys])

    if not np.all(np.isin(outcomes, [0, 1])):
        return float("nan")
    if not np.all(predictions >= 0) or not np.all(predictions <= 1):
        return float("nan")
    return float(np.mean((predictions - outcomes) ** 2))


def parse_prediction(raw_prediction: str, market_outcome_str: str) -> tuple[dict[str, float], dict[str, int]]:
    prediction_dict = json_repair.loads(raw_prediction)
    assert isinstance(prediction_dict, dict) and "content" in prediction_dict, \
        "prediction must be a dictionary with a 'content' key"
    content = prediction_dict["content"]
    market_outcome = json_repair.loads(market_outcome_str)

    try:
        # Extract the string within <probabilities> tags
        prob_str = re.search(r"<probabilities>(.*)</probabilities>", content)
        if prob_str:
            prob_str = prob_str.group(1)
        elif "</probabilities>" in content:  # we just find the part before the end tag </probabilities>
            prob_str = content.split("</probabilities>")[0]
        else:  # directly use the content
            prob_str = content
        prediction_dict = json_repair.loads(prob_str)
    except Exception as e:
        prediction_dict = None
    return prediction_dict, market_outcome


def score_predictions(pred_path: str, outcome_path: str | None = None):
    # take the outcome path to be the same folder as the pred path if not provided
    if outcome_path is None:
        outcome_path = pred_path.split(".")[0] + "_brier_score.csv"

    pred_df = pd.read_csv(pred_path)
    total_rows = len(pred_df)
    
    # outcome df should have five columns: event_ticker, submission_id, market_outcome, prediction, brier_score
    # where the first 2 comes directly from pred_df, and next 2 from parse_prediction, last one from calculate_brier_score
    parsed = pred_df.apply(lambda row: parse_prediction(row["prediction"], row["market_outcome"]), axis=1)
    outcome_df = pd.DataFrame({
        "event_ticker": pred_df.get("event_ticker"),
        "submission_id": pred_df.get("submission_id"),
        "prediction": parsed.apply(lambda x: x[0]),
        "market_outcome": parsed.apply(lambda x: x[1]),
        "brier_score": float("nan")  # initialize all as NaN
    })

    # count how many valid "prediction" dict (i.e. not None) we have
    is_valid = outcome_df["prediction"].notna()
    valid_rows = is_valid.sum()

    # calculate the brier score only for the valid predictions
    outcome_df.loc[is_valid, "brier_score"] = outcome_df[is_valid].apply(
        lambda row: calculate_brier_score(row["prediction"], row["market_outcome"]), axis=1
    )

    # count how many rows have normal brier score (i.e. not nan)
    normal_brier_scores = outcome_df["brier_score"].notna().sum()

    print(f"Number of valid predictions: {valid_rows}, out of {total_rows}, percentage: {valid_rows / total_rows}")
    print(f"Number of normal brier scores: {normal_brier_scores}, out of {total_rows}, percentage: {normal_brier_scores / total_rows}")
    print(f"Average brier score: {outcome_df['brier_score'].mean(skipna=True)}")

    outcome_df.to_csv(outcome_path, index=False)
    print(f"Saved brier scores to {outcome_path}")
    return outcome_df


def summarize_scored_predictions(brier_score_df: pd.DataFrame) -> dict:
    validity_summary = {
        "number_of_valid_predictions": brier_score_df["prediction"].notna().sum(),
        "number_of_normal_brier_scores": brier_score_df["brier_score"].notna().sum(),
        "total_number_of_predictions": len(brier_score_df),
        "total_number_of_events": brier_score_df["event_ticker"].nunique(),
        "total_number_of_submissions": brier_score_df["submission_id"].nunique(),
    }
    brier_score_summary = {
        "average_brier_score": brier_score_df["brier_score"].mean(skipna=True),
        "median_brier_score": brier_score_df["brier_score"].median(skipna=True),
        "min_brier_score": brier_score_df["brier_score"].min(skipna=True),
        "max_brier_score": brier_score_df["brier_score"].max(skipna=True),
        "std_brier_score": brier_score_df["brier_score"].std(skipna=True),
    }
    # calculate event-averaged brier score (i.e. we group by the event_ticker first and do an average within group first)
    event_averaged_brier_score = brier_score_df.groupby("event_ticker")["brier_score"].mean()
    event_averaged_brier_score_summary = {
        "average_event_averaged_brier_score": event_averaged_brier_score.mean(skipna=True),
        "median_event_averaged_brier_score": event_averaged_brier_score.median(skipna=True),
        "min_event_averaged_brier_score": event_averaged_brier_score.min(skipna=True),
        "max_event_averaged_brier_score": event_averaged_brier_score.max(skipna=True),
        "std_event_averaged_brier_score": event_averaged_brier_score.std(skipna=True),
    }
    brier_score_summary.update(event_averaged_brier_score_summary)
    return validity_summary, brier_score_summary


def summarize_all_scored_predictions(score_path: str, outcome_path: str | None = None):
    # step 1: determine whether score path is a dir or a file, if it's a dir we extract all the score files in the dir
    if os.path.isdir(score_path):
        # iteratively collect all the score files in the dir
        score_files = []
        for root, dirs, files in os.walk(score_path):
            for file in files:
                if file.endswith("brier_score.csv"):
                    score_files.append(os.path.join(root, file))
    else:
        score_files = [score_path]
    
    results = []
    # step 2: for each score file, we summarize the predictions
    for score_file in score_files:
        brier_score_df = pd.read_csv(score_file)
        validity_summary, brier_score_summary = summarize_scored_predictions(brier_score_df)
        results.append({
            "score_file": score_file,
            "validity_summary": validity_summary,
            "brier_score_summary": brier_score_summary,
        })
    
    if outcome_path is None:
        outcome_dir = score_path if os.path.isdir(score_path) else os.path.dirname(score_path)
        outcome_path = os.path.join(outcome_dir, "brier_score_summary.json")

    pd.DataFrame(results).to_json(outcome_path, orient="records", indent=2)
    print(f"Saved brier score summary to {outcome_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--outcome_path", type=str)
    parser.add_argument("--score_path", type=str)
    args = parser.parse_args()

    if args.pred_path is not None:
        score_predictions(args.pred_path, args.outcome_path)
    if args.score_path is not None:
        summarize_all_scored_predictions(args.score_path, args.outcome_path)
    else:
        print("No score path provided, exiting")