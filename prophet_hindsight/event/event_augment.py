import ast
import asyncio
import json
import logging

from pydantic import BaseModel

from prophet_hindsight.common.db import get_supabase_client
from prophet_hindsight.common.judge import LLMJudge, MessageBuilder
from prophet_hindsight.common.prompts import PromptTemplate, get_default_event_augment_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _safe_parse_markets(markets_str: str) -> str:
    """
    Safely parse markets string without using eval().

    Args:
        markets_str: String representation of markets list/dict

    Returns:
        Comma-separated string of outcomes
    """
    if not markets_str:
        return "No outcome available"

    try:
        # Try JSON parsing first
        markets = json.loads(markets_str)
    except json.JSONDecodeError:
        try:
            # Fall back to ast.literal_eval (safe alternative to eval)
            markets = ast.literal_eval(markets_str)
        except (ValueError, SyntaxError):
            logger.warning(f"Could not parse markets string: {markets_str[:100]}...")
            return "No outcome available"

    if isinstance(markets, list):
        return ", ".join(str(m) for m in markets)
    elif isinstance(markets, dict):
        return ", ".join(str(k) for k in markets.keys())
    else:
        return str(markets)


def get_event_details(
    event_tickers: list[str], demo: bool = False, batch_size: int = 100
) -> dict[str, tuple[str, str, str]]:
    """
    Get the raw event description (title), rules, and outcome (market_outcome) for each event ticker.
    If demo is True, take the first 10 event tickers.

    Args:
        event_tickers: List of event ticker strings to query
        demo: If True, limit to first 10 event tickers
        batch_size: Number of tickers to query per request (to avoid URL length limits)

    Returns:
        Dictionary mapping event_ticker -> (title, rules, market_outcome) tuple
    """
    if not event_tickers:
        return {}

    if demo:
        event_tickers = event_tickers[:10]

    supabase = get_supabase_client()

    # Batch query event tickers in chunks to avoid URL length limits
    result = {}
    total_batches = (len(event_tickers) + batch_size - 1) // batch_size

    for i in range(0, len(event_tickers), batch_size):
        batch = event_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} event tickers)")

        response = (
            supabase.table("event")
            .select("event_ticker, title, rules, markets")
            .in_("event_ticker", batch)
            .execute()
        )

        # Build dictionary mapping ticker to (title, rules, market_outcome)
        for row in response.data:
            ticker = row.get("event_ticker")
            title = row.get("title") or ""
            rules = row.get("rules") or ""
            markets = row.get("markets") or ""

            # Convert outcome to string safely (no eval!)
            outcomes = _safe_parse_markets(markets)
            result[ticker] = (title, rules, outcomes)

    return result


class ModelResponse(BaseModel):
    error: bool
    event: str


def augment_event(
    event_tickers: list[str],
    judge: LLMJudge,
    prompt: PromptTemplate | None = None,
    save_path: str | None = None,
    demo: bool = False,
) -> list[tuple[str, str, str]]:
    """
    Three steps:
    1. Use the Supabase client to retrieve the raw event description, rules, and outcome for each event ticker.
    2. Use the LLM to augment the event description.
    3. Return the augmented event description for each event ticker.

    Args:
        event_tickers: List of event ticker strings
        judge: LLMJudge instance for making LLM calls
        prompt: Optional PromptTemplate to use (defaults to built-in prompt)
        save_path: Optional path to save results
        demo: If True, only process first 10 tickers

    Returns:
        list[tuple[str, str, str]] - (event_ticker, raw_title, augmented_title)
    """
    # Use default prompt if none provided
    if prompt is None:
        prompt = get_default_event_augment_prompt()

    if not demo:
        logger.info(f"Augmenting event details for {len(event_tickers)} event tickers")
    else:
        logger.info("Augmenting event details for demo (first 10 event tickers)")

    # Batch fetch all event details efficiently
    event_details = get_event_details(event_tickers, demo=demo)
    logger.info(f"Collected {len(event_details)} event details successfully")

    # Build prompts and maintain order of event tickers to preserve alignment
    user_prompts = []
    ordered_tickers = []
    for ticker in event_details:
        ordered_tickers.append(ticker)
        raw_event, rules, outcomes = event_details[ticker]
        # Use the prompt template to format the user prompt
        user_prompts.append(prompt.format(raw=raw_event, rules=rules, outcome=outcomes))

    completed_results, cancelled_ids = asyncio.run(
        judge.async_judge(
            prompts=user_prompts,
            builder=MessageBuilder(system_prompt=prompt.system_prompt),
            structure=ModelResponse,
            ids=list(range(len(user_prompts))),
        )
    )

    if cancelled_ids:
        logger.warning(f"Failed to get responses for {len(cancelled_ids)} prompts: {cancelled_ids}")

    results = []
    err, total = 0, 0
    for i, event_ticker in enumerate(ordered_tickers):
        total += 1
        raw_title = event_details[event_ticker][0]

        if i not in completed_results:
            # Task was cancelled or failed
            err += 1
            logger.warning(f"Skipping augmentation for event_ticker {event_ticker} (index {i})")
            results.append((event_ticker, raw_title, ""))
        else:
            response = completed_results[i]
            if response.error:
                err += 1
                results.append((event_ticker, raw_title, ""))
            else:
                augmented_title = response.event.strip()  # Fixed: strip() is a method call
                results.append(
                    (event_ticker, raw_title, post_process_augmented_event(augmented_title))
                )

    logger.info(
        f"Augmented {total} event tickers with {err} errors, success rate: {total - err}/{total}"
    )

    if save_path:
        logger.info(f"Saving results to {save_path}")
        import pandas as pd

        # Convert list of tuples to DataFrame
        df = pd.DataFrame(
            data=results,
            columns=["event_ticker", "title", "augmented_title"],  # type: ignore
        )
        df.to_csv(save_path, index=False)

    return results


def post_process_augmented_event(augmented_event: str) -> str:
    augmented_event = augmented_event.strip('"').strip("'")
    # if it begins with <event> and ends with </event>, remove the tags
    if augmented_event.startswith("<event>"):
        augmented_event = augmented_event[len("<event>") :]
    if augmented_event.endswith("</event>"):
        augmented_event = augmented_event[: -len("</event>")]
    return augmented_event


if __name__ == "__main__":
    import pandas as pd

    # Step 1: Read the event tickers from a df
    df = pd.read_csv("data/rebuttal/sampled_submissions.csv")

    event_tickers = df["event_ticker"].unique().tolist()
    print(f"Collected {len(event_tickers)} event tickers")

    # Step 2: Augment the event details
    model_name = "openai/gpt-5-mini"

    judge = LLMJudge(model=model_name, use_async=True, use_openrouter=False)

    # Use default prompt (can also load custom prompt from YAML)
    prompt = get_default_event_augment_prompt()
    print(f"Using prompt: {prompt.name} v{prompt.version}")

    results = augment_event(
        event_tickers,
        judge,
        prompt=prompt,
        save_path="data/rebuttal/augmented_event_titles.csv",
        demo=False,
    )
    print(results)
