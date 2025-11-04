from prophet_hindsight.event.prompt import AUGMENT_EVENT_DETAIL_SYSTEM_PROMPT, AUGMENT_EVENT_DETAIL_USER_PROMPT
from prophet_hindsight.common.judge import LLMJudge, MessageBuilder
from prophet_hindsight.common.db import get_supabase_client
import asyncio
import logging
from pydantic import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_event_details(event_tickers: list[str], demo: bool = False) -> dict[str, tuple[str, str, str]]:
    """
    Get the raw event description (title), rules, and outcome (market_outcome) for each event ticker.
    If demo is True, take the first 10 event tickers.
    
    Args:
        event_tickers: List of event ticker strings to query
        demo: If True, limit to first 10 event tickers
        
    Returns:
        Dictionary mapping event_ticker -> (title, rules, market_outcome) tuple
    """
    if not event_tickers:
        return {}
    
    if demo:
        event_tickers = event_tickers[:10]
    
    supabase = get_supabase_client()
    
    # Batch query all event tickers efficiently
    response = (
        supabase.table("event")
        .select("event_ticker, title, rules, markets")
        .in_("event_ticker", event_tickers)
        .execute()
    )
    
    # Build dictionary mapping ticker to (title, rules, market_outcome)
    result = {}
    for row in response.data:
        ticker = row.get("event_ticker")
        title = row.get("title") or ""
        rules = row.get("rules") or ""
        markets = row.get("markets") or ""
        
        # Convert outcome to string if it's a dict/list
        outcomes = ", ".join(eval(markets)) if markets else "No outcome available"
        result[ticker] = (title, rules, outcomes)
    
    return result


class ModelResponse(BaseModel):
    error: bool
    event: str


def augment_event(event_tickers: list[str], judge: LLMJudge, save_path: str = None, demo: bool = False) -> list[tuple[str, str]]:
    """
    Three steps: 
    1. Use the Supabase client to retrieve the raw event description, rules, and outcome for each event ticker.
    2. Use the LLM to augment the event description.
    3. Return the augmented event description for each event ticker.

    Returns: list[tuple[str, str]] where the first string is the event ticker and the second string is the augmented event description.
    """
    if not demo:
        logger.info(f"Augmenting event details for {len(event_tickers)} event tickers")
    else:
        logger.info(f"Augmenting event details for demo (first 10 event tickers)")

    # Batch fetch all event details efficiently
    event_details = get_event_details(event_tickers, demo=demo)
    logger.info("Collected {} event details successfully".format(len(event_details)))

    # Build prompts and maintain order of event tickers to preserve alignment
    prompts = []
    ordered_tickers = []
    for ticker in event_details:
        ordered_tickers.append(ticker)
        raw_event, rules, outcomes = event_details[ticker]
        prompts.append(AUGMENT_EVENT_DETAIL_USER_PROMPT.format(raw=raw_event, rules=rules, outcome=outcomes))

    completed_results, cancelled_ids = asyncio.run(judge.async_judge(
        prompts=prompts,
        builder=MessageBuilder(system_prompt=AUGMENT_EVENT_DETAIL_SYSTEM_PROMPT),
        structure=ModelResponse,
        ids=list(range(len(prompts))),
    ))
    
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
                results.append((event_ticker, raw_title, post_process_augmented_event(augmented_title)))
    
    logger.info(f"Augmented {total} event tickers with {err} errors, success rate: {total - err}/{total}")

    if save_path:
        logger.info(f"Saving results to {save_path}")
        import pandas as pd
        df = pd.DataFrame(results, columns=["event_ticker", "title", "augmented_title"])
        df.to_csv(save_path, index=False)

    return results


def post_process_augmented_event(augmented_event: str) -> str:
    augmented_event = augmented_event.strip('"').strip("'")
    # if it begins with <event> and ends with </event>, remove the tags
    if augmented_event.startswith("<event>"):
        augmented_event = augmented_event[len("<event>"):]
    if augmented_event.endswith("</event>"):
        augmented_event = augmented_event[:-len("</event>")]
    return augmented_event


if __name__ == "__main__":
    import pandas as pd
    # Step 1: Read the event tickers from the filtered predictions
    filtered_predictions = pd.read_csv("data/raw/full_data/reasoning/hard_predictions.csv")

    event_tickers = filtered_predictions["event_ticker"].unique().tolist()
    print("Collected {} event tickers".format(len(event_tickers)))

    # Step 2: Augment the event details
    model_name = "openai/gpt-5-mini"

    judge = LLMJudge(model=model_name, use_async=True, use_openrouter=False)
    results = augment_event(event_tickers, judge, save_path="data/raw/full_data/augmented_hard_event_titles.csv", demo=False)
    print(results)