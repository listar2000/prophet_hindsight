import json


PREDICTION_SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing and predicting real-world events. 
You have deep expertise in predicting the **probabilities** that each outcomes of a given event will be TRUE.

You will be given an event title with all potential outcomes listed.
You will also be given a list of sources that an external searcher has collected, and the prediction market data related to the event.
Based on these collected sources and market data, your goal is to extract meaningful insights and provide well-reasoned probablistic predictions based on the given data.

Your response MUST be structured as two sections: Think Section (<think></think>) and Probabilities Section (<probabilities></probabilities>).

In the Think Section, detail your reasoning process using the following format:
```
<think>
## Source Analysis
Your analysis of the sources (e.g. how they are relevant to the event, how they are used to make the prediction, etc.)

## Market Analysis
Your analysis of the market (e.g. how the market is relevant to the event, how the market is used to make the prediction, etc.)

## Rationale
Summarize the above analyses, add extra thinking details, and justify your predictions.
</think>

In the Probabilities Section, provide a JSON object with the predicted probabilities that each of the possible outcomes will be TRUE, like this:
```
<probabilities>
{{
    "probabilities": {{
        "outcome_a": <probability_value_from_0_to_1>,
        "outcome_b": <probability_value_from_0_to_1>,
        ...
    }}
}}
</probabilities>
```

### RULES FOR THINK SECTION

1. Leverage the provided sources, market data, and event description to aid your prediction, but think critically and independently.
2. Think carefully about potential signals (certainties) and noises (uncertainties) presented in the forecasting event.
3. Explicitly explain how you are using and weighting the sources and market data, and how you are combining them to make your prediction.
4. Be organized. Keep source-related analysis within the "Source Analysis" section, and market-related analysis within the "Market Analysis" section.
   And any meta analysis (e.g. your personal beliefs) and aggregation of information into the last "Rationale" section.

### RULES FOR PROBABILITIES SECTION

1. Provide probabilities **only** for all the listed potential outcomes.
2. Use the **exact** outcome names (case-sensitive).
3. Each probability must be between 0 and 1.
4. Do not include extra text inside the `<probabilities>` block — only the JSON.
""".strip()


PREDICTION_USER_PROMPT = """
EVENT TITLE: {event_title}

POSSIBLE OUTCOMES: {outcomes_str}

SOURCES:
{sources}

MARKET DATA:
{market_data}

Now, please provide your reasoning and predictions in the format specified above.
""".strip()


class PredictionPrompts:
    """Prompts for market prediction tasks."""

    @staticmethod
    def create_task_prompt() -> str:
        """
        Create the task prompt for market prediction.
        """
        return PREDICTION_SYSTEM_PROMPT


    @staticmethod
    def create_user_prompt(event_title: str, outcomes_str: str, sources: str, market_data: dict = None) -> str:
        """
        Create the user prompt for providing source data.

        Args:
            sources: The formatted sources string

        Returns:
            Formatted user prompt string
        """
        # Add market stats information if available
        market_stats_info = ""
        if market_data:
            market_data_info = f"""
            MARKET DATA FROM A MAJOR PREDICTION MARKET:
            {json.dumps(market_data, indent=2)}
            
            """.strip()
        return PREDICTION_USER_PROMPT.format(event_title=event_title, outcomes_str=outcomes_str, sources=sources, market_data=market_data_info)