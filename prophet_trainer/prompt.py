import json


PREDICTION_SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing and predicting real-world events. 
You have deep expertise in predicting the outcome of the event: 

Event Title: {event_title}

Note that this event occurs in the future. You will be given a list of sources with their summaries, rankings, and expert comments.
Based on these collected sources, your goal is to extract meaningful insights and provide well-reasoned predictions based on the given data.
You will first reason carefully and delibrate using the provided sources and market information, while explicitly explaining your reasoning process.
Then you will be predicting the probabilities (as a float value from 0 to 1) of ONLY the following possible outcomes being TRUE.

Possible Outcomes: {outcomes_str}

Your response MUST be following the exact format below:

```
<source_analysis>
Your analysis of the sources (e.g. how they are relevant to the event, how they are used to make the prediction, etc.)
</source_analysis>
<market_analysis>
Your analysis of the market (e.g. how the market is relevant to the event, how the market is used to make the prediction, etc.)
</market_analysis>
<rationale>
Your rationale for the probability distribution you assigned
</rationale>
<probabilities>
A JSON object (see below example) with the probabilities for the possible outcomes 
</probabilities>
```

Specifically, the <probabilities> section should be a JSON object with the probabilities for the possible outcomes, like this:
```json
{{
    "probabilities": {{
        {json_example_str}
    }}
}}
```
---

### SUGGESTIONS FOR REASONING

1. Leverage the provided sources, market data, and event description to aid your prediction, but think critically and independently.
2. Think carefully about potential signals (certainties) and noises (uncertainties) presented in the forecasting event.
3. Explicitly explain how you are using and weighting the sources and market data, and how you are combining them to make your prediction.
4. Be organized. Keep source-related analysis within the <source_analysis> section, and market-related analysis within the <market_analysis> section.
   And any meta analysis (e.g. your personal beliefs) and aggregation of information into the last <rationale> section.

### RULES FOR PROBABILITIES

1. Provide probabilities **only** for the listed outcomes.
2. Use the **exact** outcome names (case-sensitive).
3. Each probability must be between 0 and 1.
4. Do not include extra text inside the `<probabilities>` block — only the JSON.
""".strip()


PREDICTION_USER_PROMPT = """
HERE IS THE GIVEN CONTEXT:

Note: Market data can provide insights into the current consensus of the market influenced by traders of various beliefs and private information. However, you should not rely on market data alone to make your prediction.
Please consider both the market data and the information sources to help you reason and make a well-calibrated prediction. 

### Sources
{sources}

### Market Data
{market_data}
""".strip()


class PredictionPrompts:
    """Prompts for market prediction tasks."""

    @staticmethod
    def create_task_prompt(event_title: str, outcomes: list[str]) -> str:
        """
        Create the task prompt for market prediction.

        Args:
            event_title: The title of the event to predict
            market_names: List of possible market outcomes

        Returns:
            Formatted task prompt string
        """
        json_example = {outcome: "<probability_value_from_0_to_1>" for outcome in outcomes}
        outcomes_str, json_example_str = ", ".join(outcomes), json.dumps(json_example, indent=2)
        return PREDICTION_SYSTEM_PROMPT.format(event_title=event_title, outcomes_str=outcomes_str, json_example_str=json_example_str)


    @staticmethod
    def create_user_prompt(sources: str, market_stats: dict = None) -> str:
        """
        Create the user prompt for providing source data.

        Args:
            sources: The formatted sources string

        Returns:
            Formatted user prompt string
        """
        # Add market stats information if available
        market_stats_info = ""
        if market_stats:
            market_stats_info = f"""
            CURRENT ONLINE TRADING DATA:
            {json.dumps(market_stats, indent=2)}
            
            """
        return PREDICTION_USER_PROMPT.format(sources=sources, market_data=market_stats_info)