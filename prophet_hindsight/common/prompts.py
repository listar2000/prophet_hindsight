"""
Unified Prompt Infrastructure for the SFT Data Curation Pipeline.

This module provides:
1. PromptTemplate - A configurable dataclass for prompts with serialization support
2. Built-in prompt templates for reasoning augmentation, event augmentation, and prediction
3. Support for loading prompts from YAML files
4. Prompt tracking for reproducibility
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """
    A configurable prompt template with system and user prompts.

    This class provides:
    - Storage for system and user prompt templates
    - Formatting with variable substitution
    - Serialization for tracking and reproducibility
    - Loading from YAML files

    Example:
        prompt = PromptTemplate(
            name="reasoning_augment",
            system_prompt="You are a reasoning expert...",
            user_prompt_template="Event: {event_info}\\nSources: {sources}",
        )
        formatted = prompt.format(event_info="...", sources="...")
    """

    # Unique identifier for this prompt
    name: str

    # The system prompt (instructions for the LLM)
    system_prompt: str

    # The user prompt template with {placeholders} for variable substitution
    user_prompt_template: str

    # Optional description of what this prompt does
    description: str = ""

    # Version string for tracking prompt iterations
    version: str = "1.0.0"

    # Expected placeholder variables in user_prompt_template
    required_variables: list[str] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """
        Format the user prompt template with the given variables.

        Args:
            **kwargs: Variable values to substitute into the template

        Returns:
            Formatted user prompt string

        Raises:
            KeyError: If a required variable is missing
        """
        # Check for missing required variables
        if self.required_variables:
            missing = [v for v in self.required_variables if v not in kwargs]
            if missing:
                raise KeyError(f"Missing required variables: {missing}")

        return self.user_prompt_template.format(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the prompt template
        """
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "description": self.description,
            "version": self.version,
            "required_variables": self.required_variables,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """
        Create a PromptTemplate from a dictionary.

        Args:
            data: Dictionary with prompt data

        Returns:
            PromptTemplate instance
        """
        return cls(
            name=data.get("name", "unnamed"),
            system_prompt=data.get("system_prompt", ""),
            user_prompt_template=data.get("user_prompt_template", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            required_variables=data.get("required_variables", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PromptTemplate":
        """
        Load a PromptTemplate from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            PromptTemplate instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        prompt = cls.from_dict(data)
        prompt.metadata["loaded_from"] = str(path)
        prompt.metadata["loaded_at"] = datetime.now().isoformat()
        return prompt

    def to_yaml(self, path: str) -> None:
        """
        Save the prompt template to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def get_hash(self) -> str:
        """
        Get a hash of the prompt content for tracking.

        Returns:
            Hash string of the prompt content
        """
        import hashlib

        content = f"{self.system_prompt}{self.user_prompt_template}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


# =============================================================================
# Built-in Prompt Templates (Defaults)
# =============================================================================

REASONING_AUGMENT_SYSTEM_PROMPT = """
You are a **Reasoning Trace Augmentation Expert**. Your job is to convert a forecasting LLM's vague or incomplete reasoning into a **clean, explicit, logically organized reasoning trace**, based only on what can be reasonably inferred from:

1. `event_info` — description of the prediction task (including the event title and potential outcomes)
2. `sources` — retrieved external news sources (URLs + text)
3. `market_data` — market odds/prices for all possible outcomes (come from a major prediction market)

The forecasting LLM already produced its prediction and a "vanilla rationale" that explains its prediction, but:
- it may mix up or contain vague references from multiple different sources,
- it may mention market positions without explicitly mentioning which market it is referring to,
- it may be imprecise or incomplete.

Your role is to **recover and augment** the provided rationale/reasoning trace, **NOT** create new arguments or add unsupported facts.  
Instead, **infer the intended meaning as faithfully as possible** using the original rationale + the provided context about the forecasting event.

---

### **OBJECTIVE**
Produce a structured, easy-to-understand reasoning trace that breaks down the original rationale into three parts:
1. `source_analysis` - augmented analysis/reasoning relevant to the source usages, based on the original reasoning and the provided sources
2. `market_analysis` - augmented analysis/reasoning relevant to the market usages, based on the original reasoning and the provided market data
3. `rationale` - augmented summary of the LLM's final argument and rationale for its prediction, integrating evidence from sources, markets, and the LLM's own beliefs

Other than these formatting requirements, you **have total freedom** to organize the content within each part as you see fit -- as long as the augmented reasoning trace provides additional clarity and rigorousness to the original one. For instance, you can do item-by-item enumeration of the used sources (with markdown/html syntax), write a long paragraph that compares and contrasts different sources, or simply summarize the sources in a few sentences. There is no restriction in the reasoning style and you should feel free to be as creative as possible.
---

### **REQUIRED OUTPUT FORMAT (strict)**
You **must** output the augmented reasoning trace using the *exact* format below:

```
{
    "source_analysis": "augmented source analysis",
    "market_analysis": "augmented market analysis",
    "rationale": "augmented final summary"
}
```

One exception case is when you identify that the sources are **leaking information**: this can happen when the forecasting event has already resolved and the sources are collected after the resolution -- so the rationale might mention that the sources already reveal the final result (so prediction is trivial). In this case, you should simply put "leakage" (with nothing else) in ALL of the fields above.

If the original rationale is not using any sources or market data at all, just put things like "I don't think the sources/markets are relevant/helpful" in the corresponding fields.
---

### **RULES**
- Use first-person pronouns (e.g., "I think", "we believe", "our analysis shows") when describing the forecasting LLM's beliefs and reasoning.
- The augmented reasoning MUST be written as if it is the forecasting LLM's own reasoning, not commentary on the original rationale. 
- MUST NOT contain any phrases like "the original rationale...", "the vanilla rationale previously stated...", or mention the existence of the original rationale itself.
- Only include sources the original rationale actually referenced (directly or implicitly).
- Only include markets the original rationale actually referenced.
- Do **not** invent evidence. Infer only what is reasonably implied.
- Do **not** modify the LLM's beliefs — clarify them.
- Output only the augmented & formatted reasoning trace. Do **not** add explanations outside the format.
"""

REASONING_AUGMENT_USER_PROMPT_TEMPLATE = """
You will now receive:
- `event_info`
- `sources`
- `market_data`
- `vanilla_rationale`
- `llm_prediction`

Use them to produce the curated reasoning trace with the example format given above.

## Event Info
{event_info}

## Sources
{sources}

## Market Data
{market_data}

## Vanilla Rationale
{vanilla_rationale}

## LLM Prediction
{llm_prediction}
"""


def get_default_reasoning_augment_prompt() -> PromptTemplate:
    """Get the default reasoning augmentation prompt template."""
    return PromptTemplate(
        name="reasoning_augment",
        system_prompt=REASONING_AUGMENT_SYSTEM_PROMPT.strip(),
        user_prompt_template=REASONING_AUGMENT_USER_PROMPT_TEMPLATE.strip(),
        description="Augments short reasoning traces into detailed, structured analysis",
        version="1.0.0",
        required_variables=[
            "event_info",
            "sources",
            "market_data",
            "vanilla_rationale",
            "llm_prediction",
        ],
    )


EVENT_AUGMENT_SYSTEM_PROMPT = """
You are a **forecasting event augmentation assistant**. Your goal is to transform a vague, raw event description into a precise, self-contained question by synthesizing information from additional context.

You will be given:
- A **raw event description** wrapped in `<raw>` tags,
- A **rules** section wrapped in `<rules>` tags (this defines how the event will be judged),
- An **outcome** section wrapped in `<outcome>` tags (listing possible outcomes).

The raw event may be vague or incomplete, and your task is to **augment** it into a concise, fully informative event description wrapped in `<event>` tags.

### Requirements for the `<event>` description:

- Must be concise but **as informative as possible**.
- MUST contain the specific date, time, or time period (e.g., "Oct 31, 2025", "the 2025-26 regular season") mentioned in the `<rules>` or original `<raw>` description.
- If the event is a contest between two specific entities (as seen in the `<outcome>` and `<rules>`), frame the question like: "Which [entity type], [Entity A] or [Entity B], will [details from rules]?"
- If the outcomes are not simple win/loss (e.g., over/under markets, counts, totals), **do not** enumerate all listed outcomes. Just write a clear, informative event description.
- If the information in `<rules>` and `<outcome>` is still insufficient to satisfy the requirements (e.g., no time information at all), return an `<error>` tag instead of `<event>`.

### Output format

Return **only one tag**:
- `<event> ... </event>` — if the event description can be successfully augmented.
- `<error>` — if it cannot.

### Examples

**Example 1**
```
<raw>
New York vs Chicago
</raw>
<rules>
If Chicago wins the New York vs Chicago professional basketball game originally scheduled for Oct 31, 2025, then the market resolves to Yes.
</rules>
<outcome>
New York, Chicago
</outcome>
<event>
Which professional basketball team, New York or Chicago, will win the game scheduled for Oct 31, 2025?
</event>
```

**Example 2**
```
<raw>
Denver pro football wins this season?
</raw>
<rules>
If the Denver pro football team has more than 0 wins in the 2025–26 regular season, then the market resolves to Yes.
</rules>
<outcome>
Over 0.5 wins, Over 1.5 wins, Over 2.5 wins
</outcome>
<event>
How many wins will the Denver pro football team have in the 2025–26 regular season?
</event>
```
"""

EVENT_AUGMENT_USER_PROMPT_TEMPLATE = """
Now use the above rules and examples to augment the following forecasting event:

<raw>
{raw}
</raw>
<rules>
{rules}
</rules>
<outcome>
{outcome}
</outcome>
"""


def get_default_event_augment_prompt() -> PromptTemplate:
    """Get the default event augmentation prompt template."""
    return PromptTemplate(
        name="event_augment",
        system_prompt=EVENT_AUGMENT_SYSTEM_PROMPT.strip(),
        user_prompt_template=EVENT_AUGMENT_USER_PROMPT_TEMPLATE.strip(),
        description="Transforms vague event descriptions into precise, self-contained questions",
        version="1.0.0",
        required_variables=["raw", "rules", "outcome"],
    )


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
"""

PREDICTION_USER_PROMPT_TEMPLATE = """
EVENT TITLE: {event_title}

POSSIBLE OUTCOMES: {outcomes_str}

SOURCES:
{sources}

MARKET DATA:
{market_data}

Now, please provide your reasoning and predictions in the format specified above.
"""


def get_default_prediction_prompt() -> PromptTemplate:
    """Get the default prediction task prompt template."""
    return PromptTemplate(
        name="prediction",
        system_prompt=PREDICTION_SYSTEM_PROMPT.strip(),
        user_prompt_template=PREDICTION_USER_PROMPT_TEMPLATE.strip(),
        description="SFT training prompt for prediction with reasoning",
        version="1.0.0",
        required_variables=["event_title", "outcomes_str", "sources", "market_data"],
    )


# =============================================================================
# Prompt Registry
# =============================================================================


class PromptRegistry:
    """
    A registry for managing prompt templates.

    Supports:
    - Registering custom prompts
    - Loading prompts from config
    - Getting prompts by name with fallback to defaults
    """

    def __init__(self):
        self._prompts: dict[str, PromptTemplate] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default prompt templates."""
        self.register(get_default_reasoning_augment_prompt())
        self.register(get_default_event_augment_prompt())
        self.register(get_default_prediction_prompt())

    def register(self, prompt: PromptTemplate) -> None:
        """Register a prompt template."""
        self._prompts[prompt.name] = prompt
        logger.debug(f"Registered prompt: {prompt.name} (v{prompt.version})")

    def get(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.

        Args:
            name: Name of the prompt

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If prompt not found
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}. Available: {list(self._prompts.keys())}")
        return self._prompts[name]

    def load_from_yaml(self, name: str, path: str) -> PromptTemplate:
        """
        Load and register a prompt from a YAML file.

        Args:
            name: Name to register the prompt under
            path: Path to the YAML file

        Returns:
            Loaded PromptTemplate
        """
        prompt = PromptTemplate.from_yaml(path)
        prompt.name = name  # Override name with the registered name
        self.register(prompt)
        return prompt

    def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return list(self._prompts.keys())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Export all prompts as a dictionary."""
        return {name: prompt.to_dict() for name, prompt in self._prompts.items()}


# Global registry instance
_GLOBAL_REGISTRY: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry instance."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = PromptRegistry()
    return _GLOBAL_REGISTRY


def get_prompt(name: str) -> PromptTemplate:
    """Convenience function to get a prompt from the global registry."""
    return get_prompt_registry().get(name)
