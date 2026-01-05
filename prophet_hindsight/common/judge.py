"""
Implement the base LLM judge, using the `instructor` library.

Provides:
- Synchronous and asynchronous LLM calls
- Retry logic with exponential backoff
- Structured error tracking
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import instructor
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm as tqdm_sync

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)


@dataclass
class LLMError:
    """Structured error information from LLM calls."""

    task_id: int
    error_type: str  # "timeout", "rate_limit", "parse_error", "api_error", "unknown"
    message: str
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
        }


class MessageBuilder:
    """
    A basic implementation of a message builder.
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def build(self, prompt: str) -> list[dict]:
        """
        Build the messages for the LLM.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class LLMJudge:
    """
    An LLM judge with retry logic and structured error handling.

    Features:
    - Synchronous and asynchronous operation modes
    - Configurable retry with exponential backoff
    - Structured error tracking
    - Rate limit detection
    """

    def __init__(
        self,
        model: str,
        use_async: bool = False,
        use_openrouter: bool = False,
        timeout: int = 180,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_backoff: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the LLM judge.

        Args:
            model: Model identifier (e.g., "openai/gpt-5-mini")
            use_async: Whether to use async client
            use_openrouter: Whether to route through OpenRouter
            timeout: Timeout for requests in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Multiplier for exponential backoff
            initial_backoff: Initial backoff delay in seconds
            **kwargs: Additional arguments passed to instructor
        """
        self.model = model
        self.use_async = use_async
        # we always use OpenAI's own endpoint for openai models
        self.use_openrouter = use_openrouter and (not model.startswith("openai"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_backoff = initial_backoff

        # Error tracking
        self.errors: list[LLMError] = []

        provider_model = model
        if self.use_openrouter:
            provider_model = f"openrouter/{model}"

        if self.use_openrouter:
            # "Hijack" the base URL to use OpenRouter
            self.client = instructor.from_provider(
                provider_model,
                base_url=OPENROUTER_BASE_URL,
                async_client=self.use_async,
                **kwargs,
            )
        else:
            self.client = instructor.from_provider(
                provider_model, async_client=self.use_async, **kwargs
            )

    def _classify_error(self, error: Exception) -> str:
        """Classify an error into a category."""
        error_str = str(error).lower()
        if "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "rate" in error_str and "limit" in error_str:
            return "rate_limit"
        elif "parse" in error_str or "validation" in error_str or "json" in error_str:
            return "parse_error"
        elif "api" in error_str or "request" in error_str:
            return "api_error"
        else:
            return "unknown"

    def _should_retry(self, error_type: str) -> bool:
        """Determine if an error type is retryable."""
        return error_type in ["timeout", "rate_limit", "api_error"]

    def judge(
        self, prompts: list[str], builder: MessageBuilder, structure: type[BaseModel], **kwargs
    ) -> list[Any]:
        """
        Judge the prompts using the LLM with retry logic.

        Args:
            prompts: List of user prompts to process
            builder: MessageBuilder for constructing messages
            structure: Pydantic model for structured output
            **kwargs: Additional arguments for the completion call

        Returns:
            List of parsed responses
        """
        if self.use_async:
            raise NotImplementedError("Please use the async judge method instead")

        responses = []

        for i, prompt in enumerate(tqdm_sync(prompts, desc="Judging prompts")):
            messages = builder.build(prompt)

            # Retry loop
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        response_model=structure,
                        **kwargs,
                    )  # type: ignore
                    responses.append(response)
                    break
                except Exception as e:
                    error_type = self._classify_error(e)

                    if attempt < self.max_retries and self._should_retry(error_type):
                        backoff = self.initial_backoff * (self.backoff_factor**attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{self.max_retries} for prompt {i} after {backoff:.1f}s ({error_type}: {e})"
                        )
                        time.sleep(backoff)
                    else:
                        # Record error and append None
                        self.errors.append(
                            LLMError(
                                task_id=i,
                                error_type=error_type,
                                message=str(e),
                                retry_count=attempt,
                            )
                        )
                        responses.append(None)
                        logger.error(f"Failed prompt {i} after {attempt + 1} attempts: {e}")
                        break

        return responses

    async def async_judge(
        self,
        prompts: list[str],
        builder: MessageBuilder,
        structure: type | None = None,
        ids: list[int] | None = None,
        timeout: int = -1,
        **kwargs,
    ) -> tuple[dict[int, Any], list[int]]:
        """
        Judge the prompts using the LLM asynchronously in parallel.

        Args:
            prompts: List of user prompts to process
            builder: MessageBuilder for constructing messages
            structure: Pydantic model for structured output
            ids: Optional list of IDs for each prompt (defaults to indices)
            timeout: Timeout for the entire batch (defaults to self.timeout)
            **kwargs: Additional arguments for the completion call

        Returns:
            tuple[dict[int, BaseModel], list[int]]: A tuple containing:
                - A dictionary mapping task IDs to their completed results
                - A list of task IDs that were cancelled/failed
        """
        if not self.use_async:
            raise NotImplementedError("Please use the sync judge method instead")
        if ids is not None:
            assert len(ids) == len(
                prompts
            ), "The number of ids must be the same as the number of prompts"
        else:
            ids = list(range(len(prompts)))
        if timeout == -1:
            timeout = self.timeout

        # Create tasks from coroutines (asyncio.wait requires tasks, not coroutines)
        # Map each task to its corresponding ID
        task_to_id = {}
        jobs = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(
                self.client.chat.completions.create(
                    messages=builder.build(prompt),
                    response_model=structure,
                    **kwargs,
                )  # type: ignore
            )  # type: ignore
            task_to_id[task] = ids[i]
            jobs.append(task)

        # Gather and await all jobs in parallel with timeout
        done, pending = await asyncio.wait(jobs, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

        # Cancel any pending tasks if timeout occurred
        cancelled_ids = []
        for task in pending:
            task.cancel()
            task_id = task_to_id[task]
            cancelled_ids.append(task_id)
            # Record timeout error
            self.errors.append(
                LLMError(
                    task_id=task_id,
                    error_type="timeout",
                    message=f"Task timed out after {timeout}s",
                )
            )

        # Collect results from completed tasks, handling any exceptions
        completed_results = {}
        for task in done:
            task_id = task_to_id[task]
            try:
                result = task.result()
                completed_results[task_id] = result
            except Exception as e:
                # If task completed but raised an exception, treat it as failed
                cancelled_ids.append(task_id)
                error_type = self._classify_error(e)
                self.errors.append(
                    LLMError(
                        task_id=task_id,
                        error_type=error_type,
                        message=str(e),
                    )
                )

        return completed_results, cancelled_ids

    def get_errors(self) -> list[dict[str, Any]]:
        """Get all recorded errors as a list of dictionaries."""
        return [e.to_dict() for e in self.errors]

    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors = []

    def get_error_summary(self) -> dict[str, int]:
        """Get a summary of errors by type."""
        summary = {}
        for error in self.errors:
            summary[error.error_type] = summary.get(error.error_type, 0) + 1
        return summary


if __name__ == "__main__":

    class UserModel(BaseModel):
        name: str
        age: int

    judge = LLMJudge(model="gpt-5", use_async=True, use_openrouter=True)
    completed_results, cancelled_ids = asyncio.run(
        judge.async_judge(
            prompts=["Text: John Doe is 30 years old.", "Text: Jane Smith is 25 years old."],
            builder=MessageBuilder(
                system_prompt="You are a helpful assistant that extracts the name and age from a text."
            ),
            structure=UserModel,
        )
    )
    print(f"Completed: {completed_results}")
    print(f"Cancelled IDs: {cancelled_ids}")
