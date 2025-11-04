"""
Implement the base LLM judge, using the `instructor` library.
"""
from pydantic import BaseModel
import instructor
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm as tqdm_sync
import asyncio


load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        return messages


class LLMJudge:
    """
    A basic implementation of an LLM judge.
    """
    def __init__(self, model: str, use_async: bool = False, use_openrouter: bool = False, timeout: int = 180, **kwargs):
        self.model = model
        self.use_async = use_async
        self.use_openrouter = use_openrouter
        self.timeout = timeout

        if self.use_openrouter:
            model = f"openrouter/{model}"

        if self.use_openrouter:
            # "Hijack" the base URL to use OpenRouter
            self.client = instructor.from_provider(
                model,
                base_url=OPENROUTER_BASE_URL,
                async_client=self.use_async,
                **kwargs,
            )
        else:
            self.client = instructor.from_provider(model, async_client=self.use_async, **kwargs)

    def judge(self, prompts: list[str], builder: MessageBuilder, structure: BaseModel, **kwargs) -> list[BaseModel]:
        """
        Judge the prompts using the LLM.
        """
        if self.use_async:
            raise NotImplementedError("Please use the async judge method instead")

        responses = []

        for prompt in tqdm_sync(prompts, desc="Judging prompts"):
            messages = builder.build(prompt)
            response = self.client.chat.completions.create(
                messages=messages,
                response_model=structure,
                **kwargs,
            )
            responses.append(response)

        return responses

    
    async def async_judge(self, prompts: list[str], builder: MessageBuilder, structure: BaseModel = None, ids: list[int] = None, timeout: int = -1, **kwargs) -> tuple[dict[int, BaseModel], list[int]]:
        """
        Judge the prompts using the LLM asynchronously in parallel.
        
        Returns:
            tuple[dict[int, BaseModel], list[int]]: A tuple containing:
                - A dictionary mapping task IDs to their completed results
                - A list of task IDs that were cancelled/failed
        """
        if not self.use_async:
            raise NotImplementedError("Please use the sync judge method instead")
        if ids is not None:
            assert len(ids) == len(prompts), "The number of ids must be the same as the number of prompts"
        else:
            ids = list(range(len(prompts)))
        if timeout == -1:
            timeout = self.timeout

        # Create tasks from coroutines (asyncio.wait requires tasks, not coroutines)
        # Map each task to its corresponding ID
        task_to_id = {}
        jobs = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(self.client.chat.completions.create(
                messages=builder.build(prompt),
                response_model=structure,
                **kwargs,
            ))
            task_to_id[task] = ids[i]
            jobs.append(task)

        # Gather and await all jobs in parallel with timeout
        done, pending = await asyncio.wait(jobs, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
        
        # Cancel any pending tasks if timeout occurred
        cancelled_ids = []
        for task in pending:
            task.cancel()
            cancelled_ids.append(task_to_id[task])
        
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
        
        return completed_results, cancelled_ids


if __name__ == "__main__":
    class UserModel(BaseModel):
        name: str
        age: int

    judge = LLMJudge(model="gpt-5", use_async=True, use_openrouter=True)
    completed_results, cancelled_ids = asyncio.run(judge.async_judge(
        prompts=["Text: John Doe is 30 years old.", "Text: Jane Smith is 25 years old."],
        builder=MessageBuilder(system_prompt="You are a helpful assistant that extracts the name and age from a text."),
        structure=UserModel,
    ))
    print(f"Completed: {completed_results}")
    print(f"Cancelled IDs: {cancelled_ids}")