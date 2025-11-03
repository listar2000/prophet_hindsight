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
    def __init__(self, model: str, use_async: bool = False, use_openrouter: bool = False, **kwargs):
        self.model = model
        self.use_async = use_async
        self.use_openrouter = use_openrouter

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

    
    async def async_judge(self, prompts: list[str], builder: MessageBuilder, structure: BaseModel = None, timeout: int = 180, **kwargs) -> list[BaseModel]:
        """
        Judge the prompts using the LLM asynchronously in parallel.
        """
        if not self.use_async:
            raise NotImplementedError("Please use the sync judge method instead")

        # Create tasks from coroutines (asyncio.wait requires tasks, not coroutines)
        jobs = [asyncio.create_task(self.client.chat.completions.create(
            messages=builder.build(prompt),
            response_model=structure,
            **kwargs,
        )) for prompt in prompts]

        # Gather and await all jobs in parallel with timeout
        done, pending = await asyncio.wait(jobs, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
        
        # Cancel any pending tasks if timeout occurred
        for task in pending:
            task.cancel()
        
        # Return results from all completed tasks
        return [task.result() for task in done]


if __name__ == "__main__":
    class UserModel(BaseModel):
        name: str
        age: int

    judge = LLMJudge(model="gpt-5", use_async=True, use_openrouter=True)
    async_responses = asyncio.run(judge.async_judge(
        prompts=["Text: John Doe is 30 years old.", "Text: Jane Smith is 25 years old."],
        builder=MessageBuilder(system_prompt="You are a helpful assistant that extracts the name and age from a text."),
        structure=UserModel,
    ))
    print(async_responses)