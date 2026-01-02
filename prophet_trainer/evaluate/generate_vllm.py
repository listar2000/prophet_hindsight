from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def get_evaluation_prompts(dataset_name: str, dataset_split: str, max_pred_single_event: int = 5):
    dataset = load_dataset(dataset_name, split=dataset_split)

    # do a little filtering that we only allow each event_ticker to appear at most max_pred_single_event times
    # this is done by first doing some grouping and then filtering
    df = dataset.to_pandas()
    df = df.groupby("event_ticker").head(max_pred_single_event)
    df = df.reset_index(drop=True)
    prompts = df["messages"].tolist()
    # discard the assistant message (last message)
    prompts = [prompt.tolist()[:2] for prompt in prompts]
    print(
        f"Due to the max {max_pred_single_event} predictions/event limit, we will only evaluate on {len(df)} events"
    )
    return prompts


def decode_think_and_content(generated_text: str, think_str_end: str = "</think>"):
    # use think_str_end to find the index of the end of the think section
    think_end_idx = generated_text.find(think_str_end) + len(think_str_end)
    if think_end_idx == -1:
        return None
    think_text = generated_text[:think_end_idx].strip()
    content_text = generated_text[think_end_idx:].strip()
    return {
        "think": think_text,
        "content": content_text,
    }


def get_model_outputs(llm: LLM, prompts: list[list[str]], sampling_params: SamplingParams):
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)  # type: ignore

    generated_outputs = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        decoded_output = decode_think_and_content(generated_text)
        if decoded_output is not None:
            generated_outputs.append(decoded_output)
        else:
            print(f"Warning: could not decode think and content for the {i}-th output")
            generated_outputs.append({"think": None, "content": generated_text})

    return generated_outputs


def main(
    model_path: str,
    output_path: str,
    dataset_name: str,
    dataset_split: str,
    max_pred_single_event: int = 5,
    **kwargs,
):
    prompts = get_evaluation_prompts(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        max_pred_single_event=max_pred_single_event,
    )

    # Extract LLM initialization parameters from kwargs
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 2)
    gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Extract sampling parameters from kwargs
    max_tokens = kwargs.get("max_tokens", 4096)
    temperature = kwargs.get("temperature", 0.6)
    top_p = kwargs.get("top_p", 0.95)
    repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    # Create a structured outputs params regex that captures the `<probabilities></probabilities>` pattern
    structured_outputs_regex = r"<probabilities>(.*?)</probabilities>"
    structured_outputs_params = StructuredOutputsParams(regex=structured_outputs_regex)

    sampling_params = SamplingParams(
        structured_outputs=structured_outputs_params,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    generated_outputs = get_model_outputs(llm, prompts, sampling_params)

    # if output_path is provided, save the generated outputs to a csv file
    if output_path is not None:
        import pandas as pd

        df = pd.DataFrame(generated_outputs)
        df.to_csv(output_path, index=False)
        print(f"Saved generated outputs to {output_path}")
    return generated_outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a language model on sports prediction tasks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model to evaluate (e.g., '/net/scratch2/listar2000/prophet-hindsight/.cache/torchtune/Qwen3-8B')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the evaluation results as CSV (e.g., 'data/evals/qwen3-8b.csv')",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="listar2000/sports_augmented_sft",
        help="Name of the dataset to evaluate on (e.g., 'listar2000/sports_augmented_sft')",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Split of the dataset to evaluate on (e.g., 'test')",
    )
    parser.add_argument(
        "--max_pred_single_event",
        type=int,
        default=5,
        help="Maximum number of predictions per event (default: 5)",
    )

    # Sampling parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)"
    )
    # LLM initialization parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization ratio (default: 0.8)",
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_pred_single_event=args.max_pred_single_event,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        repetition_penalty=args.repetition_penalty,
    )
