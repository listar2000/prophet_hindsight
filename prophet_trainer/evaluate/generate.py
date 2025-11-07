
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse
import torch


THINK_TOKEN_ID = 151668


def get_hf_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # saving memories as we do inference only
    model.eval()
    return model, tokenizer


def decode_think_and_content(output_ids: list[int], tokenizer: AutoTokenizer):
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
    except ValueError:
        index = 0
    think_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return {
        "think": think_content,
        "content": content,
    }


def get_model_outputs(model, tokenizer, messages: list[list[dict]]):
    """
    Get the outputs of the model for the given messages.
    Each message in `messages` is a list of dicts, each with a `role` and `content` key.
    The `role` can be "system", "user", or "assistant".
    The `content` is the text of the message.

    Returns:
        list[str]: The outputs of the model for the given messages.
    """
    prompts = [message[:2] for message in messages]  # discard the assistant message (last message)

    texts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for prompt in prompts
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=4096, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_p=0.95
    )

    outputs = []
    for i in range(len(messages)):
        output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
        outputs.append(decode_think_and_content(output_ids, tokenizer))

    return outputs


def get_evaluation_prompts(max_pred_single_event: int = 5):
    DATASET_NAME = "listar2000/sports_augmented_sft_v2"
    DATASET_SPLIT = "test"
    dataset = load_dataset(DATASET_NAME)[DATASET_SPLIT]
    
    # do a little filtering that we only allow each event_ticker to appear at most max_pred_single_event times
    # this is done by first doing some grouping and then filtering
    df = dataset.to_pandas()
    df = df.groupby("event_ticker").head(max_pred_single_event)
    df = df.reset_index(drop=True)
    prompts = df["messages"].tolist()
    prompts = [prompt.tolist() for prompt in prompts]
    print(f"Due to the max {max_pred_single_event} predictions/event limit, we will only evaluate on {len(df)} events")
    return prompts


def generate_model_outputs(model_name: str, batch_size: int = 16, save_path: str = None):
    """
    Evaluate the model, save the results to a csv file if `save_path` is provided, in a batched manner.
    """
    model, tokenizer = get_hf_model(model_name)
    prompts = get_evaluation_prompts()
    total_batches = len(prompts) // batch_size + 1

    batch_results = []
    for i in tqdm(range(total_batches)):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_outputs = get_model_outputs(model, tokenizer, batch_prompts)
        batch_results.extend(batch_outputs)

    batch_results_df = pd.DataFrame(batch_results)
    if save_path is not None:
        batch_results_df.to_csv(save_path, index=False)

    return batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a language model on sports prediction tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the HuggingFace model to evaluate (e.g., 'Qwen/Qwen3-8B')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the evaluation results as CSV (default: None)"
    )
    
    args = parser.parse_args()
    
    generate_model_outputs(
        model_name=args.model_name,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
