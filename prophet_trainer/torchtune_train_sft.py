"""
Torchtune version of the SFT training script.
"""
from torchtune.models.qwen3 import qwen3_tokenizer
from torchtune.datasets import chat_dataset

MODEL_PATH = ".cache/torchtune/Qwen3-8B"

tokenizer = qwen3_tokenizer(
    path=f"{MODEL_PATH}/vocab.json",
    merges_file=f"{MODEL_PATH}/merges.txt",
    special_tokens_path=f"{MODEL_PATH}/tokenizer.json",
    max_seq_len=8192,
)

ds = chat_dataset(
    tokenizer=tokenizer,
    source="listar2000/sports_augmented_sft",
    conversation_column="messages",
    conversation_style="openai",
    split="train",
)