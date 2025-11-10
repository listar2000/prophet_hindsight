# Adapted from https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os

from accelerate import logging, PartialState
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.get_logger(__name__)


def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model init kwargs
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load the dataset with proper synchronization for distributed training
    # Use main_process_first to ensure dataset is fully loaded before other ranks access it
    state = PartialState()
    
    with state.main_process_first():
        if dataset_args.datasets and script_args.dataset_name:
            logger.warning(
                "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
                "dataset and `dataset_name` will be ignored."
            )
            dataset = get_dataset(dataset_args)
        elif dataset_args.datasets and not script_args.dataset_name:
            dataset = get_dataset(dataset_args)
        elif not dataset_args.datasets and script_args.dataset_name:
            dataset = load_dataset(
                script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
            )
        else:
            raise ValueError("Either `datasets` or `dataset_name` must be provided.")
    
    # Verify required splits exist after all processes have loaded the dataset
    logger.info(f"Available dataset splits on rank {state.process_index}: {list(dataset.keys())}")
    if script_args.dataset_train_split not in dataset:
        raise ValueError(f"Train split '{script_args.dataset_train_split}' not found in dataset. Available: {list(dataset.keys())}")
    if training_args.eval_strategy != "no" and script_args.dataset_test_split not in dataset:
        raise ValueError(f"Test split '{script_args.dataset_test_split}' not found in dataset. Available: {list(dataset.keys())}")

    # Shuffle the dataset
    train_dataset = dataset[script_args.dataset_train_split].shuffle(seed=42)
    if training_args.eval_strategy != "no":
        test_dataset = dataset[script_args.dataset_test_split].shuffle(seed=42)
    else:
        test_dataset = None
    
    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    main(script_args, training_args, model_args, dataset_args)