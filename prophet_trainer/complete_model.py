#!/usr/bin/env python3
"""
Complete the converted model by adding config and tokenizer files from base model.
"""

import argparse
import os
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer


def complete_model(base_model: str, output_dir: str):
    """
    Add config and tokenizer files from base model to the converted model directory.
    
    Args:
        base_model: Base model name (e.g., "Qwen/Qwen3-8B")
        output_dir: Directory containing model.safetensors
    """
    print(f"Completing model in {output_dir}...")
    print(f"Copying config and tokenizer from {base_model}...")
    
    # Load and save config
    print("  - Downloading and saving config.json...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(output_dir)
    
    # Load and save tokenizer
    print("  - Downloading and saving tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Model is now complete and ready for HuggingFace upload!")
    print(f"\nFiles in {output_dir}:")
    for file in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, file))
        size_str = f"{size / (1024**3):.2f} GB" if size > 1e9 else f"{size / (1024**2):.2f} MB"
        print(f"  - {file} ({size_str})")
    
    print("\n📤 To upload to HuggingFace Hub:")
    print(f"   huggingface-cli upload <your-username>/<repo-name> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete converted model with config and tokenizer files"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing model.safetensors",
    )
    
    args = parser.parse_args()
    complete_model(args.base_model, args.output_dir)


if __name__ == "__main__":
    main()

