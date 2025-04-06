import argparse
from transformers import (
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperProcessor,
)
from peft import PeftModel
from pathlib import Path
from huggingface_hub import HfApi
import torch
import subprocess
import os

def is_whisper_model(model_path: str, model_name: str) -> bool:
    # crude check, can be customized
    return "whisper" in model_path.lower() or "whisper" in model_name.lower()

def main():
    parser = argparse.ArgumentParser(description="Optional LoRA merge and model upload script.")
    parser.add_argument("--adapter_path", type=str, help="Path to the adapter checkpoint.")
    parser.add_argument("--repo_owner", type=str, default="keeve101", help="Hugging Face repo owner.")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace repo ID.")
    parser.add_argument("--convert_to_pt", action="store_true", help="Whether to convert to PT format.")
    parser.add_argument("--model_name", type=str, required=True, help="Custom model name for push.")
    args = parser.parse_args()

    model_path = args.model
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model + processor
    if is_whisper_model(model_path, model_name):
        model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        processor = WhisperProcessor.from_pretrained(model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
        processor = AutoTokenizer.from_pretrained(model_path)

    model.to(device)

    # if adapter provided, merge with it
    if args.adapter_path:
        print(f"Merging model with adapter from {args.adapter_path}")
        model_to_merge = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16)
        merged_model = model_to_merge.merge_and_unload()
        merged_model_path = Path(args.adapter_path) / "merged_model"
        merged_model.save_pretrained(merged_model_path)
        processor.save_pretrained(merged_model_path)
    else:
        print("No adapter provided. Skipping LoRA merge.")
        merged_model_path = Path(model_path)  # use original model path

    repo_id = f"{args.repo_owner}/{model_name}"
    print(f"Pushing to {repo_id}")

    # push model and processor
    model_to_push = merged_model if args.adapter_path else model
    model_to_push.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)

    # optional: convert to PT
    if args.convert_to_pt:
        command = [
            "python", "convert-to-pt.py",
            "--model_name", repo_id,
            "--output_dir", str(merged_model_path),
            "--output_name", model_name,
        ]
        subprocess.call(command)

        pt_file = merged_model_path / f"{model_name}.pt"
        if pt_file.exists():
            api = HfApi()
            api.upload_file(
                path_or_fileobj=pt_file,
                path_in_repo=pt_file.name,
                repo_id=repo_id,
            )
        else:
            print(f"Warning: {pt_file} not found. Skipping PT file upload.")

if __name__ == "__main__":
    main()

