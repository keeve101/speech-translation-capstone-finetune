import argparse
from transformers import WhisperForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
from huggingface_hub import HfApi
import torch
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and upload model with LoRA adaptation.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter checkpoint.")
    parser.add_argument("--repo_owner", type=str, help="Hugging Face repo owner.", default="keeve101")
    parser.add_argument("--model", type=str, choices=["openai/whisper-large-v3-turbo", "facebook/nllb-200-distilled-600M"], default="openai/whisper-large-v3-turbo", help="Model to use for fine-tuning.")
    parser.add_argument("--convert_to_pt", action="store_true", help="Whether to convert to PT format.")
    parser.add_argument("--model_name", type=str, help="Custom model name.", required=True)
    args = parser.parse_args()

    if args.model == "openai/whisper-large-v3-turbo":
        model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # merge with adapter, then upload to hf
    model_to_merge = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16)
    merged_model_path = Path(args.adapter_path) / "merged_model"
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)
    
    model_name = args.model_name

    repo_id = f"{args.repo_owner}/{model_name}"

    merged_model.push_to_hub(repo_id)

    if args.convert_to_pt:
        command = [
            "python", "convert-to-pt.py", "--model_name", repo_id,
            "--output_dir", str(merged_model_path), "--output_name", model_name
        ]
        subprocess.call(command)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=merged_model_path / f"{model_name}.pt",
            path_in_repo=f"{model_name}.pt",
            repo_id=repo_id,
        )

if __name__ == "__main__":
    main()
