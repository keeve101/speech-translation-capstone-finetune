from transformers import WhisperForConditionalGeneration
from peft import PeftModel
from pathlib import Path
from huggingface_hub import HfApi
import torch
import subprocess

language_code = "th"
adapter_path = f"./{language_code}_output"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model_to_merge = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16)

merged_model_path = Path(adapter_path) / "merged_model"
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(merged_model_path)

kwargs = {
    "dataset_tags": "keeve101/common-voice-unified-splits",
    "dataset": "Common Voice 10.0/20.0/21.0 (en/zh-CN/th/hi/id/vi) splits",
    "language": language_code,
    "model_name": f"Whisper Large v3 Turbo - LoRA {language_code}-finetuned",
    "finetuned_from": "openai/whisper-large-v3-turbo",
    "tasks": "automatic-speech-recognition",
}

model_name = f"whisper-large-v3-turbo-cv-unified-splits-LoRA-finetuned-{language_code}"

#torch.save(merged_model.state_dict(), merged_model_path / f"{model_name}.pt")

repo_id = f"keeve101/{model_name}"
#merged_model.push_to_hub(repo_id, **kwargs)

command = ["python", "convert-to-pt.py", "--model_name", f"{repo_id}", "--output_dir", merged_model_path, "--output_name", f"{model_name}"]

subprocess.call(command)

api = HfApi()
api.upload_file(
    path_or_fileobj=merged_model_path / f"{model_name}.pt",
    path_in_repo=f"{model_name}.pt",
    repo_id=repo_id,
)