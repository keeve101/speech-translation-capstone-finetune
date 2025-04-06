import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import subprocess
import evaluate
import json
from faster_whisper import WhisperModel
from datasets import load_dataset, Audio, get_dataset_config_names
from zhconv import zhconv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm

def convert_to_ct2(model_path, output_dir):
    print(f"Converting model '{model_path}' to CTranslate2 format...")
    command = [
        "ct2-transformers-converter",
        "--model", model_path,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer_config.json", "preprocessor_config.json", "special_tokens_map.json", "added_tokens.json",
        "--quantization", "float16"
    ]
    subprocess.run(command, check=True)
    print("Conversion complete.\n")

def normalize(text, normalizer, language_code):
    text = text.lower()
    text = normalizer(text)
    
    if language_code == "zh-CN":
        text = zhconv.convert(text, 'zh-cn')

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Convert and transcribe with Whisper via CTranslate2 backend.")
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3-turbo", help="Path to Whisper model")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to run model on")
    parser.add_argument("--compute_type", type=str, choices=["float16", "int8", "int8_float16"], default="float16", help="Precision mode for inference")
    args = parser.parse_args()

    model_path = args.model_path
    model_name = os.path.basename(model_path)
    output_dir = f"ctranslate2-models/{model_name}"

    if not os.path.isdir(output_dir):
        convert_to_ct2(model_path, output_dir)

    print(f"Loading model from {output_dir} on {args.device} ({args.compute_type})...")
    model = WhisperModel(output_dir, device=args.device, compute_type=args.compute_type)
    
    print("\nLoading dataset")
    fleurs_reduced_dataset_path = "keeve101/fleurs-reduced"

    configs = get_dataset_config_names(fleurs_reduced_dataset_path)

    datasets_dict = {language_code: load_dataset(fleurs_reduced_dataset_path, language_code, split="train").cast_column(f"{language_code}_audio", Audio(sampling_rate=16000)) for language_code in configs}
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    bleu_metric = evaluate.load("sacrebleu")
    normalizer = BasicTextNormalizer()
    
    results = {}

    for language_code, dataset in datasets_dict.items():
        print(f"\nTranscribing for {language_code}")
        
        predictions = []
        references = []
        
        for sample in tqdm(dataset, desc=f"{language_code}"):
            audio_array = sample[f"{language_code}_audio"]["array"]
            
            segments, _ = model.transcribe(audio_array, without_timestamps=True)

            prediction = " ".join([segment.text for segment in segments])
            reference = sample[f"{language_code}_transcription"]
            
            predictions.append(normalize(prediction, normalizer, language_code))
            references.append(normalize(reference, normalizer, language_code))
        
        wer = 100 * wer_metric.compute(predictions=predictions, references=references)
        cer = 100 * cer_metric.compute(predictions=predictions, references=references)
        bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
        
        results[language_code] = {
            "wer": wer,
            "cer": cer,
            "sacrebleu": bleu
        }
    
    file_path = os.path.join(os.getcwd(), f"{model_name}-fleurs-reduced-eval-results.json")
    
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults exported to {file_path}")

if __name__ == "__main__":
    main()