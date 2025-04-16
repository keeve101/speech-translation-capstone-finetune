import os
import torch
import evaluate
import json
import zhconv
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, get_dataset_config_names
from transformers import AutoProcessor, Wav2Vec2ForCTC
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from pprint import pprint

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Transcribe audio using Whisper model.")
parser.add_argument(
    '--model_name', type=str, required=True,
    help="The Hugging Face model name or path for the Whisper model"
)
args = parser.parse_args()

model_name = args.model_name
base_model_name = model_name.split("/")[-1]

output_dir = base_model_name + "-eval"
os.makedirs(output_dir, exist_ok=True)
output_file_path = base_model_name + "-eval.json"

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")

processor = AutoProcessor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

normalizer = BasicTextNormalizer()

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def normalize_zh(text):
    text = insert_spaces_between_characters(text)
    text = zhconv.convert(text, "zh-cn") # convert to simplified chinese
    return text

def insert_spaces_between_characters(text):
    space_removed = "".join([t.strip() for t in text.split()])
    return " ".join(space_removed)

def prepare_dataset(batch, language_code, do_lower_case=True, do_remove_punctuation=True):
    audio = batch[f"{language_code}_audio"]
    
    # compute log-Mel input features with the processor feature extractor
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    
    transcription = batch[f"{language_code}_transcription"]

    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
        
    batch[f"{language_code}_transcription"] = transcription
    return batch

fleurs_reduced_dataset_path = "keeve101/fleurs-reduced"

configs = get_dataset_config_names(fleurs_reduced_dataset_path)

datasets_dict = {language_code: load_dataset(fleurs_reduced_dataset_path, language_code).cast_column(f"{language_code}_audio", Audio(sampling_rate=16000)) for language_code in configs}

vectorized_datasets_dict = {key: dataset.map(prepare_dataset, fn_kwargs={"language_code": key, "do_lower_case": True, "do_remove_punctuation": True}).with_format("torch") for key, dataset in datasets_dict.items()}

def normalize(batch, normalizer, lang_code):
    batch = [normalizer(text) for text in batch]
    
    if lang_code == "zh-CN":
        batch = [normalize_zh(text) for text in batch]
    elif lang_code == "th":
        batch = [insert_spaces_between_characters(text) for text in batch]
    
    return batch

def compute_metrics(preds, labels):
    wer = 100 * wer_metric.compute(predictions=preds, references=labels)
    cer = 100 * cer_metric.compute(predictions=preds, references=labels)
    bleu = bleu_metric.compute(predictions=preds, references=labels, tokenize="intl") 
    bleu_score = bleu["score"]

    return {"wer": wer, "cer": cer, "sacrebleu": bleu_score}

saved_preds = {lang_code: {} for lang_code in vectorized_datasets_dict.keys()}
results = {}

for lang_code, dataset in vectorized_datasets_dict.items():
    print(f"\nTranscribing for {lang_code}")
    
    data = dataset["train"]
    all_preds = []  
    all_labels = []
    BATCH_SIZE = 8
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=lang_code, unit="batch"):
        batch = data.select(range(i, min(i + BATCH_SIZE, len(data))))
        input_features = torch.stack([batch[j]["input_features"] for j in range(len(batch))]).to(device)

        with torch.no_grad():
            logits = model(**input_features).logits
            pred_ids = torch.argmax(logits, dim=-1)

        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels = [example[f"{lang_code}_transcription"] for example in batch]
        
        for idx, example in enumerate(batch):
            saved_preds[lang_code][int(example["id"])] = preds[idx]

        normalized_preds, normalized_labels = map(lambda x: normalize(x, normalizer, lang_code), (preds, labels))
        all_preds.extend(normalized_preds)
        all_labels.extend(normalized_labels)
    
    metrics = compute_metrics(all_preds, all_labels)
    results[lang_code] = metrics
    pprint(metrics)
    
with open(output_file_path, "w") as f:
    json.dump(results, f, indent=4)

for lang_code in datasets_dict.keys():
    preds = saved_preds[lang_code]

    def add_prediction(example):
        pred = preds.get(int(example["id"]), "")
        example[base_model_name + "-prediction"] = pred
        return example

    datasets_dict[lang_code]["train"] = datasets_dict[lang_code]["train"].map(add_prediction)

    datasets_dict[lang_code].push_to_hub(fleurs_reduced_dataset_path + "baseline-model-evaluations", lang_code)
