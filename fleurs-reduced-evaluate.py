import os
import torch
import evaluate
import json
import zhconv
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, get_dataset_config_names
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from pprint import pprint
from whisper_lib import DataCollatorSpeechSeq2SeqWithPadding

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

processor = WhisperProcessor.from_pretrained(model_name, task="transcribe", predict_timestamps=False)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
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
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] 
    
    # computer input length of audio in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    transcription = batch[f"{language_code}_transcription"]

    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

fleurs_reduced_dataset_path = "keeve101/fleurs-reduced" + "-with-whisper-predictions"

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

def decode_preds_and_labels(pred_ids, label_ids):
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
    labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, decode_with_timestamps=False)

    # filtering step to only evaluate the samples that correspond to non-zero references:
    preds = [preds[i] for i in range(len(preds)) if len(labels[i]) > 0]
    labels = [labels[i] for i in range(len(labels)) if len(labels[i]) > 0]
    
    return preds, labels

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
    
    dataloader = DataLoader(dataset["train"], batch_size=8, collate_fn=data_collator)
    all_preds = []  
    all_labels = []
    for inputs in tqdm(dataloader, desc=f"{lang_code}", unit="batch", total=len(dataloader)):
        pred_ids = model.generate(**{k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)})
        preds, labels = decode_preds_and_labels(pred_ids, inputs["labels"])
        
        if isinstance(inputs["id"], list):
            for idx in range(len(inputs["id"])):
                saved_preds[lang_code][inputs["id"][idx]] = preds[idx]
        else:
            saved_preds[lang_code][inputs["id"]] = preds

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
        pred = preds.get(example["id"], "")
        example[base_model_name + "-prediction"] = pred
        return example

    datasets_dict[lang_code]["train"] = datasets_dict[lang_code]["train"].map(add_prediction)

    datasets_dict[lang_code].push_to_hub(fleurs_reduced_dataset_path, lang_code)