import os

import evaluate
import json
from zhconv import zhconv

from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, get_dataset_config_names
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from pprint import pprint
from whisper_lib import DataCollatorSpeechSeq2SeqWithPadding

model_name = "keeve101/whisper-large-v3-turbo-full-finetune-unified-checkpoint-2400"
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

def normalize_zh(text):
    text = insert_spaces_between_characters(text)
    text = zhconv(text, "zh-cn") # convert to simplified chinese
    return text

def insert_spaces_between_characters(text):
    return " ".join(text.split())

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

fleurs_reduced_dataset_path = "keeve101/fleurs-reduced"

configs = get_dataset_config_names(fleurs_reduced_dataset_path)

datasets_dict = {language_code: load_dataset(fleurs_reduced_dataset_path, language_code).cast_column(f"{language_code}_audio", Audio(sampling_rate=16000)) for language_code in configs}

vectorized_datasets_dict = {key: dataset.map(prepare_dataset, remove_columns=list(next(iter(dataset.values())).features), fn_kwargs={"language_code": key, "do_lower_case": True, "do_remove_punctuation": True}).with_format("torch") for key, dataset in datasets_dict.items()}

def compute_metrics(pred_ids, label_ids, lang_code, do_normalize_eval=True):
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False, basic_normalize=do_normalize_eval)
    labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, decode_with_timestamps=False, basic_normalize=do_normalize_eval)

    if do_normalize_eval:
        preds = [normalizer(pred) for pred in preds]
        labels = [normalizer(label) for label in labels]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        preds = [preds[i] for i in range(len(preds)) if len(labels[i]) > 0]
        labels = [labels[i] for i in range(len(labels)) if len(labels[i]) > 0]
    
    if lang_code == 'zh-CN':
        preds = [normalize_zh(pred) for pred in preds]
        labels = [normalize_zh(label) for label in labels]
    elif lang_code == "th":
        preds = [insert_spaces_between_characters(pred) for pred in preds]
        labels = [insert_spaces_between_characters(label) for label in labels]

    wer = 100 * wer_metric.compute(predictions=preds, references=labels)
    cer = 100 * cer_metric.compute(predictions=preds, references=labels)
    bleu = bleu_metric.compute(predictions=preds, references=labels, tokenize="intl") 
    bleu_score = bleu["score"]

    return {"wer": wer, "cer": cer, "sacrebleu": bleu_score}

results = {}

for lang_code, dataset in datasets_dict.items():
    print(f"\nTranscribing for {lang_code}")
    
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    all_preds = []  
    all_labels = []
    for inputs in dataloader:
        preds = model.generate(**inputs)
        all_preds.extend(preds)
        all_labels.extend(inputs["labels"])
    
    metrics = compute_metrics(all_preds, all_labels, lang_code=lang_code)
    results[lang_code] = metrics
    
with open(output_file_path, "w") as f:
    json.dump(results, f, indent=4)