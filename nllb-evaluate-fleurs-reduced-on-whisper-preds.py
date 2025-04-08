import torch
import evaluate
import json
import zhconv
import argparse
import copy
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from pprint import pprint

LANGUAGES = {
   'hi': "hin_Deva", 
   'id': "ind_Latn", 
   'ms': "zsm_Latn",
   'th': "tha_Thai",
   'tl': "tgl_Latn",
   'vi': "vie_Latn",
   'zh-CN': "zho_Hans",
   'en': "eng_Latn"
}

LANGUAGES_INVERSE = {k: v for v, k in LANGUAGES.items()}

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Transcribe text using NLLB model.")
parser.add_argument(
    '--model_name', type=str, required=True,
    help="The Hugging Face model name or path for the NLLB model"
)
parser.add_argument(
    '--whisper_model_used', type=str, required=True,
    help="The Whisper model name used to generate the predictions"
)
args = parser.parse_args()

model_name = args.model_name
base_model_name = model_name.split("/")[-1]

whisper_model_used = args.whisper_model_used.split("/")[-1]

output_file_path = base_model_name + "-on-" + whisper_model_used + "-eval.json"

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")

# Use NLLB tokenizer and model instead of Whisper
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
normalizer = BasicTextNormalizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def normalize_zh(text):
    text = insert_spaces_between_characters(text)
    text = zhconv.convert(text, "zh-cn")  # convert to simplified chinese
    return text

def insert_spaces_between_characters(text):
    space_removed = "".join([t.strip() for t in text.split()])
    return " ".join(space_removed)

def prepare_dataset(batch, language_code, do_lower_case=False, do_remove_punctuation=False, max_length=1024):
    en_transcription = batch["en_transcription"]
    other_transcription = batch[f"{language_code}_transcription"]

    if do_lower_case:
        en_transcription = en_transcription.lower()
        other_transcription = other_transcription.lower()
    if do_remove_punctuation:
        en_transcription = normalizer(en_transcription).strip()
        other_transcription = normalizer(other_transcription).strip()

    tokenizer.src_lang = LANGUAGES['en']
    batch["en_input_ids"] = tokenizer(en_transcription, padding=True, truncation=True, max_length=max_length, return_tensors="pt").input_ids
    
    tokenizer.src_lang = LANGUAGES[language_code]
    batch[f"{language_code}_input_ids"] = tokenizer(other_transcription, padding=True, truncation=True, max_length=max_length, return_tensors="pt").input_ids

    return batch

fleurs_reduced_dataset_path = "keeve101/fleurs-reduced" + "-with-whisper-predictions"

# Loading datasets
configs = get_dataset_config_names(fleurs_reduced_dataset_path)
datasets_dict = {language_code: load_dataset(fleurs_reduced_dataset_path, language_code) for language_code in configs}

datasets_dict_copy = copy.deepcopy(datasets_dict)

# Delete {other_language}_transcription column, replace it with {whisper_model_used}-prediction 
for language_code in datasets_dict.keys():
    def replace_transcription(example):
        example[f"{language_code}_transcription"] = example[f"{whisper_model_used}-prediction"]
        return example

    datasets_dict[language_code]["train"] = datasets_dict[language_code]["train"].map(replace_transcription)

vectorized_datasets_dict = {key: dataset.map(prepare_dataset, fn_kwargs={"language_code": key, "do_lower_case": False, "do_remove_punctuation": False}).with_format("torch") for key, dataset in datasets_dict.items()}

def normalize(batch, normalizer, lang_code):
    batch = [normalizer(text) for text in batch]
    
    if lang_code == 'zh-CN':
        batch = [normalize_zh(text) for text in batch]
    elif lang_code == "th":
        batch = [insert_spaces_between_characters(text) for text in batch]
    
    return batch

def decode_preds_and_labels(pred_ids, label_ids):
    tokenizer.tgt_lang = LANGUAGES[lang_code]
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, basic_normalize=False)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, basic_normalize=False)

        
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
    
    all_preds = {}
    all_labels = {}
    for inputs in tqdm(dataset["train"], desc=f"{lang_code}", unit="batch", total=len(dataset["train"])):
        for key, value in inputs.items():
            if "input_ids" in key:
                inputs[key] = value.to(device)
        
        src_lang = lang_code
        tgt_lang = "en"

        pred_ids = model.generate(
            input_ids=inputs[f"{src_lang}_input_ids"],
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANGUAGES[tgt_lang]),
            max_new_tokens=int(16 + 1.5 * inputs[f"{tgt_lang}_input_ids"].shape[1]),
        )
        
        label_ids = inputs[f"{tgt_lang}_input_ids"]

        preds, labels = decode_preds_and_labels(pred_ids, label_ids)

        if isinstance(inputs["id"], list):
            for idx in range(len(inputs["id"])):
                saved_preds[lang_code][int(inputs["id"][idx])] = preds[idx]
        else:
            saved_preds[lang_code][int(inputs["id"])] = preds

        normalized_preds, normalized_labels = map(lambda x: normalize(x, normalizer, lang_code), (preds, labels))

        all_preds.setdefault((src_lang, tgt_lang), [])
        all_preds[(src_lang, tgt_lang)].extend(normalized_preds)

        all_labels.setdefault((src_lang, tgt_lang), [])
        all_labels[(src_lang, tgt_lang)].extend(normalized_labels)
   
    for direction, preds in all_preds.items():
        labels = all_labels[direction]
        metrics = compute_metrics(preds, labels)
        results[direction] = metrics
        print(direction)
        pprint(metrics)

with open(output_file_path, "w") as f:
    results = {f"{k[0]}-{k[1]}": v for k, v in results.items()}
    json.dump(results, f, indent=4)

for lang_code in datasets_dict_copy.keys():
    preds = saved_preds[lang_code]

    def add_prediction(example):
        pred = preds.get(int(example["id"]), "")
        if isinstance(pred, list):
            pred = pred[0]

        example[f"{base_model_name}-on-{whisper_model_used}-prediction"] = pred
        return example

    datasets_dict_copy[lang_code]["train"] = datasets_dict_copy[lang_code]["train"].map(add_prediction)

    datasets_dict_copy[lang_code].push_to_hub(fleurs_reduced_dataset_path, lang_code)
