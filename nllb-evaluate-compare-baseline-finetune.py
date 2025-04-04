import os
import transformers
import ctranslate2
import subprocess
import evaluate
import torch
import matplotlib.pyplot as plt
import numpy as np
import zhconv

from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, get_dataset_config_names

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

def convert_to_simplified_chinese(text):
    return zhconv.convert(text, 'zh-cn')

def translate_dataset(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size):
    tokenizer.src_lang = src_lang
    
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    batched_dataset = dataset.map(group_batch, batched=True, batch_size=batch_size)
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    bleu_metric = evaluate.load("sacrebleu")
    
    wer_scores = []
    cer_scores = []
    bleu_scores = []

    print(f"Processing translation: {src_lang} -> {tgt_lang}")
    
    for batch in tqdm(batched_dataset, total=len(batched_dataset), desc="Translating", unit="batch"):
        src_batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(x)) for x in batch[src_key]]
        tgt_prefix = [[tgt_lang]] * len(src_batch)
        translated_batch = translator.translate_batch(src_batch, target_prefix=tgt_prefix)
        predictions = [tokenizer.decode(tokenizer.convert_tokens_to_ids(x.hypotheses[0][1:])) for x in translated_batch]
        
        predictions = [normalizer(pred) for pred in predictions]
        references = [normalizer(ref) for ref in batch[tgt_key]]
        
        if src_lang == LANGUAGES["zh-CN"] or tgt_lang == LANGUAGES["zh-CN"]:
            predictions = [convert_to_simplified_chinese(pred) for pred in predictions]
            references = [convert_to_simplified_chinese(ref) for ref in references]
        
        wer_scores.append(wer_metric.compute(predictions=predictions, references=references))
        cer_scores.append(cer_metric.compute(predictions=predictions, references=references))
        bleu_scores.append(bleu_metric.compute(predictions=predictions, references=[[r] for r in references], tokenize="intl"))
        
    return {
        "WER": sum(wer_scores) / len(wer_scores) * 100,
        "CER": sum(cer_scores) / len(cer_scores) * 100,
        "BLEU": sum([b["score"] for b in bleu_scores]) / len(bleu_scores)
    }

def evaluate_on_fleurs(translator, tokenizer, normalizer, batch_size=64):
    dataset_path = "keeve101/fleurs-reduced"
    config_names = get_dataset_config_names(dataset_path)
    dataset_dicts = {config: load_dataset(dataset_path, config, split="train") for config in config_names}
    
    results = {}
    
    for config, dataset in dataset_dicts.items():
        src_key = "en_transcription"
        tgt_key = f"{config}_transcription"
        src_lang = LANGUAGES['en']
        tgt_lang = LANGUAGES[config]
        
        results[f"{src_lang} -> {tgt_lang}"] = translate_dataset(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size)
        
        src_key, tgt_key = tgt_key, src_key
        src_lang, tgt_lang = tgt_lang, src_lang
        
        results[f"{LANGUAGES_INVERSE[src_lang]} -> {LANGUAGES_INVERSE[tgt_lang]}"] = translate_dataset(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size)
    
    return results


device = "cuda" if torch.cuda.is_available() else "cpu"

baseline_model_path = "facebook/nllb-200-distilled-600M"
other_model_path = "keeve101/nllb-200-distilled-600M-finetune-lora-balanced-multi-corpora-checkpoint-100925"

baseline_model_name = os.path.basename(baseline_model_path)
other_model_name = os.path.basename(other_model_path)

baseline_model_local_path = "ctranslate2-models/" + baseline_model_name
other_model_local_path = "ctranslate2-models/" + other_model_name

if not os.path.isdir(baseline_model_local_path):
    command = ["ct2-transformers-converter", "--model", baseline_model_path, "--output_dir", baseline_model_local_path]
    subprocess.call(command)

if not os.path.isdir(other_model_local_path):
    command = ["ct2-transformers-converter", "--model", other_model_path, "--output_dir", other_model_local_path]
    subprocess.call(command)

baseline_translator = ctranslate2.Translator(baseline_model_local_path, device=device)
other_translator = ctranslate2.Translator(other_model_local_path, device=device)

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
normalizer = BasicTextNormalizer()

baseline_results = evaluate_on_fleurs(baseline_translator, tokenizer, normalizer)
other_results = evaluate_on_fleurs(other_translator, tokenizer, normalizer)

for direction, scores in baseline_results.items():
    print(f"Results for {direction}:")
    print(f"  WER: {scores['WER']:.4f}")
    print(f"  CER: {scores['CER']:.4f}")
    print(f"  BLEU: {scores['BLEU']:.4f}")

for direction, scores in other_results.items():
    print(f"Results for {direction}:")
    print(f"  WER: {scores['WER']:.4f}")
    print(f"  CER: {scores['CER']:.4f}")
    print(f"  BLEU: {scores['BLEU']:.4f}")
    
directions = list(baseline_results.keys())
baseline_scores = baseline_results.values()
other_scores = other_results.values()

metrics = ["WER", "CER", "BLEU"]

# Extract metric values per direction
directions = list(baseline_results.keys())
baseline_wer = [baseline_results[dir]["WER"] for dir in directions]
other_wer = [other_results[dir]["WER"] for dir in directions]

baseline_cer = [baseline_results[dir]["CER"] for dir in directions]
other_cer = [other_results[dir]["CER"] for dir in directions]

baseline_bleu = [baseline_results[dir]["BLEU"] for dir in directions]
other_bleu = [other_results[dir]["BLEU"] for dir in directions]

# Function to plot horizontal bar graphs
def plot_metric_comparison(metric_name, baseline_vals, other_vals):
    y = np.arange(len(directions))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y - height / 2, baseline_vals, height, label='Baseline', color='skyblue')
    ax.barh(y + height / 2, other_vals, height, label='Fine-tuned', color='salmon')

    ax.set_xlabel(metric_name)
    ax.set_yticks(y)
    ax.set_yticklabels(directions)
    ax.invert_yaxis()  # highest direction at top
    ax.legend()
    ax.set_title(f"{metric_name} Comparison")
    plt.tight_layout()
    plt.savefig(f"comparison-{metric_name.lower()}.png")

plot_metric_comparison("WER (%)", baseline_wer, other_wer)
plot_metric_comparison("CER (%)", baseline_cer, other_cer)
plot_metric_comparison("BLEU", baseline_bleu, other_bleu)