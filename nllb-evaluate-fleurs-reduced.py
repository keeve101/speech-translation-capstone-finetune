import os
import argparse
import transformers
import ctranslate2
import subprocess
import evaluate
import torch

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

def translate_batch(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size):
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

    print(f"Processing translation: {src_lang} -> {tgt_lang}")  # Status message
    
    for batch in tqdm(batched_dataset, total=len(batched_dataset), desc="Translating", unit="batch"):
        src_batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(x)) for x in batch[src_key]]
        
        tgt_prefix = [[tgt_lang]] * len(src_batch)
        
        translated_batch = translator.translate_batch(src_batch, target_prefix=tgt_prefix)
        
        predictions = [tokenizer.decode(tokenizer.convert_tokens_to_ids(x.hypotheses[0][1:])) for x in translated_batch]

        predictions = [normalizer(pred) for pred in predictions]
        references = [normalizer(ref) for ref in batch[tgt_key]]
        
        wer_scores.append(wer_metric.compute(predictions=predictions, references=references))
        cer_scores.append(cer_metric.compute(predictions=predictions, references=references))
        bleu_scores.append(bleu_metric.compute(predictions=predictions, references=[[r] for r in references], tokenize="intl"))
        
    return {
        "WER": sum(wer_scores) / len(wer_scores) * 100,
        "CER": sum(cer_scores) / len(cer_scores) * 100,
        "BLEU": sum([b["score"] for b in bleu_scores]) / len(bleu_scores)
    }

def main():
    parser = argparse.ArgumentParser(description="Convert and use a CTranslate2 model for translation.")
    parser.add_argument("--model_path", type=str, help="Path to the model directory. If omitted, uses the default Facebook model.")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path else "facebook/nllb-200-distilled-600M"
    model_name = os.path.basename(model_path)  # Take as output directory
    
    output_path = "ctranslate2-models/" + model_name

    if not os.path.isdir(output_path):
        command = ["ct2-transformers-converter", "--model", model_path, "--output_dir", output_path]
        
        subprocess.call(command)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = ctranslate2.Translator(output_path, device=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    normalizer = BasicTextNormalizer()
    
    dataset_path = "keeve101/fleurs-reduced"
    config_names = get_dataset_config_names(dataset_path)
    dataset_dicts = {config: load_dataset(dataset_path, config, split="train") for config in config_names}
    
    batch_size = 64
    
    results = {}
    
    for config, dataset in dataset_dicts.items():
        # run en --> other first, then other --> en
        src_key = "en_transcription"
        tgt_key = f"{config}_transcription"
        
        src_lang = LANGUAGES['en']
        tgt_lang = LANGUAGES[config]
        
        results[f"{src_lang} -> {tgt_lang}"] = translate_batch(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size)
        
        src_key, tgt_key = tgt_key, src_key
        src_lang, tgt_lang = tgt_lang, src_lang

        results[f"{src_lang} -> {tgt_lang}"] = translate_batch(dataset, tokenizer, translator, normalizer, src_lang, tgt_lang, src_key, tgt_key, batch_size)
        
    
    for direction, scores in results.items():
        print(f"Results for {direction}:")
        print(f"  WER: {scores['WER']:.4f}")
        print(f"  CER: {scores['CER']:.4f}")
        print(f"  BLEU: {scores['BLEU']:.4f}")

if __name__ == "__main__":
    main()
