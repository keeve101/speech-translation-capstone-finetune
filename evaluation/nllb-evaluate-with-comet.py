import argparse
import json
from datasets import load_dataset, get_dataset_config_names
from comet import download_model, load_from_checkpoint

parser = argparse.ArgumentParser(description="Transcribe text using NLLB model.")
parser.add_argument(
    '--whisper_model_used', type=str, required=True,
    help="The Whisper model name used to generate the predictions"
)
parser.add_argument(
    '--nllb_model_used', type=str, required=True,
    help="The NLLB model used to generate the predictions"
)

args = parser.parse_args()

whisper_model_used = args.whisper_model_used.split("/")[-1]
nllb_model_used = args.nllb_model_used.split("/")[-1]

fleurs_reduced_dataset_path = "keeve101/fleurs-reduced" + "-with-whisper-predictions"

# Loading datasets
configs = get_dataset_config_names(fleurs_reduced_dataset_path)

datasets_dict = {language_code: load_dataset(fleurs_reduced_dataset_path, language_code) for language_code in configs}

file_path = f"{nllb_model_used}-on-{whisper_model_used}-eval.json"

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

with open(file_path, "r") as f:
    results = json.load(f)

for lang_code in datasets_dict.keys():
    dataset = datasets_dict[lang_code]["train"]
    
    key = f"{nllb_model_used}-on-{whisper_model_used}-prediction"
    
    srcs = dataset[f"{lang_code}_transcription"]
    preds = dataset[key]
    refs = dataset["en_transcription"]
    
    data = [{
        "src": src,
        "mt": pred,
        "ref": ref
    } for src, pred, ref in zip(srcs, preds, refs)]
    
    scores = comet_model.predict(data, batch_size=8, gpus=1)

    results[f"{lang_code}-en"].update({"comet": scores.system_score})
    
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)