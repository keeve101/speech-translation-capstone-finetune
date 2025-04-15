# CrossTalk Secure Finetuning Code Repository
Code repository containing code for finetuning and evaluating models for the CrossTalk Secure project.

```
├── preprocessing                                       | Preprocessing notebooks for data
│   ├── fleurs-subsets-upload.ipynb 
│   ├── parse_magichub_datasets.ipynb 
│   └── parse_translation_datasets.ipynb 
├── finetuning 
│   ├── nllb-finetune-lora-balanced-multi-corpora.py
│   │   └── Finetune NLLB using LoRA on balanced multilingual corpora.
│   ├── whisper-finetune-lora-cross-val.py
│   │   └── Finetune Whisper with LoRA using cross-validation monolingual subsets.
│   ├── whisper-finetune-lora-unified.py 
│   │   └── Finetune Whisper with LoRA on a unified multilingual dataset.
│   └── whisper-finetune-full-unified.py
│       └── Full finetune of Whisper on unified multilingual dataset.
├── evaluation
│   ├── nllb-evaluate-example.py
│   │   └── Example script showing how to evaluate using NLLB models on sample data.
│   ├── nllb-evaluate-fleurs-reduced-on-whisper-preds.py
│   │   └── Evaluates NLLB translations against Whisper-generated predictions on a reduced FLEURS dataset.
│   ├── nllb-evaluate-fleurs-reduced.py
│   │   └── Evaluates NLLB model outputs directly on the reduced FLEURS dataset.
│   ├── nllb-evaluate-with-comet.py
│   │   └── Uses COMET scoring to evaluate NLLB translation quality.
│   ├── whisper-evaluate-ctranslate2.py
│   │   └── Evaluates Whisper outputs generated with CTranslate2 backend.
│   ├── whisper-evaluate-fleurs-reduced-ctranslate2.py
│   │   └── Evaluates Whisper (CTranslate2) predictions on the reduced FLEURS dataset.
│   └── whisper-evaluate-fleurs-reduced.py
│       └── Evaluates Whisper model outputs on the reduced FLEURS dataset (transformers backend).
├── .gitignore
├── README.md
├── jfk.flac
├── whisper_lib.py
├── merge-and-upload.py
├── convert-to-pt.py
└── requirements.txt
```

## Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/crosstalk-secure-finetuning.git
   cd crosstalk-secure-finetuning
   ```

2. **Create and activate a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Utilities

- `convert-to-pt.py`  
  Converts raw or intermediate model outputs to `.pt` format for compatibility with evaluation scripts.

- `merge-and-upload.py`  
  Merges prediction files and uploads them to cloud or experiment tracking services.

- `whisper_lib.py`  
  Core utility library used by Whisper fine-tuning and evaluation scripts (e.g., data loading, preprocessing, decoding).

- `jfk.flac`  
  Sample audio file (John F. Kennedy speech) used for testing inference pipelines.

## Finetuning

See the [`finetuning`](./finetuning) directory for training scripts:
- Scripts use either **LoRA** for lightweight updates or full fine-tuning.
- You can modify corpus paths, model configs, or training settings directly in each script.

## Evaluation

Evaluation scripts are in the [`evaluation`](./evaluation) folder and include:
- BLEU and COMET-based scoring
- Support for Whisper and NLLB models
- CTranslate2 and Transformers inference backends

## Preprocessing

The [`preprocessing`](./preprocessing) notebooks:
- Upload and slice subsets of the FLEURS dataset
- Parse open-source datasets (e.g., MagicHub)
- Normalize and prepare text-to-text and speech-to-text corpora

## Example Usage

### Fine-tune Whisper using LoRA:
```bash
python finetuning/whisper-finetune-lora-unified.py \
  --data_dir data/processed \
  --output_dir checkpoints/whisper-lora-unified
```

### Evaluate NLLB on FLEURS subset:
```bash
python evaluation/nllb-evaluate-fleurs-reduced.py \
  --preds_file outputs/nllb_preds.json \
  --refs_file data/fleurs/ground_truth.json
```

## Notes
