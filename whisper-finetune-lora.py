import torch
import evaluate

from datasets import IterableDatasetDict, load_dataset, Audio 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

LANGUAGES = { 
    # full forms derived from https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    "zh-CN": "chinese",
    "en": "english",
    "hi": "hindi",
    "id": "indonesian",
    "th": "thai",
    "vi": "vietnamese",
}

language_code = "zh-CN"

dataset_path = "keeve101/common-voice-unified-splits"

streaming = False

dataset = IterableDatasetDict()

split_percentage = "10%"

train_split = "train" if split_percentage == "" else f"train[:{split_percentage}]"
dev_split = "dev" if split_percentage == "" else f"dev[:{split_percentage}]"
test_split = "test" if split_percentage == "" else f"test[:{split_percentage}]"

subsample_size = 30
dataset["train"] = load_dataset(dataset_path, language_code, split=train_split, streaming=streaming).shuffle(seed=0).take(subsample_size)
dataset["validation"] = load_dataset(dataset_path, language_code, split=dev_split, streaming=streaming).shuffle(seed=0).take(subsample_size)
dataset["test"] = load_dataset(dataset_path, language_code, split=test_split, streaming=streaming).shuffle(seed=0).take(subsample_size)

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", langauge=LANGUAGES[language_code], task="transcribe", predict_timestamps=False)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # cast all to 16kHz

"""
In our pre-processing strategy, we will do normalization as described in the Whisper paper with the exception of lower-case and punctuation removal. 

However, during evaluation the normalization is fully applied
"""
normalizer = EnglishTextNormalizer() if language_code == "en" else BasicTextNormalizer()

def prepare_dataset(batch, do_lower_case=False, do_remove_punctuation=False):
    audio = batch["audio"]
    
    # compute log-Mel input features with the processor feature extractor
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] 
    
    # computer input length of audio in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    transcription = batch["sentence"]

    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

def compute_metrics(pred, do_normalize_eval=True):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=label_str, tokenize="intl") 
    bleu_score = bleu["score"]

    return {"wer": wer, "cer": cer, "sacrebleu": bleu_score}

# remove all columns, leave just columns input_features and labels
vectorized_dataset = dataset.map(prepare_dataset, remove_columns=list(next(iter(dataset.values())).features)).with_format("torch")

vectorized_dataset["train"] = vectorized_dataset["train"].shuffle(seed=0)

def is_audio_in_length_range(length, max_input_length=30.0):
    return length < max_input_length

# filter out audios > 30.0 seconds as it would be truncated by processor
vectorized_dataset["train"] = vectorized_dataset["train"].filter(is_audio_in_length_range, input_columns=["input_length"],)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

#model.enable_input_require_grads()

model = get_peft_model(model, lora_config)

"""
Model  | Batch Size | Gradient Accumulation | Learning Rate
small	16	2	8
medium	2	16	1

If experience OOM, reduce per_device_train_batch_size by factor of 2 and increase gradient_accumulation_steps
"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps", # do_eval is True if "steps" is passed
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_dataset["train"],
    eval_dataset=vectorized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    # callbacks=[ShuffleCallback()],
)

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=training_args.output_dir)

# Evaluate the base model before fine-tuning
print("Evaluating base model...")
base_results = trainer.evaluate(eval_dataset=vectorized_dataset["test"])

# Save base model results to file
base_results_file = "./base_evaluation_results.txt"
with open(base_results_file, "w") as f:
    f.write("Base model evaluation results:\n")
    for key, value in base_results.items():
        if isinstance(value, dict):
            value = value["score"]
        f.write(f"{key}: {value}\n")

print("Base model evaluation complete. Starting training...")

# Start training
trainer.train()

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
final_results = trainer.evaluate(eval_dataset=vectorized_dataset["test"])

# Save fine-tuned results to file
results_file = "./final_evaluation_results.txt"
with open(results_file, "w") as f:
    f.write("Final evaluation results:\n")
    for key, value in final_results.items():
        if isinstance(value, dict):
            value = value["score"]
        f.write(f"{key}: {value}\n")

# Close the TensorBoard writer
writer.close()