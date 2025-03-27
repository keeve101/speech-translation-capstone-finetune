import torch
import evaluate

from datasets import IterableDatasetDict, load_dataset, Audio 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

from dataclasses import dataclass
from typing import Any, Dict, List, Union

LANGUAGES = { 
    # full forms derived from https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    "zh-CN": "chinese",
    "en": "english",
    "hi": "hindi",
    "id": "indonesian",
    "th": "thai",
    "vi": "vietnamese",
}
language_code = "th"


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", langauge=LANGUAGES[language_code], task="transcribe", predict_timestamps=False)

"""
In our pre-processing strategy, we will do normalization as described in the Whisper paper with the exception of lower-case and punctuation removal. 

However, during evaluation the normalization is fully applied
"""
normalizer = EnglishTextNormalizer() if language_code == "en" else BasicTextNormalizer()

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

def prepare_dataset(batch, do_lower_case=False, do_remove_punctuation=False):
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

fleurs_reduced_dataset = load_dataset(fleurs_reduced_dataset_path, language_code)

dataset = fleurs_reduced_dataset.cast_column(f"{language_code}_audio", Audio(sampling_rate=16000)) # cast all to 16kHz

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

vectorized_dataset = dataset.map(prepare_dataset, remove_columns=list(next(iter(dataset.values())).features)).with_format("torch")

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps", # do_eval is True if "steps" is passed
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
    push_to_hub=False,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)
model_pre = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

trainer_pre = Seq2SeqTrainer(
    args=training_args,
    model=model_pre,
    train_dataset=vectorized_dataset,
    eval_dataset=vectorized_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    # callbacks=[ShuffleCallback()],
)
print(trainer_pre.evaluate(vectorized_dataset))

config = PeftConfig.from_pretrained(f"{language_code}_output/checkpoint-1000")
model = PeftModel.from_pretrained(model, f"{language_code}_output/checkpoint-1000", is_trainable=False)

trainer_post = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_dataset,
    eval_dataset=vectorized_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    # callbacks=[ShuffleCallback()],
)

print(trainer_post.evaluate(vectorized_dataset, metric_key_prefix="post"))
