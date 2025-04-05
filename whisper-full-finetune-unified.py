import os
os.environ["HF_HOME"] = "/workspace/.cache"

import torch
import evaluate

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset, Audio
from whisper_lib import LANGUAGES, DataCollatorSpeechSeq2SeqWithPadding, take_dataset, prepare_dataset
from torch.utils.data import WeightedRandomSampler
from typing import Optional

language_codes = [key for key in LANGUAGES.keys()]

cv_unified_dataset_path = "keeve101/common-voice-unified-splits"
magic_hub_ms_tl_dataset_path = "keeve101/magic-hub-ms-tl-datasets"

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", task="transcribe", predict_timestamps=False)

combine_train_val = True # whether to combine train and validation into one dataset

cv_unified_dataset_config_names = get_dataset_config_names(cv_unified_dataset_path)
magic_hub_ms_tl_dataset_config_names = get_dataset_config_names(magic_hub_ms_tl_dataset_path)

dataset_dicts = {
    config: take_dataset(cv_unified_dataset_path, config, split_percentage="", streaming=False, subsample_size=500, combine_train_val=combine_train_val) for config in cv_unified_dataset_config_names
}

dataset_dicts.update({
    config: load_dataset(magic_hub_ms_tl_dataset_path, config).cast_column("audio", Audio(sampling_rate=16000)) for config in magic_hub_ms_tl_dataset_config_names
})

assert set(cv_unified_dataset_config_names) | set(magic_hub_ms_tl_dataset_config_names) == set(language_codes) # sanity check

dataset_train_subsets = []

total_train = 0

for key, dataset in dataset_dicts.items():
    fn_kwargs = {"processor": processor, "language_code": key}

    dataset_dicts[key] = dataset.map(prepare_dataset, remove_columns=list(next(iter(dataset.values())).features), fn_kwargs=fn_kwargs).with_format("torch")
    
    dataset_dicts[key]["train"] = dataset_dicts[key]["train"].shuffle(seed=0) # for consistency of evaluation

    def is_audio_in_length_range(length, max_input_length=30.0):
        return length < max_input_length

    # filter out audios > 30.0 seconds as it would be truncated by processor
    dataset_dicts[key]["train"] = dataset_dicts[key]["train"].filter(is_audio_in_length_range, input_columns=["input_length"],)
    
    dataset_train_subsets.append(dataset_dicts[key]["train"])
    
    total_train += len(dataset_dicts[key]["train"])

# Concatenate subsets
dataset_train = concatenate_datasets(dataset_train_subsets)

class_weights = [len(dataset_train_subset) / total_train for dataset_train_subset in dataset_train_subsets]

sample_weights = []
for idx, subset in enumerate(dataset_train_subsets):
    sample_weights.extend([class_weights[idx]] * len(subset))
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

"""
Model  | Batch Size | Gradient Accumulation | Learning Rate
small	16	2	8
medium	2	16	1

If experience OOM, reduce per_device_train_batch_size by factor of 2 and increase gradient_accumulation_steps
"""

def compute_metrics(pred, do_normalize_eval=True):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False, basic_normalize=do_normalize_eval)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, decode_with_timestamps=False, basic_normalize=do_normalize_eval)

    if do_normalize_eval:
        #pred_str = [normalizer(pred) for pred in pred_str]
        #label_str = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=label_str, tokenize="intl") 
    bleu_score = bleu["score"]

    return {"wer": wer, "cer": cer, "sacrebleu": bleu_score}

eval_dataset = {key: dataset_dicts[key]["test"] for key in language_codes}

output_dir = "./output-unified-weighted-random-sampler"

class Seq2SeqTrainerWithWeightedRandomSampler(Seq2SeqTrainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.sample_weights, num_samples=len(self.train_dataset), replacement=True
            )
        else:
            return super()._get_train_sampler()

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # Set per-language checkpoint directory
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=9000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    #load_best_model_at_end=True,
    #metric_for_best_model="sacrebleu",
    #greater_is_better=True,
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainerWithWeightedRandomSampler(
    args=training_args,
    model=model,
    train_dataset=dataset_train,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    sample_weights=sample_weights,
)

checkpoint_path = get_last_checkpoint(output_dir) 

# **Train with checkpoint resumption**
trainer.train(resume_from_checkpoint=checkpoint_path)

# Save after training is done
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
