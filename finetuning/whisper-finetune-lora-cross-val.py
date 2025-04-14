import torch
import evaluate

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from whisper_lib import LANGUAGES, DataCollatorSpeechSeq2SeqWithPadding, take_dataset, prepare_dataset

language_codes = [key for key in LANGUAGES.keys()]

dataset_path = "keeve101/common-voice-unified-splits"

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", task="transcribe", predict_timestamps=False)

dataset_dicts = {key: take_dataset(dataset_path, key, split_percentage="", streaming=False, subsample_size=500) for key in language_codes}
for key, dataset in dataset_dicts.items():
    fn_kwargs = {"processor": processor, "language_code": key}

    dataset_dicts[key] = dataset.map(prepare_dataset, remove_columns=list(next(iter(dataset.values())).features), fn_kwargs=fn_kwargs).with_format("torch")
    
    dataset_dicts[key]["train"] = dataset_dicts[key]["train"].shuffle(seed=0) # for consistency of evaluation

    def is_audio_in_length_range(length, max_input_length=30.0):
        return length < max_input_length

    # filter out audios > 30.0 seconds as it would be truncated by processor
    dataset_dicts[key]["train"] = dataset_dicts[key]["train"].filter(is_audio_in_length_range, input_columns=["input_length"],)

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

# model.enable_input_require_grads()
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

model.gradient_checkpointing_enable({"use_reentrant": False})

model = get_peft_model(model, lora_config)

"""
Model  | Batch Size | Gradient Accumulation | Learning Rate
small	16	2	8
medium	2	16	1

If experience OOM, reduce per_device_train_batch_size by factor of 2 and increase gradient_accumulation_steps
"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./unified_output",
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
    #load_best_model_at_end=True,
    #metric_for_best_model="sacrebleu",
    #greater_is_better=True,
    push_to_hub=False,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

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

eval_dataset = {key: dataset_dicts[key]["validation"] for key in language_codes}

print(f"Language sequence: {language_codes}")

for idx, language_code in enumerate(language_codes):
    output_dir = "./output-unified"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # Set per-language checkpoint directory
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=1000 + 1000 * idx,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=100,
        report_to=["tensorboard"],
        #load_best_model_at_end=True,
        #metric_for_best_model="sacrebleu",
        #greater_is_better=True,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dicts[language_code]["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    checkpoint_path = get_last_checkpoint(output_dir) 

    print(f"Training on {language_code} - Resuming from checkpoint: {checkpoint_path}")

    # **Train with checkpoint resumption**
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # Save after training is done
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
