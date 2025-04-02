from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

import evaluate
import torch
import numpy as np

from transformers import DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets, Dataset

LANGUAGES = {
   'hi': "hin_Deva", 
   #'id': "ind_Latn", # remove first, data leakage
   'ms': "zsm_Latn",
   'th': "tha_Thai",
   'tl': "tgl_Latn",
   'vi': "vie_Latn",
   'zh': "zho_Hans",
   'en': "eng_Latn"
}

max_length = 512

class DataCollatorForSeq2SeqRandomizedBidirectional(DataCollatorWithPadding):
    def __call__(self, features, return_tensors="pt"):
        tokenized_inputs = []
        tokenized_labels = []
        
        for feature in features:
            src_lang = LANGUAGES['en']
            src = feature['en']

            tgt_lang = LANGUAGES[feature['target_lang']]
            tgt = feature['other']
            
            if np.random.rand() < 0.5:
                src_lang, tgt_lang = tgt_lang, src_lang
                src, tgt = tgt, src

            # Set the tokenizer's source language dynamically
            self.tokenizer.src_lang = src_lang
            tokenized_input = self.tokenizer(
                src,
                truncation=True,
                max_length=self.max_length,
            )
            
            self.tokenizer.src_lang = tgt_lang
            tokenized_label = self.tokenizer(
                tgt,
                truncation=True,
                max_length=self.max_length,
            )
            
            tokenized_inputs.append(tokenized_input)
            tokenized_labels.append(tokenized_label)
        
        source_inputs = {
            "input_ids": [x["input_ids"] for x in tokenized_inputs],
            "attention_mask": [x["attention_mask"] for x in tokenized_inputs],
        }
        
        target_inputs = {
            "input_ids": [x["input_ids"] for x in tokenized_labels],
        }
        
        batch = self.tokenizer.pad(
            source_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        labels_batch = self.tokenizer.pad(
            target_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        labels_batch["input_ids"][labels_batch["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels_batch["input_ids"]
        
        return batch

model_path = "facebook/nllb-200-distilled-600M"
dataset_path = "keeve101/balanced-multi-corpora-mt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
data_collator = DataCollatorForSeq2SeqRandomizedBidirectional(tokenizer=tokenizer, max_length=max_length)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
    
model.config.use_cache = False

model.model.encoder.embed_tokens.register_forward_hook(make_inputs_require_grad)

model.gradient_checkpointing_enable({"use_reentrant": False})

model = get_peft_model(model, lora_config)

model.to(device) 

config_names = get_dataset_config_names(dataset_path)
config_names = [config_name for config_name in config_names if config_name in LANGUAGES.keys()]

dataset: Dataset = concatenate_datasets([load_dataset(dataset_path, config_name, split="train").filter(lambda x: x["en"] != "" and x["other"] != "") for config_name in config_names])

train_test_split = dataset.train_test_split(test_size=0.01)

train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("sacrebleu")
    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=label_str, tokenize="intl") 
    bleu_score = bleu["score"]

    return {"wer": wer, "cer": cer, "sacrebleu": bleu_score}

output_dir = "nllb-finetune-lora-balanced-multi-corpora"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    #max_steps=2000,
    num_train_epochs=1,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    save_steps=10000,
    eval_steps=10000,
    logging_steps=200,
    report_to=["tensorboard"],
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

checkpoint_path = get_last_checkpoint(output_dir) 

# **Train with checkpoint resumption**
trainer.train(resume_from_checkpoint=checkpoint_path)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
