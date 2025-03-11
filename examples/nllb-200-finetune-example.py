from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from argparse import ArgumentParser

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

modelPath = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath, device_map="auto")

parser = ArgumentParser()
parser.add_argument('--source-lang', type=str, default='eng_Latn')
parser.add_argument('--target-lang', type=str, default='rus_Cyrl')
parser.add_argument('--delimiter', type=str, default=';')
args = parser.parse_args()

dff = pd.read_csv('dataset/data.csv', sep=args.delimiter)

source = dff[args.source_lang].values.tolist()
target = dff[args.target_lang].values.tolist()

max = 512
X_train, X_val, y_train, y_val = train_test_split(source, target, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max, return_tensors="pt")
y_train_tokenized = tokenizer(y_train, padding=True, truncation=True, max_length=max, return_tensors="pt")
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max, return_tensors="pt")
y_val_tokenized = tokenizer(y_val, padding=True, truncation=True, max_length=max, return_tensors="pt")


class ForDataset(torch.utils.data.Dataset):
    def init(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def len(self):
        return len(self.targets)

    def getitem(self, index):
        input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
        target_ids = torch.tensor(self.targets["input_ids"][index]).squeeze()

        return {"input_ids": input_ids, "labels": target_ids}


train_dataset = ForDataset(X_train_tokenized, y_train_tokenized)
test_dataset = ForDataset(X_val_tokenized, y_val_tokenized)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir="mymodel",
    evaluation_strategy="epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('finalmodel')