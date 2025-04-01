import torch

from datasets import IterableDatasetDict, load_dataset, Audio, concatenate_datasets
from transformers import TrainerCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from torch.utils.data import IterableDataset

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

def take_dataset(dataset_path, language_code, split_percentage="", streaming=False, subsample_size=500, combine_train_val=False):
    dataset = IterableDatasetDict()

    train_split = "train" if split_percentage == "" else f"train[:{split_percentage}]"
    dev_split = "dev" if split_percentage == "" else f"dev[:{split_percentage}]"
    test_split = "test" if split_percentage == "" else f"test[:{split_percentage}]"
    
    if combine_train_val:
        dataset["train"] = concatenate_datasets([load_dataset(dataset_path, language_code, split=train_split, streaming=streaming), load_dataset(dataset_path, language_code, split=dev_split, streaming=streaming)])
    else:
        dataset["train"] = load_dataset(dataset_path, language_code, split=train_split, streaming=streaming)
        dataset["validation"] = load_dataset(dataset_path, language_code, split=dev_split, streaming=streaming).shuffle(seed=0).take(subsample_size)

    dataset["test"] = load_dataset(dataset_path, language_code, split=test_split, streaming=streaming).shuffle(seed=0).take(subsample_size)
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # cast all to 16kHz

    return dataset


"""
In our pre-processing strategy, we will do normalization as described in the Whisper paper with the exception of lower-case and punctuation removal. 

However, during evaluation the normalization is fully applied
"""

def prepare_dataset(batch, processor, language_code, do_lower_case=False, do_remove_punctuation=False):
    #normalizer = EnglishTextNormalizer() if language_code == "en" else BasicTextNormalizer()
    normalizer = BasicTextNormalizer()

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