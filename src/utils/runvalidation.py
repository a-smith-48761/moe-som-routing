import sys

import torch

from dataclasses import dataclass
from transformers import GemmaTokenizerFast, Gemma3ForCausalLM, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

from ..models.gemma3moe import Gemma3MoEForCausalLM


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"
datasetId = sys.argv[2] if len(sys.argv) > 2 else "openai/gsm8k"
datasetSplit = sys.argv[3] if len(sys.argv) > 3 else "test"
datasetConfig = sys.argv[4] if len(sys.argv) > 4 else "main"

if modelId.startswith("google/"):
    # official model, so use the standard class
    model = Gemma3ForCausalLM.from_pretrained(
        modelId,
        device_map="cuda",
        attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
    )
else:    
    model = Gemma3MoEForCausalLM.from_pretrained(
        modelId,
        device_map="cuda",
        attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
    )

tokenizer = GemmaTokenizerFast.from_pretrained (modelId)

#
# preprocess data sets for question/answer to appropriate format for LM input/output.
#
def preprocess_answer_only(examples):
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for question, answer in zip(
        examples["question"],
        examples["answer"],
    ):
        prompt_messages = [
            {
                "role": "user",
                "content": question,
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        full = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        full_ids = full.input_ids[:512]
        prompt_length = min(len(prompt.input_ids), len(full_ids))

        labels = full_ids.copy()
        labels[:prompt_length] = [-100] * prompt_length

        batch["input_ids"].append(full_ids)
        batch["attention_mask"].append([1] * len(full_ids))
        batch["labels"].append(labels)

    return batch

dataset = load_dataset(datasetId, datasetConfig, split=datasetSplit)
tokenized_dataset = dataset.map(preprocess_answer_only, batched = True, remove_columns = dataset.column_names)


@dataclass
class CausalLMCollator:
    tokenizer: object

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = [
            label + [-100] * (max_length - len(label))
            for label in labels
        ]

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch
    

trainer = Trainer  (
    model = model,
    data_collator = CausalLMCollator(tokenizer),
    processing_class = tokenizer
)
metrics = trainer.evaluate (eval_dataset = tokenized_dataset)

print (metrics)
