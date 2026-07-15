from transformers import PreTrainedTokenizerBase
#
# preprocess data sets for question/answer to appropriate format for LM input/output.
#
def preprocess_qa_dataset(examples, tokenizer:PreTrainedTokenizerBase = None):
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "untokenized": [],
    }

    if "question" in examples:
        qfield = "question"
    else:
        qfield = "problem"
        
    for question, answer in zip(
        examples[qfield],
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
        batch["untokenized"].append(question + "\n" + answer)

    return batch


#
# Filter for removing examples that have no labels from tokenized datasets
#

def filter_require_label_presence (example):
    return sum(label != -100 for label in example["labels"]) > 0

#
# check a dataset doesn't contain items with no labels
#

# check that the training dataset is valid before continuin
def count_valid_labels(example):
    return sum(label != -100 for label in example["labels"])

def check_dataset_has_labels (tokenized_dataset):
    invalid_indices = [
        index
        for index, example in enumerate(tokenized_dataset)
        if count_valid_labels(example) == 0
    ]

    if len(invalid_indices) > 0:
        print("Examples with no valid labels:", len(invalid_indices))
        print("First indices:", invalid_indices[:20])
        print("Number of input tokens:", [ len(tokenized_dataset[i]["input_ids"]) for i in invalid_indices ])
        #item = tokenized_dataset[invalid_indices[0]]
        #print(item["problem"] if "problem" in item else item["question"])
        #print(item["answer"])
        
        return False
    return True

__all__ = ["preprocess_qa_dataset", "filter_require_label_presence", "check_dataset_has_labels"]
