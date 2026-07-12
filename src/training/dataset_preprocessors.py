from transformers import { PreTrainedTokenizerBase }
#
# preprocess data sets for question/answer to appropriate format for LM input/output.
#
def preprocess_qa_dataset(examples, tokenizer:PreTrainedTokenizerBase = None):
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

__all__ = ["preprocess_qa_dataset"]
