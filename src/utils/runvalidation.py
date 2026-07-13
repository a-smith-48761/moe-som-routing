import argparse

from transformers import GemmaTokenizerFast, Gemma3ForCausalLM, Trainer
from datasets import load_dataset

from ..models.gemma3moe import Gemma3MoEForCausalLM
from ..training import preprocess_qa_dataset, CausalLMCollator


DEFAULT_MODEL_ID = "google/gemma-3-270m-it"
DEFAULT_DATASET_ID = "openai/gsm8k"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_SAMPLE_EVERY = 0
DEFAULT_SAMPLE_MAX = 5
DEFAULT_GENERATE_MAX_NEW_TOKENS = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model on a dataset split."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier or local directory containing model to load."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset identifier to load."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_DATASET_CONFIG,
        help="Dataset configuration name."
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help="Print detokenized input and prediction for every Nth evaluation example. Set 0 to disable."
    )
    parser.add_argument(
        "--sample-max",
        type=int,
        default=DEFAULT_SAMPLE_MAX,
        help="Maximum number of sample outputs to print when sampling evaluation examples."
    )
    parser.add_argument(
        "--generate-max-new-tokens",
        type=int,
        default=DEFAULT_GENERATE_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate for each sample prediction."
    )
    return parser.parse_args()


def load_model(model_id: str):
    if model_id.startswith("google/"):
        return Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            attn_implementation="sdpa",
        )

    return Gemma3MoEForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        attn_implementation="sdpa",
    )


def print_sample_predictions(
    model,
    tokenizer,
    dataset,
    sample_every: int,
    sample_max: int,
    max_new_tokens: int,
) -> None:
    if sample_every <= 0:
        return

    print(
        f"Printing sample predictions for every {sample_every} evaluation examples"
        f" (up to {sample_max})."
    )

    sample_indices = list(range(0, len(dataset), sample_every))[:sample_max]
    for sample_index in sample_indices:
        item = dataset[sample_index]
        question = item.get("question")

        if question is None:
            print(f"Skipping sample {sample_index}: no question field available.")
            continue

        prompt_messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        generated = model.generate(
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(
            generated[0][len(prompt.input_ids) :],
            skip_special_tokens=True,
        )
        prompt_text = tokenizer.decode(
            prompt.input_ids,
            skip_special_tokens=True,
        )

        print("--- Sample", sample_index, "---")
        print("QUESTION:", question)
        print("INPUT PROMPT:", prompt_text)
        print("PREDICTED OUTPUT:", generated_text)
        print()


def main() -> None:
    args = parse_args()

    model = load_model(args.model_id)
    tokenizer = GemmaTokenizerFast.from_pretrained(args.model_id)

    dataset = load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)
    tokenized_dataset = dataset.map(
        preprocess_qa_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    trainer = Trainer(
        model=model,
        data_collator=CausalLMCollator(tokenizer),
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset)

    if args.sample_every > 0:
        print_sample_predictions(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            sample_every=args.sample_every,
            sample_max=args.sample_max,
            max_new_tokens=args.generate_max_new_tokens,
        )

    print(metrics)


if __name__ == "__main__":
    main()
