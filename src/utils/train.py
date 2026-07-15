import argparse
import torch

from transformers import GemmaTokenizerFast, Gemma3ForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from bitsandbytes.optim import AdamW8bit

from ..models.gemma3moe import Gemma3MoEForCausalLM
from ..training import preprocess_qa_dataset, CausalLMCollator, load_model

DEFAULT_MODEL_ID = "google/gemma-3-270m-it"
DEFAULT_DATASET_ID = "openai/gsm8k"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_EVAL_SPLIT = "test"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_OUTPUT_DIR = "output/snapshots"
DEFAULT_SEED = 44961561

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train model."
    )
    parser.add_argument(
        "--model",
        dest="model_id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier or local directory containing model to load."
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset identifier to load."
    )
    parser.add_argument(
        "--split",
        dest="dataset_split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to train on."
    )
    parser.add_argument(
        "--eval-split",
        dest="dataset_eval_split",
        default=DEFAULT_DATASET_EVAL_SPLIT,
        help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--config",
        dest="dataset_config",
        default=DEFAULT_DATASET_CONFIG,
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for storing output snapshots"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile",
        help="Use PyTorch compilation"
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="PyTorch compile backend to use"
    )
    parser.add_argument(
        "--create-eval-split",
        action="store_true",
        dest="create_eval_split",
        help="Use 10% of the training split to create an evaluation split"
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for creating splits and/or shuffling"
    )
    return parser.parse_args()


def build_composite_optimizer (model):
    optimizer_parameters = [p for n, p in model.named_parameters() ]    # FIXME: exclude routers if the network is using som training

    if isinstance(model, Gemma3ForCausalLM) or model.config.expert_router_type != "som":
        return AdamW8bit(optimizer_parameters)
    else:
        raise NotImplemented("SOM router training not implemented yet")

def main() -> None:
    args = parse_args()

    model = load_model(args.model_id)
    tokenizer = GemmaTokenizerFast.from_pretrained(args.model_id)

    dataset = load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)

    if args.create_eval_split:
        created_splits = dataset.train_test_split (test_size=0.1, seed=args.seed ^ 31415)
        dataset = created_splits["train"]
        eval_dataset = created_splits["test"]
    else:
        eval_dataset =  load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_eval_split)
        
    tokenized_dataset = dataset.map(
        preprocess_qa_dataset,
        batched=True,
        #remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_qa_dataset,
        batched=True,
        #remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # check that the training dataset is valid before continuin
    def count_valid_labels(example):
        return sum(label != -100 for label in example["labels"])


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
        
        return

    compile_args = {}
    if args.compile:
        compile_args = {
            "torch_compile": True,
            "torch_compile_backend": args.compile_backend
        }

    trainingArguments = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        fp16 = True,
        logging_steps = 100,
        include_num_input_tokens_seen = "non_padding",
        eval_strategy = "steps",
        eval_steps = 100,
        seed = args.seed ^ 1415961,  # ensuring samplign is reproducible
        **compile_args,
    )
    trainer = Trainer(
        args=trainingArguments,
        model=model,
        data_collator=CausalLMCollator(tokenizer),
        processing_class=tokenizer,
        train_dataset = tokenized_dataset,
        eval_dataset = tokenized_eval_dataset,
        optimizers=(
            build_composite_optimizer (model),
            None    # use the default schedule
        )
    )

    metrics = trainer.train()
    trainer.save_model ()

    print(metrics)


if __name__ == "__main__":
    main()
