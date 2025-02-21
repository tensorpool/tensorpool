import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek Distill Qwen-2 on a dataset.")

    # Basic model/data params
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="The model name or path on Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of the dataset from the Hugging Face Hub or a local path."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration name (if needed)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./deepseek-qwen2-finetuned",
        help="Where to store the final model."
    )

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Eval batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X update steps.")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="When to evaluate (epoch/steps).")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="When to save the model (epoch/steps).")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps when using 'steps' strategy.")
    

    # Misc
    parser.add_argument("--block_size", type=int, default=512, help="Block size for grouping text.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For quick debugging, limit train samples.")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integration: 'none', 'wandb', etc.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed-precision training if GPU is available."
    )

    return parser.parse_args()


def group_texts(examples, block_size):
    """
    Concatenate texts and split into blocks of size `block_size`.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # drop the small remainder
    total_length = (total_length // block_size) * block_size
    return {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }


def main():
    args = parse_args()

    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # 2. Load dataset
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    # 3. Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    # Optional: limit training set size for debugging
    if args.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))

    # 4. Group text into blocks for causal LM
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, args.block_size),
        batched=True,
    )

    # 5. Data collator (creates labels from input_ids for causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 6. Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=args.report_to,
        learning_rate=args.learning_rate,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Train
    trainer.train()

    # 9. Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
