from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn.functional as F
import yaml
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
    set_seed,
)

from minillm.collator import PackedCausalCollator
from minillm.trainer import PackedCausalTrainer
from minillm.data import (
    build_model_config, 
    resolve_dataset, 
    validate_dataset, 
    count_parameters,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Top-level YAML value must be a mapping.")

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML file.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Resume from the latest checkpoint in output_dir, or provide "
            "an explicit checkpoint directory."
        ),
    )
    args = parser.parse_args()

    raw_config = load_yaml(args.config)

    data_config = raw_config["data"]
    model_values = raw_config["model"]
    training_values = raw_config["training"]

    seed = int(training_values.get("seed", 42))
    set_seed(seed)

    tokenizer_path = raw_config["tokenizer"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
    )

    llama_config = build_model_config(model_values, tokenizer)
    context_length = llama_config.max_position_embeddings

    train_dataset = resolve_dataset(
        data_config["train_path"],
        data_config.get("train_split"),
    )

    eval_path = data_config.get("eval_path")
    eval_dataset = None

    if eval_path:
        eval_dataset = resolve_dataset(
            eval_path,
            data_config.get("eval_split"),
        )

    validate_dataset(
        train_dataset,
        name="train",
        context_length=context_length,
        vocab_size=llama_config.vocab_size,
    )

    if eval_dataset is not None:
        validate_dataset(
            eval_dataset,
            name="eval",
            context_length=context_length,
            vocab_size=llama_config.vocab_size,
        )

    llama_config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(llama_config)

    if training_values.pop("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )
        model.config.use_cache = False

    training_args = TrainingArguments(**training_values)

    total_parameters, trainable_parameters = count_parameters(model)

    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )

    steps_per_epoch = math.ceil(
        len(train_dataset) / effective_batch_size
    )

    print(f"Tokenizer size:           {len(tokenizer):,}")
    print(f"Training samples:         {len(train_dataset):,}")
    print(f"Context length:           {context_length:,}")
    print(f"Total parameters:         {total_parameters:,}")
    print(f"Trainable parameters:     {trainable_parameters:,}")
    print(f"Effective batch size:     {effective_batch_size:,} sequences")
    print(f"Approx. steps per epoch:  {steps_per_epoch:,}")
    print(
        f"Tokens per optimizer step:"
        f" {effective_batch_size * context_length:,}"
    )

    collator = PackedCausalCollator(
        context_length=context_length,
    )

    trainer = PackedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None and eval_loss < 100:
            metrics["perplexity"] = math.exp(eval_loss)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_state()


if __name__ == "__main__":
    main()