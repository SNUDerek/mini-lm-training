from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import LlamaConfig, PreTrainedTokenizerBase



def resolve_dataset(
    path: str | Path,
    split: str | None = None,
) -> Dataset:
    dataset = load_from_disk(str(path))

    if isinstance(dataset, DatasetDict):
        if split is None:
            raise ValueError(
                f"{path} contains a DatasetDict; specify a split in the YAML."
            )
        if split not in dataset:
            raise KeyError(
                f"Split {split!r} not found. Available splits: {list(dataset)}"
            )
        return dataset[split]

    if split is not None:
        print(
            f"Warning: split={split!r} was specified, but {path} contains "
            "a single Dataset. Ignoring the split name."
        )

    return dataset


def validate_dataset(
    dataset: Dataset,
    *,
    name: str,
    context_length: int,
    vocab_size: int,
) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty.")

    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"{name} dataset does not contain an 'input_ids' column. "
            f"Columns: {dataset.column_names}"
        )

    # Inspect several rows without scanning the entire dataset.
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})

    for index in indices:
        row = dataset[index]["input_ids"]

        if len(row) != context_length + 1:
            raise ValueError(
                f"{name}[{index}] has {len(row)} tokens; expected "
                f"{context_length + 1}."
            )

        minimum = min(row)
        maximum = max(row)

        if minimum < 0:
            raise ValueError(
                f"{name}[{index}] contains negative token ID {minimum}."
            )

        if maximum >= vocab_size:
            raise ValueError(
                f"{name}[{index}] contains token ID {maximum}, but model "
                f"vocab_size is {vocab_size}."
            )


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def build_model_config(
    model_values: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> LlamaConfig:
    values = dict(model_values)

    configured_vocab_size = values.get("vocab_size")
    tokenizer_vocab_size = len(tokenizer)

    if configured_vocab_size is None:
        values["vocab_size"] = tokenizer_vocab_size
    elif configured_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"YAML vocab_size={configured_vocab_size}, but "
            f"len(tokenizer)={tokenizer_vocab_size}."
        )

    values.setdefault("bos_token_id", tokenizer.bos_token_id)
    values.setdefault("eos_token_id", tokenizer.eos_token_id)
    values.setdefault("pad_token_id", tokenizer.pad_token_id)

    return LlamaConfig(**values)