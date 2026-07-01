import argparse
import pathlib
import random
from pathlib import Path
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer


def text_iterator(file_paths: list[Path]) -> tuple[str, str]:
    """yield batches of texts"""
    for file in file_paths:
        yield file.read_text(encoding="utf-8", errors="replace"), file.name


def block_generator(tokenizer: AutoTokenizer, file_paths: list[Path], max_context: int):
    """process texts into fixed-size blocks (packing)"""
    block_size = max_context + 1
    buffer = []
    file_names = []

    for text, file_name in text_iterator(file_paths):
        batch_ids = tokenizer(
            text,
            truncation=False,
            padding=False,
        )["input_ids"]

        buffer.extend(batch_ids)
        file_names.extend([file_name] * len(batch_ids))

        while len(buffer) >= block_size:
            block = buffer[:block_size]
            del buffer[:block_size]
            yield {"input_ids": block, "file_names": list(dict.fromkeys(file_names[:block_size]))}


def main(args):

    tokenizer_path = args.tokenizer_path
    data_path = args.data_path
    output_path = args.output_path
    max_context = args.max_context

    data_files: list[Path] = sorted(list(pathlib.Path(data_path).rglob("*.txt")))
    random.shuffle(data_files)
    print(f"found {len(data_files)} train files")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(f"loaded tokenizer from {tokenizer_path}")

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "file_names": Sequence(Value("string"))
    })

    dataset = Dataset.from_generator(
        block_generator,
        gen_kwargs={
            "tokenizer": tokenizer,
            "file_paths": data_files,
            "max_context": max_context,
        },
        features=features,
    )

    dataset.save_to_disk(output_path)
    print(f"saved dataset to {output_path}")

    # Calculate number of training examples (total tokens / context length)
    total_tokens = sum(len(seq) for seq in dataset["input_ids"])
    num_training_examples = total_tokens // (max_context + 1)
    print(f"dataset has {num_training_examples} training examples with {total_tokens} total tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, required=True, help="path to trained tokenizer.")
    parser.add_argument("--data-path", type=str, required=True, help="path to training data.")
    parser.add_argument("--output-path", type=str, required=True, help="path to save prepacked dataset.")
    parser.add_argument("--max-context", type=int, default=4096, help="max context length for training examples. default: 4096")
    args = parser.parse_args()
    main(args)