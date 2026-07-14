from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer, LlamaForCausalLM, set_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Top-level YAML value must be a mapping.")

    return config


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def resolve_model_path(model_path: Path, checkpoint: str) -> Path:
    if checkpoint != "latest":
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = model_path / checkpoint
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")
        return candidate

    checkpoints = [
        path
        for path in model_path.glob("checkpoint-*")
        if path.is_dir() and checkpoint_step(path) >= 0
    ]

    if checkpoints:
        return max(checkpoints, key=checkpoint_step)

    return model_path


def resolve_tokenizer_path(
    checkpoint_path: Path,
    model_path: Path,
    config: dict[str, Any] | None,
    explicit_path: str | None,
) -> Path:
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(Path(explicit_path))

    candidates.extend([checkpoint_path, model_path])

    if config is not None:
        tokenizer_path = config.get("tokenizer", {}).get("path")
        if tokenizer_path is not None:
            candidates.append(Path(tokenizer_path))

    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() or (
            candidate / "tokenizer_config.json"
        ).exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a tokenizer in: {searched}")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively generate text from a trained mini LLM checkpoint."
    )
    parser.add_argument(
        "--config",
        help="Path to the training YAML. Used to infer output_dir and tokenizer path.",
    )
    parser.add_argument(
        "--model-path",
        help="Model output directory. Overrides training.output_dir from --config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help=(
            "Checkpoint directory to load. Use 'latest' for the highest-numbered "
            "checkpoint-* under model-path, or pass a relative/absolute path."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Tokenizer directory. Defaults to checkpoint, model-path, then config tokenizer.path.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print prompt plus completion instead of only the generated continuation.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.config is None and args.model_path is None:
        raise ValueError("Pass --config, --model-path, or both.")

    config = load_yaml(args.config) if args.config is not None else None

    model_path_value = args.model_path
    if model_path_value is None:
        assert config is not None
        model_path_value = config["training"]["output_dir"]

    model_path = Path(model_path_value)
    checkpoint_path = resolve_model_path(model_path, args.checkpoint)
    tokenizer_path = resolve_tokenizer_path(
        checkpoint_path,
        model_path,
        config,
        args.tokenizer_path,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = resolve_dtype(args.dtype, device)

    if args.seed is not None:
        set_seed(args.seed)

    print(f"Loading model from:     {checkpoint_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"Device:                 {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    context_length = getattr(model.config, "max_position_embeddings", None)
    if context_length is not None and args.max_new_tokens >= context_length:
        raise ValueError(
            f"--max-new-tokens must be smaller than context length {context_length}."
        )

    print("\nEnter an empty prompt or press Ctrl-D to exit.\n")

    while True:
        try:
            prompt = input("prompt> ")
            prompt = prompt.strip("\n").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if context_length is not None:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": context_length - args.max_new_tokens,
                }
            )

        inputs = tokenizer(prompt, **tokenizer_kwargs).to(device)
        prompt_tokens = inputs["input_ids"].shape[-1]

        # print(inputs)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": not args.no_sample,
            "repetition_penalty": args.repetition_penalty,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            "eos_token_id": tokenizer.eos_token_id,
        }

        if not args.no_sample:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)[0]

        if not args.include_prompt:
            output_ids = output_ids[prompt_tokens:]

        print()
        print(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        print()


if __name__ == "__main__":
    main()
