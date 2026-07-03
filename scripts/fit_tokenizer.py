import argparse
import pathlib
from pathlib import Path
import random

import tokenizers

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


VOCAB_SIZE: int = 32000

SPECIAL_TOKENS: dict = {
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "unk_token": "<unk>",
    "msg_start_token": "<|im_start|>",
    "msg_end_token": "<|im_end|>",
}

# from gutenberg https://www.gutenberg.org/files/228/228-h/228-h.htm#chap01
TEST_TEXT = """Arms, and the man I sing, who, forc’d by fate,
And haughty Juno’s unrelenting hate,
Expell’d and exil’d, left the Trojan shore.
Long labours, both by sea and land, he bore,
And in the doubtful war, before he won
The Latian realm, and built the destin’d town;
His banish’d gods restor’d to rites divine,
And settled sure succession in his line,
From whence the race of Alban fathers come,
And the long glories of majestic Rome.
O Muse! the causes and the crimes relate;
What goddess was provok’d, and whence her hate;
For what offence the Queen of Heav’n began
To persecute so brave, so just a man;
Involv’d his anxious life in endless cares,
Expos’d to wants, and hurried into wars!
Can heav’nly minds such high resentment show,
Or exercise their spite in human woe?"""


def get_text_file_paths(path_sting: str) -> list[Path]:
    """load text file paths from corpus location"""
    data_dir: Path = pathlib.Path(path_sting)
    data_filepaths: list[Path] = list(data_dir.rglob("*.txt"))
    return data_filepaths


def batch_iterator(file_paths: list[Path], batch_size=100, replace_newlines: bool=False):
    """iterator over all data"""
    for i in range(0, len(file_paths), batch_size):
        yield [
            f.read_text(encoding="utf-8", errors="replace").replace("\n", " ") if replace_newlines 
            else f.read_text(encoding="utf-8", errors="replace")
            for f in file_paths[i : i + batch_size]
        ]


def fit_new_tokenizer(data_filepaths: list[Path], vocab_size: int, batch_size: int, replace_newlines: bool) -> Tokenizer:
    """fit new tokenizer from data"""

    # BPE tokenizer with byte-level preprocessing
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))

    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # note: need to send the initial alphabet or else won't handle ood words
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS.values()),
        initial_alphabet=tokenizer.pre_tokenizer.alphabet(),
    )

    tokenizer.train_from_iterator(batch_iterator(
        data_filepaths, 
        batch_size=batch_size, 
        replace_newlines=replace_newlines), 
        trainer
    )

    # since we need to specify the vocab, do this after fitting i guess
    eos_token = SPECIAL_TOKENS["eos_token"]
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"$A {eos_token}",
        pair=f"$A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[(f"{eos_token}", tokenizer.get_vocab()[eos_token])],
    )

    return tokenizer


def create_transformers_tokenizer(raw_tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    """create final transformer tokenizer with chat template"""

    auto_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token=None,
        eos_token=SPECIAL_TOKENS["eos_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        pad_token=None,
        mask_token=None,
        additional_special_tokens=[
            SPECIAL_TOKENS["msg_start_token"],
            SPECIAL_TOKENS["msg_end_token"],
        ]
    )

    CHATML_TEMPLATE = """{%- for message in messages %}
        {{- 'MSG_START_TOKEN' + message['role'] + '\\n' }}
        {{- message['content'] + 'MSG_END_TOKEN' + '\\n' }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- 'MSG_START_TOKEN' + 'assistant' + '\\n' }}
    {%- endif %}""".replace(
        "MSG_START_TOKEN", SPECIAL_TOKENS["msg_start_token"]
    ).replace(
        "MSG_END_TOKEN", SPECIAL_TOKENS["msg_end_token"]
    )

    auto_tokenizer.chat_template = CHATML_TEMPLATE

    return auto_tokenizer


# save tokenizer
def save_tokenizer(tokenizer: Tokenizer, output_base_path: str) -> str:
    """save base tokenizers.Tokenizer to <base_dir>/base_tokenizer/tokenizer.json"""
    out_file_path = pathlib.Path(output_base_path, "base_tokenizer", "tokenizer.json")
    out_file_path.parent.mkdir(exist_ok=True, parents=True)
    save_path_string: str = str(out_file_path.absolute())
    tokenizer.save(save_path_string)
    return save_path_string


# save autotokenizer
def save_autotokenizer(autotokenizer: PreTrainedTokenizerFast, output_base_path: str) -> str:
    """save PreTrainedTokenizerFast to <base_dir>/tokenizer"""
    out_file_path = pathlib.Path(output_base_path, "tokenizer")
    out_file_path.mkdir(exist_ok=True, parents=True)
    save_path_string: str = str(out_file_path.absolute())
    autotokenizer.save_pretrained(save_path_string)
    return save_path_string


## checks
def test_tokenizer_vocabulary(tokenizer: Tokenizer, exp_voc_size: int) -> bool:
    print("testing raw tokenizer vocabulary")
    inv_voc = tokenizer.get_vocab()
    voc = dict([(v, k) for k, v in inv_voc.items()])
    print(f"vocab length: {len(voc)}")
    assert len(voc) <= exp_voc_size
    print("sample vocabulary items:")
    print(sorted(voc.items())[:16])
    return True


def test_tokenizer_functionality(tokenizer: Tokenizer) -> bool:
    print("testing tokenizer:")
    print("input :")
    print(TEST_TEXT + "\n\n")

    enc = tokenizer.encode(TEST_TEXT)
    print(f"ids   : {enc.ids}")
    print(f"tokens: {enc.tokens}\n\n")

    out = tokenizer.decode(enc.ids)
    print(f"output:\n{out}\n")
    if TEST_TEXT.strip() != out.strip():
        print("WARNING: decoded text is not the same as original, is this okay?")
    return TEST_TEXT.strip() == out.strip()


def test_autotokenizer_functionality(autotokenizer: PreTrainedTokenizerFast) -> bool:
    print("testing autotokenizer:\n")
    print("input :")
    print(TEST_TEXT + "\n\n")

    enc = autotokenizer.encode(TEST_TEXT)
    print(f"ids: {enc}\n\n")

    out = autotokenizer.decode(enc, skip_special_tokens=True)
    print(f"output:\n{out}\n")
    if TEST_TEXT.strip() != out.strip():
        print("WARNING: decoded text is not the same as original, is this okay?")
    return TEST_TEXT.strip() == out.strip()


def test_autotokenizer_chat_template(auto_tokenizer: PreTrainedTokenizerFast) -> bool:
    messages = [
        {"role": "system", "content": "You are a debug testing bot."},
        {"role": "user", "content": "Hey, how are you doing today?"},
    ]
    print(f"testing chat template:\n")
    print(f"messages:")
    for message in messages:
        print(message)
    print()

    print("no generation prompt:\n")
    print(auto_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    print()

    print("with generation prompt:\n")
    print(auto_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    print()

    return True


def main(args):
    # get documents
    if not pathlib.Path(args.dataset).is_dir():
        raise NotADirectoryError(f"no data directory: {args.dataset}")
    file_paths = get_text_file_paths(args.dataset)
    if args.verbose:
        print(f"found {len(file_paths)} txt files in {args.dataset}")

    # sample if requested
    if args.max_documents > -1:
        random.shuffle(file_paths)
        file_paths = file_paths[: args.max_documents]
        if args.verbose:
            print(f"--max-documents is {args.max_documents}, documents set to {len(file_paths)}")

    # fit tokenizer
    if args.verbose:
        print("\n-----------------------------------------------------\n")
        print(f"fitting Tokenizer on data with vocab {args.vocab_size}, this may take some time...")
    tokenizer = fit_new_tokenizer(
        file_paths,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        replace_newlines=args.replace_newlines
    )

    # save tokenizer
    out_path = save_tokenizer(tokenizer, args.output_path)
    if args.verbose:
        print(f"saved tokenizer to '{out_path}'\n")

    # test tokenizer
    print("\n-----------------------------------------------------\n")
    test_tokenizer_vocabulary(tokenizer, exp_voc_size=args.vocab_size)
    test_tokenizer_functionality(tokenizer)

    # create autotokenizer
    if args.verbose:
        print("\n-----------------------------------------------------\n")
        print("creating transformers tokenizer with chatML template...")
    auto_tokenizer = create_transformers_tokenizer(tokenizer)

    # save autotokenizer
    out_path = save_autotokenizer(auto_tokenizer, args.output_path)
    if args.verbose:
        print(f"saved transformers tokenizer to '{out_path}'\n")

    # test autotokenizer
    print("\n-----------------------------------------------------\n")
    test_autotokenizer_functionality(auto_tokenizer)
    test_autotokenizer_chat_template(auto_tokenizer)

    print(f"tokenizer fitting is done, see '{args.output_path}'")


if __name__ == "__main__":
    # read args
    parser = argparse.ArgumentParser(
        prog="Mini-LLM BPE Tokenizer Fitter",
        description="fit tokenizer to data and save",
        epilog="Run with --help to see parameters",
    )
    # output base dir, vocab size, --max_documents, --batch_size
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="directory containing text files to train tokenizer on",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./tokenizer",
        help="directory to save tokenizer to. default: ./tokenizer",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32_000, help="total vocabulary size. default: 32000"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=-1,
        help="max (shuffled) documents to use. default: -1 (all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="batch size for tokenizer fitting. default: 100"
    )
    parser.add_argument("--replace-newlines", action="store_true", help="replace all newline characters with ' '")
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
    args = parser.parse_args()

    main(args)
