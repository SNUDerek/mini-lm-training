# mini lm from scratch (WIP)

## goal

train a small transformer LM using huggingface `nanotron` library, for educational purposes.

plan to do full training including pre-training, annealing/mid-training, SFT and other post-training.

target data is English-language public domain texts (cutoff date approximately 1930).

## progress

- [x] pretraining data processing scripts (gutenberg, ncse v2, oldbailey)
- [x] pretraining tokenizer fitting and dataset packing scripts
- [x] pretraining datasets - training, validation 
- [ ] nanotron-based pretraining script and config
- [ ] nanotron-based pretraining run
- [ ] gutenberg SFT dataset processing scripts (extract dialogs from target books)

## initial setup

install `uv` if needed

update with `uv sync`

## data processing

### gutenberg

```
uv run scripts/preprocessing/process_gutenberg.py \
--input-path /data/datasets/gutenberg \
--output-path /data/train_data/gutenberg \
--workers 8
```

### british library books

NOTE: script technically untested, i processed from notebook (and then copied code to script)

```
uv run scripts/preprocessing/process_blbooks.py \
--jsonl-dir /data/datasets/british-library-books/jsonl \
--raw-outpath /data/train_data/blbooks/raw \
--clean-outpath /data/train_data/blbooks/cleaned
```

### ncse v2

```
uv run scripts/preprocessing/process_ncse.py \
--input-path /data/datasets/ncse_v2 \
--output-path /data/train_data/ncse_v2
```

NOTE: LLM artifacts are still present. due to small number of files, quick visual inspection was done and repeated lines removed. Also dropped all Publisher's Circular files.

### old bailey 2

```
uv run scripts/preprocessing/process_old_bailey.py \
--input-path /data/datasets/OldBaileyCorpus2/OBC2 \
--output-path /data/train_data/old_bailey
```

## post-ocr correction (pleias)

NOTE: script technically untested, i processed from notebook (and then copied code to script)

```
uv run scripts/preprocessing/process_ocs_correction.py \
--input-path /data/datasets/pleias-post-ocr-correction \
--output-path /data/train_data/post-ocr-correction
```


## split dataset

defaults to 1% due to already small dataset.

```
uv run scripts/create_val_split.py \
--train-path /data/train_data \
--val-path /data/val_data
```

## deduplication

there are two rough deduplication scripts that use `Simhash` to get candidate matches, then use exact jaccard similarity over word n-grams to folder the candidates. I then removed anything with `word_shingle_jaccard` or `word_shingle_containment` >= 0.90. Then I grouped using a graph, and kept the longest document in each connected component. CSV of the results is in ./info, along with final train and validation file lists.

## fit tokenizer

creates a byte-pair encoding tokenizer, converts to `transformers` tokenizer with chatML template, and saves it.

optional flag `--replace-newlines` replaces all newlines with spaces

```
uv run scripts/fit_tokenizer.py \
--dataset /data/train_data \
--output-path /data/artifacts/tokenizer32K \
--vocab-size 32000
```

### prepack datasets

creates pre-tokenized, pre-packed chunks of specified context length (default 4096).

TODO: more complex dataset configuration such as dynamic chunking

```
uv run scripts/prepack_data.py \
--tokenizer-path /data/artifacts/tokenizer32K/tokenizer \
--data-path /data/val_data \
--output-path /data/artifacts/val_dataset
```

```
uv run scripts/prepack_data.py \
--tokenizer-path /data/artifacts/tokenizer32K/tokenizer \
--data-path /data/train_data \
--output-path /data/artifacts/train_dataset
```

## training with huggingface

first, create/edit a config yaml.

then start training:

```
cd training
uv python train.py --config <path>/<to>/<config>.yaml
```

can monitor with tensorboard: (*runs path specified in config yaml*)

```
uv run tensorboard --logdir <path>/<to>/<output_dir>/runs
```

### dockerize (todo)

build the docker image

```
docker build -t minillm:latest . --no-cache
```

## training with nanotron (WIP)

build docker image - will take some time due to compiling Flash Attention

```
docker build -t nanotron:latest ./nanotron
```

start docker container via docker compose

```
docker compose up -
```

## AI disclosure

- coded in antigravity
- ai tools used: antigravity ide, gemini (web and ide extension), claude (web, claude code extension), openai gpt (web, codex extension)
- unless indicated below, scripts by me, reviewed/corrected by AI:
    - Old Bailey XML parsing functions by chatgpt
    - deduplication scripts about 50/50 between me and chatgpt over numerous iterations
    - nanotron Dockerfile by chatgpt
    - training code based on nanotron example, edited by me, corrected by AI