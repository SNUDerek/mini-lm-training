import argparse
import json
import pathlib
import re
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--jsonl-dir', type=str, required=True, help="path to the top directory containing the jsonl files")
parser.add_argument('--raw-outpath', type=str, required=True, help="path to the output path for raw texts")
parser.add_argument('--clean-outpath', type=str, required=True, help="path to the output path for cleaned texts")
args = parser.parse_args()

jsonl_top_path_str = args.jsonl_dir
output_path_str = args.raw_outpath
clean_out_path_str = args.clean_outpath


jsonl_top_path = pathlib.Path(jsonl_top_path_str)
jsonl_files = sorted(list(jsonl_top_path.rglob("*.jsonl")))
print(f"found {len(jsonl_files)} files")


output_path = pathlib.Path(output_path_str)
if output_path.exists():
    print(f"{output_path} exists, delete first or specify new directory")
    sys.exit(0)
else:
    output_path.mkdir(parents=True)


books_data = []

for jsonl_file in tqdm(jsonl_files):

    fname = jsonl_file.name
    fname.replace(".jsonl", ".txt")

    raw = jsonl_file.read_text(encoding='utf-8')
    raw_lines = raw.split('\n')
    json_data = [json.loads(s) for s in raw_lines if s]

    # like gutenberg, we only want the contents, not title, author. but we'll save for archiving
    languages = [v for k, v in json_data[0].items() if k.startswith("Language_") and v]
    metadata = {
        "record_id": json_data[0].get('record_id'),
        "Name": json_data[0].get('Name'),
        "title": json_data[0].get('title'),
        "Publisher": json_data[0].get('Publisher'),
        "Physical description": json_data[0].get('Physical description'),
        "Languages": languages,
        "multi_language": json_data[0].get('multi_language'),
    }

    # do language check
    if 'English' in metadata["Languages"] and not metadata["multi_language"]:

        # extract the contents
        texts = []
        for entry in json_data:
            if entry.get('text'):
                texts.append(entry['text'])

        # only write long texts
        if len(texts) < 100:
            continue

        # write
        books_data.append(metadata)
        out_file = output_path / fname
        out_file.write_text("\n".join(texts))

print(f"wrote {len(books_data)} books")

# save metadata as csv
out_csv = output_path.parent / "blbooks.csv"
df = pd.DataFrame(books_data)
df.to_csv(str(out_csv), index=False)


raw_files = sorted(list(output_path.rglob("*.txt")))
print(f"found {len(raw_files)} files")


cleaned_path = pathlib.Path(clean_out_path_str)
if cleaned_path.exists():
    print(f"{cleaned_path} exists, continue?")
else:
    cleaned_path.mkdir(parents=True)
    

def clean_text(text_content: str) -> str:
    lines = text_content.split("\n")
    cleaned_lines = []
    for idx, line in enumerate(lines):
        # remove all-uppercase lines
        if re.sub(r'\W', '', line).isupper():
            continue
        # remove toc like lines
        if line.count('- - -') > 1 or line.count('. . . .') > 2:
            continue
        # strip 
        ## REMOVE THE WEIRD ?PAGE # TRANSCRIPTS?
        # check for parentheticals
        split = re.split(r'\((.*?)\)', line, 1)
        if len(split) > 1:
            # print(split)
            # don't do it on erroneous lines
            if len(' '.join(split[:-1])) < len(split[-1]):
                content = split[-1]
            else:
                content = line
        else:
            content = line
        
        # find first digit and try to split to remove page stuff
        split = re.split(r'\b\d{1,3}\b', line, 1)
        if len(split) > 1:
            # don't do it on erroneous lines
            if len(' '.join(split[:-1])) < len(split[-1]):
                content = split[-1]
            else:
                content = line
        else:
            content = line

        # then remove any uppercase tokens
        words = content.split()
        rm_idx = 0
        for idx, word in enumerate(words):
            if word.isupper():
                rm_idx = idx + 1
                # print(f"'{word}' is upper, idx = {rm_idx}")
            else:
                break

        words = words[rm_idx:]

        # remove transcription errors
        OCR_ERR_DROP_SYMBOLS = ['^']

        cleaned_words = []
        for word in words:
            skip: bool = False
            for symb in OCR_ERR_DROP_SYMBOLS:
                if symb in word:
                    skip = True
            if skip:
                continue
            cleaned_word = re.sub(r'[»•«■]', '', word)
            cleaned_words.append(cleaned_word)
                    
        content = ' '.join(cleaned_words)
        line = re.sub(r'[\s]+', ' ', content).strip()
        if line:
            cleaned_lines.append(line)

    # fix some case of duplicated words
    for idx, line in enumerate(cleaned_lines[:-1]):
        next_line = cleaned_lines[idx+1]
        last_word = line.split()[-1]
        next_start = next_line.split()[0]
        if last_word == next_start:
            cleaned_lines[idx] = ' '.join(line.split()[:-1])

    return '\n'.join(cleaned_lines)
        

for idx, file in enumerate(tqdm(raw_files)):
    try:
        text = file.read_text(encoding='utf-8')
        cleaned_text = clean_text(text)
        out_path = cleaned_path / file.name
        out_path.write_text(cleaned_text)
    except Exception as e:
        print(f"{idx} {file.name} - {type(e).__name__}: {str(e)}")