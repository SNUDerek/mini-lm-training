import json
import pathlib
import re
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from fastparquet import ParquetFile
from tqdm import tqdm


MIN_TEXT_LINES: int = 10

MIN_TEXT_WORDS: int = 2_000

NCSE_COLUMNS = [
    'issue_id', 
    'page_number', 
    'block', 
    'column',
    'class',
    'reading_order',
    'content',
    'segment_count',
    'prompt_tokens',
    'completion_tokens',
    'total_tokens',
    'box_page_id',
    'page_id',
]


def parse_parquet_files(parquet_files: list[Path], output_dir: Path):

    for idx, parquet_file in enumerate(parquet_files): 
        print(f"processing file {parquet_file}...")
        # read parquet file to json list
        file_name = parquet_file.name.rsplit('.', 1)[0]
        pf = ParquetFile(parquet_file)
        df = pf.to_pandas(NCSE_COLUMNS, categories=['issue_id']).sort_values(by=['issue_id', 'page_number', 'block', 'column', 'reading_order'])
        json_segments = [json.loads(line) for line in df.to_json(orient='records', lines=True, force_ascii=False).split('\n') if line]

        # parse to articles
        articles = defaultdict(list)
        for segment in json_segments:
            article = segment['issue_id']
            seg_type = segment['class']
            content = segment['content']
            if not content:
                continue
            sentences = content.split('\n')
            sentences = [s for s in sentences if s]

            # simple preprocessing
            cleaned_sentences = []
            previous_start = ""
            for sentence in sentences:
                skip: bool = False
                words = sentence.split()
                # brute force weird formatting stuff
                if '.........' in sentence \
                    or '- - - - - - - - - - - ' in sentence \
                    or '.......' in sentence:
                    skip = True
                if sentence.count('...') > 2:
                    skip = True
                # skip very short sentences
                if len(words) < 8:
                    skip = True
                # skip lines where first five tokens same as previous (tables, lists etc)
                current_start = ' '.join(words[:5])
                if current_start == previous_start:
                    skip = True
                previous_start = current_start
                # skip sentences where total words are > 5 and most frequent word occurs more than half
                if len(words) > 5 and Counter(words).most_common(1)[0][1] > len(words) // 2:
                    skip = True
                # skip sections where it seems to repeat (llm issue?)
                if len(words) > 20 and len(set(words)) < len(words) // 5:
                    skip = True
                # remove uppercase
                if sentence.isupper():
                    skip = True
                # skip non-alphanumeric lines
                if not skip and not re.search(r'[A-Za-z]+', sentence):
                    skip = True

                # add
                if not skip:
                    cleaned_sentences.append(sentence)
                
            sentences = cleaned_sentences

            if seg_type == 'title' or seg_type == 'text':
                articles[article].extend(sentences)

        # write files
        article_ids = sorted(list(articles.keys()))
        for article_id in tqdm(article_ids):
            lines = articles[article_id]
            text = "\n".join(lines)
            # only write long files
            if len(lines) >= MIN_TEXT_LINES and len(text.split()) > MIN_TEXT_WORDS:
                # output_dir/file_name_article-article_id.txt
                with open(output_dir.joinpath(f"{file_name}_article-{article_id}.txt"), "w") as f:
                    f.write(text)



if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    
    args = parser.parse_args()

    ncse_parquet_dir: Path = pathlib.Path(args.input_path)
    output_dir: Path = pathlib.Path(args.output_path)
    
    # validate input dir
    if not ncse_parquet_dir.exists():
        print(f"Input directory not found: {ncse_parquet_dir}")
        sys.exit(1)
    
    # prompt user to confirm overwrite
    if output_dir.exists():
        overwrite = input(f"Output directory already exists: {output_dir}\nOverwrite? (y/n): ")
        if overwrite.lower() != "y":
            sys.exit(0)
    else:
        # create output dir
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(ncse_parquet_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        print(f"No parquet files found in {ncse_parquet_dir}")
        sys.exit(0)
    parse_parquet_files(parquet_files, output_dir)