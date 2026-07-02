import argparse
import json
import pathlib
from pathlib import Path

import pandas as pd
from fastparquet import ParquetFile
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Process OCS Correction dataset")
parser.add_argument("--input-path", type=str, required=True, help="Input path to OCS Correction dataset")
parser.add_argument("--output-path", type=str, required=True, help="Output path for processed data")
args = parser.parse_args()


news_basepath: Path = pathlib.Path(args.input_path)
news_parquet_files = sorted(list(news_basepath.rglob("*.parquet")))
print(len(news_parquet_files))


outpath = pathlib.Path(args.output_path)
outpath.mkdir(exist_ok=True, parents=True)


for parquet_file in news_parquet_files:
    print(f"processing {parquet_file.name}")
    pf = ParquetFile(parquet_file)
    cols = ['index_id', 'id', 'date', 'edition', 'page', 'file_name', 'word_count', 'text', 'corrected_text']
    df = pf.to_pandas(cols, categories=['index_id']).sort_values(by=['id', 'edition', 'page'])
    json_data = json.loads(df.to_json(orient="records"))
    for article in tqdm(json_data):
        src_id = article['id']
        idx_id = article['index_id']
        raw_text: str = article['corrected_text']
        # basic processing
        content: list = raw_text.split("\n")
        # remove empties, all caps
        content = [l.strip() for l in content]
        content = [l for l in content if l and not l.isupper()]
        # remove short lines
        content = [l for l in content if len(l.split()) > 8]
        # deduplicate lines (dataset issue)
        lines_set: set = set()
        dedup_content = []
        for line in content:
            if line not in lines_set:
                dedup_content.append(line)
                lines_set.add(line)
        content = dedup_content

        if len(content) >= 10:
            cleaned_text = "\n".join(content)
        if len(cleaned_text.split()) > 500:
            fpath: Path = outpath / f"{src_id}-{idx_id:>06d}.txt"
            fpath.write_text(cleaned_text)

