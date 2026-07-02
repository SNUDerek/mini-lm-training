# Datasets and Data Processing Notes


# Datasets

## Gutenberg.org Books

page: [Index of /cache/epub/feeds](https://www.gutenberg.org/cache/epub/feeds/)

date accessed: June 19th

10 GB download, 30 GB extracted

1. download the `txt-files.tar.zip` and `pg_catalog.csv` files
2. extract the zip file and move the `cache` directory into same directory as the csv
3. run the processing script (will automatically filter and pre-process)

filtering rules: English; at least one author with birth/death dates; earliest author birth before 1875

## British Library Books Corpus (1800-1899)

page: [TheBritishLibrary/blbooks](https://huggingface.co/datasets/TheBritishLibrary/blbooks)

10.3 GB download, 37 GB extracted jsonl, ~17 GB filtered texts

1. use the script in scripts/download to download all ZIP files from 1800-1899 (one per decade)
2. extract all ZIP files and rename all extracted extensionless-files to *.gz and extract
3. extract all *.jsonl.gz files with bash like `find . -type f -name '*.gz' -exec gzip -d {} \;`

suggested to manually remove files: 1800_000004976

## The Nineteenth Century Serials Edition (NCSE) v2.0

page: [NCSE v2.0: A Dataset of OCR-Processed 19th Century English Newspapers](https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610)

1 GB extracted

download `NCSE_v2.zip` and extract. contains six parquet files and readme.

NOTE: i discarded all the "Publisher's Circular" data since they're mostly long lists of books

## Proceedings of the Old Bailey (OldBaileyv2)

page: [Old Bailey Corpus](https://fedora.clarin-d.uni-saarland.de/oldbailey/)

1.2 GB extracted

## Post-OCR-Correction (English, from Chronicling America)

page: [PleIAs/Post-OCR-Correction](https://huggingface.co/datasets/PleIAs/Post-OCR-Correction)

2.1 GB downloaded, 1.3 GB processed text

# Other Datasets, Currently Unused

- [dell-research-harvard/AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)

American Stories OCR-text transcription quality is very low, much worse than BL Books or NCSE. 