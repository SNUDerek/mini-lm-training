import argparse
import concurrent.futures
import pathlib
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


MIN_BOOK_LINES: int = 50

MIN_BOOK_WORDS: int = 2_000

END_KEYWORDS = [
    "APPENDICES", 
    "APPENDIX", 
    "FINIS", 
    "FOOTNOTE", 
    "FOOTNOTES", 
    "GLOSSARY", 
    "ILLUSTRATIONS", 
    "INDEX", 
    "NOTES", 
    "REFERENCES", 
    "THE END", 
    "Transcriber’s Note", 
    "TRANSCRIBER’S NOTE"
]


def validate_authors(authors: str, debug: bool = False) -> bool:
    # validate authors is string
    if not isinstance(authors, str):
        return False
    
    if debug:
        print("authors:", authors)
    
    # re.findall scans the whole string and returns a list of all matches
    author_years = re.findall(r'\b\d{4}-\d{4}\b', authors)
    if debug:
        print("matches:", author_years)
    
    # If the list is empty, there were no valid year patterns
    if not author_years:
        return False
        
    # check if ANY first year is before 1875
    # any() will automatically return True if the condition is met, False otherwise
    return any(int(year.split('-')[0]) < 1875 for year in author_years)


def strip_header_footer(book_contents: str) -> str:
    """strip the gutenberg text"""
    # regex to strip gutenberg intro and ending
    start_re = re.compile(
        r'^\*{3} START OF THE PROJECT GUTENBERG(?: EBOOK)? (.+?) \*{3}$',
        re.MULTILINE
    )

    end_re = re.compile(
        r'^\*{3} END OF THE PROJECT GUTENBERG(?: EBOOK)? (.+?) \*{3}$',
        re.MULTILINE
    )
    body_text: str = start_re.split(book_contents, maxsplit=1)[-1]
    body_text = end_re.split(body_text, maxsplit=1)[0].strip()
    
    return body_text


def clean_section(section: str) -> str:
    """do some basic cleaning"""
    # remove underscores (italic indicators)
    joined_text = section.replace("_", "").strip()
    # replace quotes
    joined_text.replace('``', '"').replace("''", '"')

    # remove section if encapsulated by square brackets
    if joined_text.startswith("[") and joined_text.endswith("]"):
        return None
    # normalize text--text to text -- text
    joined_text = re.sub(r'\s?(?<!-)--(?!-)\s?', ' -- ', joined_text)
    # remove section if no text
    if not re.search(r'[0-9A-Za-z]', joined_text):
        return None
    # remove end lines
    if joined_text.startswith("End of Project Gutenberg"):
        return None

    # remove square bracket footnotes
    joined_text = re.sub(r'\[[0-9]+\]', '', joined_text)

    return joined_text


def process_section(section: str) -> str | None:
    """normalize one paragraph of gutenberg text"""
    # split by newlines (since gutenberg is line-length limited) & re-join
    lines: list[str] = section.split('\n')
    joined_text: str = ' '.join(lines)
    # clean and drop empties
    cleaned_text: str | None = clean_section(joined_text)
    if cleaned_text:
        # remove multiple spaces
        cleaned_text = re.sub(r'[\s]+', ' ', cleaned_text).strip()
    return cleaned_text
    

def detect_toc_lines(section: str) -> bool:
    """detect toc dots"""
    if ". . . . ." in section or "........" in section:
        return True
    return False


def detect_chapter_terms(section: str) -> bool:
    """detect initial chapter text"""
    return section.lower().strip().startswith('chapter') \
        or section.lower().strip().startswith('chap.') \
        or section.lower().strip().startswith('chapt.')


def detect_page_number(section: str) -> bool:
    """detect line that ends in digit == page number"""

    words = section.split()
    if len(words) < 1:
        return False
    if words[-1].strip().isdigit():
        return True
    return 
    

def detect_modern_lines(section: str) -> bool:
    """cleanup lines from gutenberg editors"""

    email_re = re.compile(
        r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
    )
    if re.search(email_re, section):
        return True
    return False


def detect_roman_numerals(section: str) -> bool:
    """detect lines with initial roman numerals"""

    roman_re = re.compile(
        r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$'
    )
    words = section.split()
    if len(words) < 1:
        return False
    if re.match(roman_re, words[0]):
        return not section.strip().startswith("I ")


def heuristic_removal(sections: list[str], debug_fname: str= "", debug: bool=False) -> tuple[list[str], list[str]]:
    """remove low-content lines with some heuristics"""
    rm_idxs: set[int] = set()

    rm_keywords = ["Project Gutenberg", "http", "HTML", "E-Text"]

    break_idxs: dict = {}

    # check for bad lines
    for idx, line in enumerate(sections):

        # drop very top short lines
        if idx < 16:
            if len(line.split()) < 8:
                rm_idxs.add(idx)
                continue

        # examine top of documents and tag ToC-like lines for removal
        if idx < max(200, len(sections) // 8):
            if detect_toc_lines(line) \
                or detect_chapter_terms(line) \
                or detect_page_number(line) \
                or detect_modern_lines(line) \
                or detect_roman_numerals(line):
                rm_idxs.add(idx)
                continue

        # tag any all uppercase sections for removal
        if line.isupper():
            rm_idxs.add(idx)
            continue

        # tag any table data sections for removal (heuristic)
        if "+----" in line or '┌' in line or line.count('|') > 2 or line.count('│') > 1:
            rm_idxs.add(idx)
            continue

        # remove gutenberg keyword lines
        for keyword in rm_keywords:
            if keyword in line:
                rm_idxs.add(idx)
                break
            if idx in rm_idxs:
                continue

        # remove lines/tuples entirely within square brackets
        if line.startswith('[') and line.endswith(']'):
            rm_idxs.add(idx)
        elif idx < len(sections) - 1 and line.startswith('[') \
            and '[' not in sections[idx+1] and sections[idx+1].endswith(']'):
            rm_idxs.add(idx)

        # [FINAL] 'ending' keyword check, skip first part of book
        if idx < len(sections) // 3:
            continue
        for keyword in END_KEYWORDS:
            if line.startswith(keyword) and keyword not in break_idxs:
                break_idxs.update({keyword: idx})
                break
    if debug:
        print(debug_fname, break_idxs)

    if break_idxs.values():

        break_from_idx = min(break_idxs.values())
        for idx in range(break_from_idx, len(sections)):
            rm_idxs.add(idx)

    target, rejected = [], []
    for idx, section in enumerate(sections):
        if idx in rm_idxs:
            rejected.append(section)
        else:
            target.append(section)

    return target, rejected


def process_book(book_path: Path, debug: bool=False) -> tuple[str, str]:

    # open one book for testing
    book_contents: str = book_path.read_text()

    # strip gutenberg stuff
    body_text: str = strip_header_footer(book_contents)

    # split into text sections
    text_sections: list[str] = [section for section in body_text.split("\n\n") if section]

    # normalize sections
    clean_sections: list[str] = [process_section(section) for section in text_sections]
    clean_sections = [section for section in clean_sections if section]

    # toc removal
    clean_sections, rejected_sections = heuristic_removal(clean_sections, debug_fname=book_path.name, debug=debug)

    return '\n'.join(clean_sections), '\n'.join(rejected_sections)


def process_and_save_book(book: Path, output_dir: Path, save_dropped: bool):
    try:
        normed_text, dropped_text = process_book(book)
        # skip too-short
        if len(normed_text.split('\n')) < MIN_BOOK_LINES or len(normed_text.split()) < MIN_BOOK_WORDS:
            return
        output_path: Path = output_dir / book.name
        output_path.write_text(normed_text, encoding='utf-8')
        if save_dropped:
            reject_path: Path = output_dir / book.name.replace(".txt", ".dropped.txt")
            reject_path.write_text(dropped_text, encoding='utf-8')
    except Exception as e:
        print(f"exception for book {book.name}: {type(e).__name__} - {str(e)}")


def main(args):

    gutenberg_path: Path = pathlib.Path(args.input_path)
    output_path: Path = pathlib.Path(args.output_path)

    # validate gutenberg path exists, and contains 'cache' directory and 'pg_catalog.csv'
    if not gutenberg_path.exists():
        raise ValueError(f"input_path {gutenberg_path} does not exist")
    if not (gutenberg_path / "cache").exists():
        raise ValueError(f"input_path {gutenberg_path} does not contain 'cache' directory")
    if not (gutenberg_path / "pg_catalog.csv").exists():
        raise ValueError(f"input_path {gutenberg_path} does not contain 'pg_catalog.csv'")

    # if output path exists, ask user. if ok, then clear contents first
    if output_path.exists():
        response = input(f"output_path {output_path} already exists, clear contents and continue? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)
        # clear contents
        for f in output_path.iterdir():
            f.unlink()
    output_path.mkdir(exist_ok=True, parents=True)

    # read catalog
    df = pd.read_csv(pathlib.Path(gutenberg_path, "pg_catalog.csv"))
    total_books = len(df)
    
    # filter where Type == 'Text'
    df = df[df['Type'] == 'Text']
    df = df[df['Language'] == 'en']
    df = df[df['Authors'].apply(validate_authors)]
    target_ids: set[int] = set(df['Text#'].to_list())
    print(f"targeting {len(target_ids)} of {total_books} books")

    # get list of all books, in gutenberg_path + /cache/epub + <book_id> + /<book_id>.txt
    book_dirs: list[Path] = list(gutenberg_path.glob("cache/epub/*"))
    books: list[Path] = sorted([list(book.glob("*.txt"))[0] for book in book_dirs])
    print(f"found {len(books)} total book files")

    # filter list by whether book_id path is in target ids
    target_book_paths: list[Path] = [book for book in books if int(book.parent.name) in target_ids]
    print(f"total target books: {len(target_book_paths)}")


    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_and_save_book, book, output_path, args.save_dropped) for book in target_book_paths]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(target_book_paths)):
                pass
    else:
        for book in tqdm(target_book_paths):
            process_and_save_book(book, output_path, args.save_dropped)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Gutenberg Corpus Preprocessor",
        description="process Gutenberg Corpus",
        epilog="Run with --help to see parameters",
    )
    parser.add_argument("--input-path", type=str, required=True, help="path to gutenberg dir with 'cache' and 'pg_catalog.csv")
    parser.add_argument("--output-path", type=str, required=True, help="path to output")
    parser.add_argument("--workers", type=int, required=False, default=1, help="number of parallel workers. default: 1")
    parser.add_argument("--save-dropped", action='store_true', help="save dropped lines as *.dropped.txt. default: False")
    args = parser.parse_args()

    main(args)
