from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tqdm import tqdm


# Tags whose *own contents* are metadata, not transcript text.
# Critical: their .tail must still be preserved.
DROP_TAGS = {
    "interp",
    "join",
    "xptr",
    "ptr",
    "ref",
    "link",
}

# Block-ish tags that should become paragraph boundaries when extracting whole text.
BLOCK_TAGS = {
    "p",
    "head",
    "ab",
    "sp",
    "stage",
    "div0",
    "div1",
    "div2",
    "div3",
}


def local_name(tag: str) -> str:
    """Return the local XML tag name without namespace, if any."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def iter_visible_text(elem: ET.Element) -> Iterator[str]:
    """
    Yield visible transcript text from an XML element.

    - Keeps elem.text for normal/content-bearing elements.
    - Recurses into text-bearing children such as <rs>, <persName>, <placeName>.
    - Drops metadata children such as <interp>, <join>, <xptr>.
    - Preserves child.tail in all cases; this is essential for Old Bailey XML.
    """
    if elem.text:
        yield elem.text

    for child in elem:
        name = local_name(child.tag)

        if name in DROP_TAGS:
            # Drop metadata tag contents, but keep prose immediately after the tag.
            if child.tail:
                yield child.tail
            continue

        yield from iter_visible_text(child)

        if child.tail:
            yield child.tail


def normalize_inline_whitespace(text: str) -> str:
    """Collapse XML indentation/newlines while preserving ordinary word spacing."""
    text = re.sub(r"\s+", " ", text).strip()

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove extra space after opening punctuation and before closing punctuation.
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)

    # Old Bailey has many separated currency abbreviations. Keep these conservative.
    text = re.sub(r"\b([0-9]+)\s+l\s+\.\b", r"\1 l.", text)
    text = re.sub(r"\b([0-9]+)\s+s\s+\.\b", r"\1 s.", text)
    text = re.sub(r"\b([0-9]+)\s+d\s+\.\b", r"\1 d.", text)

    # Clean a few tag-boundary artifacts without modernizing spelling.
    text = re.sub(r"\s+;", ";", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)

    return text


def paragraph_text(elem: ET.Element) -> str:
    return normalize_inline_whitespace("".join(iter_visible_text(elem)))


def find_body(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if local_name(elem.tag) == "body":
            return elem
    return root


def selected_paragraphs(root: ET.Element, trials_only: bool = False) -> list[str]:
    """
    Extract paragraph-level text.

    If trials_only=True, only paragraphs under <div1 type="trialAccount"> are used.
    Otherwise, all <p> elements under <body> are used, including front/back matter.
    """
    body = find_body(root)
    paragraphs: list[str] = []

    if trials_only:
        containers = [
            elem
            for elem in body.iter()
            if local_name(elem.tag) == "div1" and elem.attrib.get("type") == "trialAccount"
        ]
    else:
        containers = [body]

    for container in containers:
        for elem in container.iter():
            if local_name(elem.tag) != "p":
                continue
            text = paragraph_text(elem)
            if text:
                paragraphs.append(text)

    return paragraphs


def convert_xml_file(input_path: Path, trials_only: bool = False) -> str:
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(input_path, parser=parser)
    root = tree.getroot()

    paragraphs = selected_paragraphs(root, trials_only=trials_only)
    output_text = "\n\n".join(paragraphs).strip() + "\n"

    return output_text



if __name__ == "__main__":
    
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Process Old Bailey XML files for text extraction."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the directory containing Old Bailey XML files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the directory where processed text files will be saved.",
    )
    args = parser.parse_args()

    # verify input path exists
    old_bailey_dir: Path = Path(args.input_path)
    if not old_bailey_dir.exists():
        print(f"Error: Input path {old_bailey_dir} does not exist.")
        sys.exit()

    # create output directory if it doesn't exist (ask user)
    output_dir: Path = Path(args.output_path)
    if output_dir.exists():
        if input(f"Directory {output_dir} already exists. Overwrite? (y/n): ") != "y":
            sys.exit()
        else:
            output_dir.rmdir()
    output_dir.mkdir(exist_ok=True, parents=True)

    old_bailey_xml_files = sorted(list(pathlib.Path(old_bailey_dir).rglob("*.xml")))
    print("number of files:", len(old_bailey_xml_files))
    
    for file in tqdm(old_bailey_xml_files):
        parsed = convert_xml_file(file, trials_only=True)
        parsed = re.sub(r'[\n]+', '\n', parsed)
        output_file = Path(output_dir, file.name.replace(".xml", ".txt"))
        # skip the POS tag files
        if output_file.name.startswith("OBC2POS"):
            continue
        output_file.write_text(parsed)