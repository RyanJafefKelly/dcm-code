#!/usr/bin/env python3
"""Trim a BibTeX file to entries cited by a LaTeX .aux file.

Defaults are set for the SPAR files currently exported in ~/Downloads. You can
also pass explicit paths, for example:

    python trim_bib.py main.aux master.bib -o references.bib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_AUX = Path.home() / "Downloads" / "spar_output.aux"
DEFAULT_BIB = Path.home() / "Downloads" / "spar_references.bib"
DEFAULT_OUTPUT = Path("references.bib")

CITATION_RE = re.compile(r"\\citation\{([^}]*)\}")
IGNORED_ENTRY_TYPES = {"comment", "string", "preamble"}


def parse_cited_keys(aux_path: Path) -> list[str]:
    """Return cited keys in first-citation order, deduplicated."""
    cited: list[str] = []
    seen: set[str] = set()

    for line in aux_path.read_text(encoding="utf-8-sig").splitlines():
        for match in CITATION_RE.finditer(line):
            for raw_key in match.group(1).split(","):
                key = raw_key.strip()
                if key and key not in seen:
                    seen.add(key)
                    cited.append(key)

    return cited


def is_escaped(text: str, index: int) -> bool:
    """Return True if text[index] is preceded by an odd number of backslashes."""
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def find_matching_outer_delimiter(text: str, open_index: int) -> int:
    """Return the index just after the matching outer delimiter."""
    opener = text[open_index]
    closer = "}" if opener == "{" else ")"
    depth = 0

    for index in range(open_index, len(text)):
        char = text[index]
        if char == opener and not is_escaped(text, index):
            depth += 1
        elif char == closer and not is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index + 1

    raise ValueError(f"Unclosed BibTeX block starting at byte offset {open_index}")


def iter_bib_entries(bib_text: str):
    """Yield (entry_type, key, entry_text) for BibTeX entries with keys."""
    cursor = 0
    length = len(bib_text)

    while cursor < length:
        at_index = bib_text.find("@", cursor)
        if at_index == -1:
            return

        type_start = at_index + 1
        while type_start < length and bib_text[type_start].isspace():
            type_start += 1

        type_end = type_start
        while type_end < length and (
            bib_text[type_end].isalpha() or bib_text[type_end] in "_-"
        ):
            type_end += 1

        entry_type = bib_text[type_start:type_end].lower()
        if not entry_type:
            cursor = at_index + 1
            continue

        delimiter_index = type_end
        while delimiter_index < length and bib_text[delimiter_index].isspace():
            delimiter_index += 1

        if delimiter_index >= length or bib_text[delimiter_index] not in "{(":
            cursor = at_index + 1
            continue

        block_end = find_matching_outer_delimiter(bib_text, delimiter_index)
        entry_text = bib_text[at_index:block_end]
        cursor = block_end

        if entry_type in IGNORED_ENTRY_TYPES:
            continue

        key_start = delimiter_index + 1
        while key_start < block_end and bib_text[key_start].isspace():
            key_start += 1

        comma_index = bib_text.find(",", key_start, block_end)
        if comma_index == -1:
            continue

        key = bib_text[key_start:comma_index].strip()
        if key:
            yield entry_type, key, entry_text


def parse_bib_entries(bib_path: Path) -> dict[str, str]:
    """Return a mapping from BibTeX key to full entry text."""
    bib_text = bib_path.read_text(encoding="utf-8-sig")
    entries: dict[str, str] = {}

    for _entry_type, key, entry_text in iter_bib_entries(bib_text):
        entries.setdefault(key, entry_text)

    return entries


def write_trimmed_bib(cited_keys: list[str], entries: dict[str, str], output_path: Path) -> list[str]:
    missing = [key for key in cited_keys if key not in entries]
    found_entries = [entries[key].rstrip() for key in cited_keys if key in entries]
    output_path.write_text("\n\n".join(found_entries) + ("\n" if found_entries else ""), encoding="utf-8")
    return missing


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create references.bib containing only entries cited in a LaTeX .aux file."
    )
    parser.add_argument(
        "aux_file",
        nargs="?",
        default=DEFAULT_AUX,
        type=Path,
        help=f".aux file to scan for citations (default: {DEFAULT_AUX})",
    )
    parser.add_argument(
        "bib_file",
        nargs="?",
        default=DEFAULT_BIB,
        type=Path,
        help=f"BibTeX file to trim (default: {DEFAULT_BIB})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"output BibTeX file (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    cited_keys = parse_cited_keys(args.aux_file)
    entries = parse_bib_entries(args.bib_file)
    missing = write_trimmed_bib(cited_keys, entries, args.output)

    written_count = len(cited_keys) - len(missing)
    print(f"AUX file: {args.aux_file}")
    print(f"BibTeX file: {args.bib_file}")
    print(f"Output file: {args.output}")
    print(f"Total cited keys found in .aux: {len(cited_keys)}")
    print(f"Entries written to references.bib: {written_count}")
    print("Cited keys not found in master.bib:")
    if missing:
        for key in missing:
            print(f"  - {key}")
    else:
        print("  none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
