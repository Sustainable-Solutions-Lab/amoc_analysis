"""Split a multi-page PDF into one PDF per page (for pasting individual figures).

Each output is written next to the input with ``_pageNN`` appended to the stem,
e.g. ``predicted_change_tas.pdf`` -> ``predicted_change_tas_page01.pdf``,
``..._page02.pdf``, .... The page number is zero-padded to at least two digits
(more if the document has 100+ pages).

    python ./scripts/split_pdf_into_pages.py input_pdf_file_path.pdf
"""

import argparse
import os
import sys

from pypdf import PdfReader, PdfWriter


def split_pdf(in_path):
    if not os.path.isfile(in_path):
        sys.exit(f"error: no such file: {in_path}")

    stem, ext = os.path.splitext(in_path)
    reader = PdfReader(in_path)
    n = len(reader.pages)
    width = max(2, len(str(n)))  # _page01.. (grow if 100+ pages)

    out_paths = []
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = f"{stem}_page{i:0{width}d}{ext or '.pdf'}"
        with open(out_path, "wb") as f:
            writer.write(f)
        out_paths.append(out_path)
        print(f"wrote {out_path}")
    print(f"split {n} page(s) from {in_path}")
    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_pdf", help="path to the multi-page PDF to split")
    split_pdf(parser.parse_args().input_pdf)


if __name__ == "__main__":
    main()
