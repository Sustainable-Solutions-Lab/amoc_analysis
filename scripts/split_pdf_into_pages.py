"""Split a multi-page PDF into one file per page (for pasting individual figures).

Each output is written next to the input with ``_pageNN`` appended to the stem,
e.g. ``predicted_change_tas.pdf`` -> ``predicted_change_tas_page01.pdf``,
``..._page02.pdf``, .... The page number is zero-padded to at least two digits
(more if the document has 100+ pages).

By default each page is written as a one-page PDF. With ``--png`` the pages are
instead rasterized to high-resolution PNGs (``--dpi``, default 300) -- lighter for
a LaTeX engine to load than vector PDFs.

    python ./scripts/split_pdf_into_pages.py input_pdf_file_path.pdf
    python ./scripts/split_pdf_into_pages.py input_pdf_file_path.pdf --png
    python ./scripts/split_pdf_into_pages.py input_pdf_file_path.pdf --png --dpi 450
"""

import argparse
import os
import sys

import fitz  # PyMuPDF, for PDF -> PNG rasterization
from pypdf import PdfReader, PdfWriter


def split_pdf(in_path, png=False, dpi=300):
    if not os.path.isfile(in_path):
        sys.exit(f"error: no such file: {in_path}")

    stem = os.path.splitext(in_path)[0]
    out_paths = []

    if png:
        doc = fitz.open(in_path)
        n = doc.page_count
        width = max(2, len(str(n)))
        for i, page in enumerate(doc, start=1):
            out_path = f"{stem}_page{i:0{width}d}.png"
            page.get_pixmap(dpi=dpi).save(out_path)
            out_paths.append(out_path)
            print(f"wrote {out_path}")
        print(f"split {n} page(s) from {in_path} to PNG at {dpi} dpi")
    else:
        reader = PdfReader(in_path)
        n = len(reader.pages)
        width = max(2, len(str(n)))
        for i, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            out_path = f"{stem}_page{i:0{width}d}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)
            out_paths.append(out_path)
            print(f"wrote {out_path}")
        print(f"split {n} page(s) from {in_path}")
    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_pdf", help="path to the multi-page PDF to split")
    parser.add_argument("--png", action="store_true",
                        help="rasterize pages to PNG instead of writing one-page PDFs")
    parser.add_argument("--dpi", type=int, default=300,
                        help="resolution for --png output (default 300)")
    args = parser.parse_args()
    split_pdf(args.input_pdf, png=args.png, dpi=args.dpi)


if __name__ == "__main__":
    main()
