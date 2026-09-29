#!/usr/bin/env python
"""Build the four-case pair-comparison book.

    python scripts/make_pair_compare_book.py TREFHT
    python scripts/make_pair_compare_book.py                 # every field
    python scripts/make_pair_compare_book.py --by-letter --resume
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from amoc_cesm.books import is_complete, write_book  # noqa: E402
from amoc_cesm.config import BOOK_DIR  # noqa: E402
from amoc_cesm.io import all_variables  # noqa: E402
from amoc_cesm.workflows import pair_compare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variables", nargs="*")
    parser.add_argument("--png", action="store_true", help="also write each page as a PNG")
    parser.add_argument("--name", default=None, help="output file stem")
    parser.add_argument(
        "--by-letter", action="store_true",
        help="one volume per initial letter, so an interrupted run costs one volume",
    )
    parser.add_argument("--resume", action="store_true",
                        help="skip volumes already on disk and complete")
    parser.add_argument("--color-percentile", type=float, default=None,
                        help="clip both color scales at this percentile (e.g. 98)")
    parser.add_argument("--significance-style", choices=["field", "outline"], default="field")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-fdr", action="store_true")
    args = parser.parse_args()

    variables = sorted(args.variables) if args.variables else all_variables()
    stem = args.name or f"pair_compare_book_{datetime.now():%Y-%m-%d-%H-%M-%S}"

    if args.by_letter:
        groups: dict[str, list[str]] = {}
        for v in variables:
            groups.setdefault(v[0].upper(), []).append(v)
        volumes = list(groups.items())
    else:
        volumes = [("", variables)]

    for tag, chunk in volumes:
        path = BOOK_DIR / (f"{stem}.pdf" if not tag else f"{stem}_{tag}.pdf")
        if args.resume and is_complete(path):
            print(f"skipping {path.name} (already complete)")
            continue
        pages = pair_compare.book_pages(
            chunk,
            significance_style=args.significance_style,
            alpha=args.alpha,
            false_discovery_rate=not args.no_fdr,
            color_percentile=args.color_percentile,
        )
        n = write_book(pages, path, png_dir=path.parent if args.png else None)
        print(f"wrote {path}  ({n} pages, {len(chunk)} variables)", flush=True)


if __name__ == "__main__":
    main()
