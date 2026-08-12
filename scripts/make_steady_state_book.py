#!/usr/bin/env python
"""Build the quasi-steady-state PDF book.

    python scripts/make_steady_state_book.py RHREFHT
    python scripts/make_steady_state_book.py TREFMXAV PRECT --png
    python scripts/make_steady_state_book.py            # every available field
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amoc_cesm.books import write_book  # noqa: E402
from amoc_cesm.config import OUTPUT_DIR  # noqa: E402
from amoc_cesm.io import all_variables  # noqa: E402
from amoc_cesm.workflows import steady_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variables", nargs="*",
        help="variable names to include; default is every field any case archives",
    )
    parser.add_argument("--png", action="store_true", help="also write each page as a PNG")
    parser.add_argument(
        "--significance-style", choices=["field", "outline"], default="field",
        help="'outline' traces significant regions; 'field' contours the field inside them",
    )
    parser.add_argument(
        "--name", default=None,
        help="output file stem; default is a timestamped steady_state_book_<when>",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-fdr", action="store_true", help="skip Benjamini-Hochberg control")
    args = parser.parse_args()

    variables = sorted(args.variables) if args.variables else all_variables()

    # Timestamped by default so successive runs accumulate rather than
    # overwrite — a book is a record of what the data looked like when it was
    # built, and cases are still arriving.
    stem = args.name or f"steady_state_book_{datetime.now():%Y-%m-%d-%H-%M-%S}"
    path = OUTPUT_DIR / "books" / f"{stem}.pdf"

    pages = steady_state.book_pages(
        variables,
        significance_style=args.significance_style,
        alpha=args.alpha,
        false_discovery_rate=not args.no_fdr,
    )
    n_pages = write_book(pages, path, png_dir=path.parent if args.png else None)
    print(f"wrote {path}  ({n_pages} pages)")
    print(f"  {len(variables)} variables: {', '.join(variables)}")


if __name__ == "__main__":
    main()
