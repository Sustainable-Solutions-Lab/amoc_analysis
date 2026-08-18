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

from amoc_cesm.books import is_complete, write_book  # noqa: E402
from amoc_cesm.config import BOOK_DIR  # noqa: E402
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
    parser.add_argument(
        "--volume-size", type=int, default=None, metavar="N",
        help="split into volumes of N variables each, written one at a time; "
             "an interrupted run then costs one volume rather than the whole book",
    )
    parser.add_argument(
        "--by-letter", action="store_true",
        help="one volume per initial letter (C, F, L, ...), named for the letter "
             "rather than numbered, so a field is found by its own name",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip volumes already on disk and complete (trailer present)",
    )
    parser.add_argument(
        "--color-percentile", type=float, default=None,
        help="clip the shared difference color scale at this percentile of |difference| "
             "(e.g. 98); default spans the full range so nothing is clipped",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-fdr", action="store_true", help="skip Benjamini-Hochberg control")
    args = parser.parse_args()

    variables = sorted(args.variables) if args.variables else all_variables()

    # Timestamped by default so successive runs accumulate rather than
    # overwrite — a book is a record of what the data looked like when it was
    # built, and cases are still arriving.
    stem = args.name or f"steady_state_book_{datetime.now():%Y-%m-%d-%H-%M-%S}"

    # Volumes are written one at a time and closed before the next begins, so an
    # interrupted run loses only the volume in flight; --resume picks up there.
    if args.by_letter:
        # dict preserves insertion order, and `variables` is already sorted, so
        # the letters come out alphabetical without a second sort.
        groups: dict[str, list[str]] = {}
        for v in variables:
            groups.setdefault(v[0].upper(), []).append(v)
        volumes = [(letter, chunk) for letter, chunk in groups.items()]
    else:
        size = args.volume_size or len(variables)
        volumes = [(f"{i // size + 1:02d}", variables[i:i + size])
                   for i in range(0, len(variables), size)]
    total = len(volumes)

    for n, (tag, chunk) in enumerate(volumes, start=1):
        suffix = tag if args.by_letter else f"vol{tag}"
        path = BOOK_DIR / (f"{stem}.pdf" if total == 1 else f"{stem}_{suffix}.pdf")
        if args.resume and is_complete(path):
            print(f"skipping {path.name} (already complete)")
            continue

        label = (f"volume {n} of {total} — {tag}" if args.by_letter
                 else f"volume {n} of {total}")
        pages = steady_state.book_pages(
            chunk,
            volume=None if total == 1 else label,
            significance_style=args.significance_style,
            alpha=args.alpha,
            false_discovery_rate=not args.no_fdr,
            color_percentile=args.color_percentile,
        )
        n_pages = write_book(pages, path, png_dir=path.parent if args.png else None)
        print(f"wrote {path}  ({n_pages} pages, {len(chunk)} variables: {', '.join(chunk)})",
              flush=True)


if __name__ == "__main__":
    main()
