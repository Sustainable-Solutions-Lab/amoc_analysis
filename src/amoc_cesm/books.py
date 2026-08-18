"""Multi-page PDF assembly: one page per figure, one book per analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def write_book(
    pages: Iterable[plt.Figure],
    path: Path,
    png_dir: Path | None = None,
    dpi: int = 200,
) -> int:
    """Write figures to a multi-page PDF, returning the page count.

    ``pages`` may be a generator, so a 39-variable book never holds more than
    one figure in memory — which is also why the count is returned rather than
    taken from ``len()`` up front. Pass ``png_dir`` to also drop each page as a
    PNG for quick review.
    """
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for n, fig in enumerate(pages, start=1):
            pdf.savefig(fig, dpi=dpi)
            if png_dir is not None:
                fig.savefig(png_dir / f"{path.stem}_p{n}.png", dpi=dpi)
            plt.close(fig)
    return n  # enumerate starts at 1, so this is the page count


def is_complete(path: Path) -> bool:
    """Whether a PDF was finished rather than left half-written.

    A book takes minutes to build and the writer only closes the file at the
    very end, so an interrupted run leaves a large, plausible-looking PDF with
    no trailer. Checking for the trailer is how `--resume` tells a finished
    volume from one that needs rebuilding; size and page count cannot.
    """
    if not path.exists():
        return False
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 1024))
        return f.read().rstrip().endswith(b"%%EOF")


def chain(*page_groups: Iterable[plt.Figure]) -> Iterator[plt.Figure]:
    """Concatenate page generators while keeping them lazy."""
    for group in page_groups:
        yield from group
