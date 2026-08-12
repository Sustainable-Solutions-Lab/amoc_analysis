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
) -> Path:
    """Write figures to a multi-page PDF, closing each as it is consumed.

    ``pages`` may be a generator, so a 39-variable book never holds more than
    one figure in memory. Pass ``png_dir`` to also drop each page as a PNG for
    quick review.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for n, fig in enumerate(pages, start=1):
            pdf.savefig(fig, dpi=dpi)
            if png_dir is not None:
                fig.savefig(png_dir / f"{path.stem}_p{n}.png", dpi=dpi)
            plt.close(fig)
    return path


def chain(*page_groups: Iterable[plt.Figure]) -> Iterator[plt.Figure]:
    """Concatenate page generators while keeping them lazy."""
    for group in page_groups:
        yield from group
