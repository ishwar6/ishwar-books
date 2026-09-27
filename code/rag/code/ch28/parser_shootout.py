"""Chapter 28 - what two parsers actually get back out of the same PDF.

    uv run python code/ch28/make_sample_pdf.py      # build the fixture first
    uv run python code/ch28/parser_shootout.py

Compares:
  pypdf     - plain text in content-stream order. No coordinates.
  pymupdf   - blocks/lines/spans WITH bounding boxes, images, drawing operators.

and then shows the three things every naive pipeline gets wrong:
  1. reading order in multi-column layouts (and a column-aware fix, ~25 lines)
  2. how little a figure contributes to a text index
  3. that a rasterised figure contributes exactly nothing
"""
from __future__ import annotations

import time

import pymupdf
import pypdf

from ragbook import DATA_DIR

PDF = DATA_DIR / "pdfs" / "lumora_report.pdf"


# ------------------------------------------------------- reading order ------
def spans_with_boxes(page) -> list[tuple[float, float, float, str]]:
    """(y_top, x_left, x_centre, text) for every text span on the page."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:                       # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0, y0, x1, _ = span["bbox"]
                if span["text"].strip():
                    out.append((y0, x0, (x0 + x1) / 2, span["text"].strip()))
    return out


def naive_reading_order(spans) -> list[str]:
    """Sort top-to-bottom, then left-to-right. This is what most quick scripts do
    and it is wrong on any two-column page: two columns share the same y."""
    return [t for _, _, _, t in sorted(spans, key=lambda s: (round(s[0], 1), s[1]))]


def find_gutter(spans, page_width: float) -> float | None:
    """Find the column gutter with a projection profile.

    Paint every span's horizontal extent onto the x axis, merge the painted
    intervals, and look at the gaps. A two-column page has one wide unpainted
    band near the middle: that is the gutter. Full-width spans (headings,
    footers) would bridge it, so they are excluded from the profile - exactly
    what a layout parser does before it decides on columns.

    Returns the x to split at, or None if the page is single-column.
    """
    intervals = sorted((sp[1], 2 * sp[2] - sp[1]) for sp in spans)   # (x0, x1)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if x1 - x0 > 0.55 * page_width:              # spans the whole width: ignore
            continue
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    best, best_w = None, 0.04 * page_width           # a gutter is at least 4% wide
    for (_, end), (start, _) in zip(merged, merged[1:]):
        centre = (end + start) / 2
        if start - end > best_w and 0.25 * page_width < centre < 0.75 * page_width:
            best, best_w = centre, start - end
    return best


def column_aware_reading_order(spans, page_width: float) -> list[str]:
    """Assign each span to a column, then read each column top-to-bottom."""
    split = find_gutter(spans, page_width)
    if split is None:
        return naive_reading_order(spans)
    ordered: list[str] = []
    for lo, hi in ((0.0, split), (split, page_width)):
        col = [s for s in spans if lo <= s[2] < hi]
        ordered += [t for _, _, _, t in sorted(col, key=lambda s: (round(s[0], 1), s[1]))]
    return ordered


# ------------------------------------------------------------------ main ----
def main() -> None:
    if not PDF.exists():
        raise SystemExit("run: uv run python code/ch28/make_sample_pdf.py")

    t0 = time.perf_counter()
    reader = pypdf.PdfReader(PDF)
    pypdf_pages = [p.extract_text() for p in reader.pages]
    t_pypdf = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc = pymupdf.open(PDF)
    mupdf_pages = [p.get_text() for p in doc]
    t_mupdf = time.perf_counter() - t0

    chars_pypdf = sum(len(t) for t in pypdf_pages)
    chars_mupdf = sum(len(t) for t in mupdf_pages)
    print(f"{'parser':10} {'ms':>7} {'chars':>7}  gives you")
    print(f"{'pypdf':10} {t_pypdf*1000:7.0f} {chars_pypdf:7} "
          f" text in stream order, no coordinates")
    print(f"{'pymupdf':10} {t_mupdf*1000:7.0f} {chars_mupdf:7} "
          f" blocks/spans + bbox, images, drawing ops")

    # ---- 1. the reading-order trap -----------------------------------------
    page = doc[1]
    spans = spans_with_boxes(page)
    print("\n=== page 2 (two columns) ===")
    print("naive y-then-x sort - the two columns are welded together:")
    for line in naive_reading_order(spans)[1:6]:
        print(f"   {line}")
    print("column-aware order - left column first, as written:")
    for line in column_aware_reading_order(spans, page.rect.width)[1:6]:
        print(f"   {line}")
    print("pypdf (stream order) happens to be right here, but it has no bbox,")
    print("so when a producer DOES interleave, pypdf cannot repair it.")

    # ---- 2. what each page contributes to a text index ----------------------
    print("\n=== what the indexer receives, per page ===")
    print(f"{'page':>4} {'kind':22} {'chars':>6} {'draw ops':>9} {'imgs':>5}")
    kinds = ["title", "two-column prose", "table", "vector chart",
             "vector tree diagram", "raster chart"]
    for i, kind in enumerate(kinds):
        p = doc[i]
        print(f"{i+1:>4} {kind:22} {len(mupdf_pages[i]):6} "
              f"{len(p.get_drawings()):9} {len(p.get_images(full=True)):5}")

    print("\npage 4 (vector chart) text - axis vocabulary, no data:")
    print("   " + " | ".join(mupdf_pages[3].split("\n")[:14]))
    print("\npage 5 (tree diagram) text - every node, not one edge:")
    print("   " + " | ".join(t for t in mupdf_pages[4].split("\n") if t)[:200])
    print("\npage 6 (same chart, rasterised) text - the caption and nothing else:")
    print("   " + " | ".join(t for t in mupdf_pages[5].split("\n") if t))

    # ---- 3. pulling the figures out for a vision model ---------------------
    out_dir = DATA_DIR / "pdfs"
    for page_no in (4, 5):
        pix = doc[page_no - 1].get_pixmap(dpi=110)
        path = out_dir / f"page{page_no}.png"
        pix.save(path)
        print(f"\nrendered page {page_no} -> {path.relative_to(DATA_DIR.parent)} "
              f"({pix.width}x{pix.height}px) - this is what a VLM reads (28.5)")
    doc.close()


if __name__ == "__main__":
    main()
