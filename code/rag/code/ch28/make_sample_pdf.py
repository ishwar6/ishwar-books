"""Chapter 28 - build the test fixture: a realistic multi-page PDF.

Every other ch28 script parses this file. We generate it instead of shipping a
binary so you can see exactly what went in, and then watch how much of it a
parser can get back out.

    uv run python code/ch28/make_sample_pdf.py

Writes data/pdfs/lumora_report.pdf with six pages, each chosen to break a
different assumption:

  1  title page            plain text, one column
  2  two-column prose      the reading-order trap
  3  a table               the row/column-association trap
  4  a VECTOR bar chart    labels are text, the DATA is drawing operators
  5  a VECTOR tree diagram labels are text, the STRUCTURE is drawing operators
  6  a RASTER copy of the  nothing but pixels: extraction returns nothing
     same chart

The one rcParam that matters:  pdf.fonttype = 42.  matplotlib's default is 3
(Type 3 fonts), which many extractors read as mojibake because Type 3 glyphs
carry no reliable ToUnicode map. The producer decides whether your PDF is
machine-readable at all - which is the first thing to check when extraction
returns garbage.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType: extractable text
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.patches as patches             # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ragbook import DATA_DIR                     # noqa: E402

OUT = DATA_DIR / "pdfs" / "lumora_report.pdf"
CHART_PNG = DATA_DIR / "pdfs" / "chart_raster.png"

# ---------------------------------------------------------------- content ---
LEFT_COL = [
    "Lumora Robotics deployed 6,000 Atlas",
    "robots across 120 warehouses in 14",
    "countries during the 2025 financial",
    "year. The largest single site, near",
    "Rotterdam, runs 410 robots on one",
    "Beacon tenant and a single fleet",
    "gateway. Fleet uptime across all",
    "regions held at 99.94 percent.",
]
RIGHT_COL = [
    "Field service costs fell 18 percent",
    "after predictive battery scheduling",
    "shipped in Beacon 4.2. The charging",
    "scheduler now predicts runtime per",
    "robot from Compass battery health",
    "instead of a fleet-wide average,",
    "which cut charging-related idle time",
    "on every measured site.",
]

SPEC_ROWS = [                     # the table on page 3 (from the Atlas A2 spec sheet)
    ("Property", "Atlas A2", "Atlas A2 Lite"),
    ("Maximum payload", "250 kg", "120 kg"),
    ("Maximum speed", "2.0 m/s", "1.6 m/s"),
    ("Battery runtime", "8 hours", "8 hours"),
    ("Full charge time", "90 minutes", "75 minutes"),
    ("Robot weight", "145 kg", "110 kg"),
    ("Ingress protection", "IP54", "IP54"),
]

SITES = [("Pune", 210), ("Bengaluru", 95), ("Rotterdam", 410), ("Austin", 160)]

TREE = {                          # the org chart on page 5
    "Lumora Robotics": ["Robot Platform", "Fleet Platform", "Field & Support"],
    "Robot Platform": ["Navigation", "Firmware"],
    "Fleet Platform": ["Beacon", "Compass"],
    "Field & Support": ["EU Service"],
}


# ------------------------------------------------------------------ pages ---
def page_title(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))       # A4 portrait, inches
    fig.text(0.5, 0.70, "Lumora Robotics", ha="center", size=28, weight="bold")
    fig.text(0.5, 0.64, "Annual Fleet Report 2025", ha="center", size=18)
    fig.text(0.5, 0.58, "Internal - Fleet Platform", ha="center", size=11)
    pdf.savefig(fig)
    plt.close(fig)


def page_two_columns(pdf: PdfPages) -> None:
    """Two columns, written column by column - the order a real typesetter uses.

    The trap is not the write order. It is that line 1 of the left column and
    line 1 of the right column sit at the SAME y. Any extractor that sorts by
    vertical position without detecting columns welds them into one line.
    """
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "2.1  Fleet in numbers", size=14, weight="bold")
    y0, dy = 0.88, 0.028
    for i, line in enumerate(LEFT_COL):
        fig.text(0.08, y0 - i * dy, line, size=10)
    for i, line in enumerate(RIGHT_COL):
        fig.text(0.54, y0 - i * dy, line, size=10)
    fig.text(0.08, 0.05, "Lumora Robotics - Internal", size=8, color="0.45")
    fig.text(0.92, 0.05, "2", size=8, color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def page_table(pdf: PdfPages) -> None:
    """A table drawn the way PDFs really carry tables: independent text cells
    at coordinates, plus a few lines. There is no <table> anywhere in the file."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "3.1  Atlas specifications", size=14, weight="bold")
    xs, y0, dy = (0.08, 0.45, 0.70), 0.86, 0.035
    for r, row in enumerate(SPEC_ROWS):
        for x, cell in zip(xs, row):
            fig.text(x, y0 - r * dy, cell, size=10,
                     weight="bold" if r == 0 else "normal")
        line_y = y0 - r * dy - 0.010
        fig.add_artist(plt.Line2D([0.06, 0.92], [line_y, line_y],
                                  color="0.75", lw=0.6))
    fig.text(0.08, 0.05, "Lumora Robotics - Internal", size=8, color="0.45")
    fig.text(0.92, 0.05, "3", size=8, color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def _chart_axes(fig):
    ax = fig.add_axes((0.14, 0.40, 0.76, 0.42))
    ax.bar([s for s, _ in SITES], [n for _, n in SITES], color="#4C72B0")
    ax.set_title("Robots deployed per site, 2025")
    ax.set_ylabel("robots")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def page_chart_vector(pdf: PdfPages) -> None:
    """Vector chart: tick labels and the title ARE text; the bar heights are
    path-fill operators. An extractor gets the words and none of the numbers."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "4.1  Deployment by site", size=14, weight="bold")
    _chart_axes(fig)
    fig.text(0.14, 0.34, "Figure 1: robots deployed per site at end of 2025.", size=9)
    fig.text(0.92, 0.05, "4", size=8, color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def page_tree(pdf: PdfPages) -> None:
    """An org chart: boxes, connector lines, and text labels. The labels extract.
    The parent/child relationships live only in the line coordinates."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "5.1  Engineering organisation", size=14, weight="bold")
    ax = fig.add_axes((0.04, 0.34, 0.92, 0.52))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    pos = {
        "Lumora Robotics": (6.0, 5.0),
        "Robot Platform": (2.4, 3.2), "Fleet Platform": (6.0, 3.2),
        "Field & Support": (9.6, 3.2),
        "Navigation": (1.2, 1.3), "Firmware": (3.6, 1.3),
        "Beacon": (5.0, 1.3), "Compass": (7.2, 1.3),
        "EU Service": (9.6, 1.3),
    }
    for parent, children in TREE.items():
        px, py = pos[parent]
        for child in children:
            cx, cy = pos[child]
            ax.plot([px, px, cx, cx], [py - 0.35, (py + cy) / 2,
                                       (py + cy) / 2, cy + 0.35],
                    color="0.35", lw=1.0, solid_joinstyle="miter")
    for name, (x, y) in pos.items():
        ax.add_patch(patches.FancyBboxPatch((x - 1.05, y - 0.35), 2.1, 0.7,
                                            boxstyle="round,pad=0.02",
                                            fc="#EAF0F8", ec="#4C72B0", lw=1.0))
        ax.text(x, y, name, ha="center", va="center", size=9)
    fig.text(0.08, 0.30, "Figure 2: engineering reporting lines.", size=9)
    fig.text(0.92, 0.05, "5", size=8, color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def page_chart_raster(pdf: PdfPages) -> None:
    """The same chart, pasted as a screenshot. Every glyph is now a pixel."""
    src = plt.figure(figsize=(6.2, 3.4))
    ax = src.add_axes((0.14, 0.16, 0.80, 0.72))
    ax.bar([s for s, _ in SITES], [n for _, n in SITES], color="#4C72B0")
    ax.set_title("Robots deployed per site, 2025")
    ax.set_ylabel("robots")
    ax.spines[["top", "right"]].set_visible(False)
    src.savefig(CHART_PNG, dpi=150)
    plt.close(src)

    img = plt.imread(CHART_PNG)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "6.1  Deployment by site (exported)", size=14, weight="bold")
    ax = fig.add_axes((0.10, 0.42, 0.80, 0.40))
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.10, 0.37, "Figure 3: the same chart, pasted as an image.", size=9)
    fig.text(0.92, 0.05, "6", size=8, color="0.45", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        page_title(pdf)
        page_two_columns(pdf)
        page_table(pdf)
        page_chart_vector(pdf)
        page_tree(pdf)
        page_chart_raster(pdf)
    print(f"wrote {OUT.relative_to(DATA_DIR.parent)}  ({OUT.stat().st_size / 1024:.0f} KB, 6 pages)")
    print(f"wrote {CHART_PNG.relative_to(DATA_DIR.parent)}  (raster source for page 6)")


if __name__ == "__main__":
    main()
