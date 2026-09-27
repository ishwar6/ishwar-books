"""Tiny ASCII plotting helpers shared by the Chapter 22 scripts.

A table tells you the numbers; a curve tells you the *shape*, and the shape is what
you have to remember in an interview ("recall saturates, latency keeps climbing").
"""
from __future__ import annotations


def ascii_curve(
    xs: list, ys: list[float], height: int = 9,
    ylabel: str = "", ymin: float | None = None, ymax: float | None = None,
) -> str:
    """One series plotted against categorical x values, each in its own labelled column."""
    lo = min(ys) if ymin is None else ymin
    hi = max(ys) if ymax is None else ymax
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    cell = max(len(str(x)) for x in xs) + 2          # one column per x value
    rows = []
    for r in range(height, 0, -1):
        top = lo + (hi - lo) * r / height
        bot = lo + (hi - lo) * (r - 1) / height
        line = "".join(
            ("*" if (bot <= y <= top or (r == 1 and y < bot)) else " ").center(cell)
            for y in ys
        )
        rows.append(f"{top:8.3f} |{line}")
    axis = "         +" + "-" * (cell * len(xs))
    labels = "          " + "".join(str(x).center(cell) for x in xs)
    return f"{ylabel}\n" + "\n".join(rows) + f"\n{axis}\n{labels}"


def ascii_xy(xs: list[float], ys: list[float], width: int = 52, height: int = 11,
             xlabel: str = "x", ylabel: str = "y") -> str:
    """Scatter one series in real x/y space - for recall-vs-latency style trade-off curves."""
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xhi = xhi + 1e-12 if xhi == xlo else xhi
    yhi = yhi + 1e-12 if yhi == ylo else yhi
    grid = [[" "] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        c = int((x - xlo) / (xhi - xlo) * (width - 1))
        r = height - 1 - int((y - ylo) / (yhi - ylo) * (height - 1))
        grid[r][c] = "*"
    out = [f"{ylabel}"]
    for r, row in enumerate(grid):
        val = yhi - (yhi - ylo) * r / (height - 1)
        out.append(f"{val:7.3f} |" + "".join(row))
    out.append("        +" + "-" * width)
    out.append(f"        {xlo:<{width // 2}.1f}{xhi:>{width // 2}.1f}  {xlabel}")
    return "\n".join(out)


def bar(value: float, vmax: float, width: int = 28, ch: str = "#") -> str:
    n = 0 if vmax <= 0 else int(round(value / vmax * width))
    return ch * n
