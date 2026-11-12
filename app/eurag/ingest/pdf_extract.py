"""Extract clean, reading-ordered text lines from Official Journal and consolidated EUR-Lex PDFs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Running headers/footers and other OJ page furniture.
_NOISE_PATTERNS = [
    re.compile(p)
    for p in (
        r"^EN$",
        r"^Official Journal of the European (Union|Communities)$",
        r"^L \d+/\d+$",
        r"^L series$",
        r"^\d{1,2}\.\d{1,2}\.\d{4}$",
        r"^ELI: ?http\S*$",
        r"^OJ L, \d{1,2}\.\d{1,2}\.\d{4}$",
        r"^\d{1,3}/\d{1,3}$",
        r"^\d{4}/\d{3,4}$",
        r"^[IVX]+$",
        r"^\((Legislative acts|Text with EEA relevance)\)$",
        # Consolidated texts: page header, bare amendment markers, deleted provisions.
        r"^\d{4,5}[A-Z]\d{4} — [A-Z]{2} — \d{2}\.\d{2}\.\d{4} — \d{3}\.\d{3} — \d+$",
        r"^[▼►][BMCA]\d*$",
        r"^▼[BMCA]\d* _+$",
    )
]
# Inline amendment markers around changed passages: "►M1 new text ◄".
_INLINE_MARKER = re.compile(r"►[BMCA]\d*\s*|\s*◄")
# Footnotes start with "(n)" and are set in a smaller font than the body text.
_FOOTNOTE = re.compile(r"^\(\d+\)")


@dataclass(frozen=True)
class Line:
    page: int  # 1-based
    text: str


def _is_noise(text: str) -> bool:
    return any(p.match(text) for p in _NOISE_PATTERNS)


def _order_blocks(blocks: list[tuple], page_width: float) -> list[tuple]:
    """Order text blocks in reading order, handling two-column layouts.

    Full-width blocks act as separators; between two separators the left
    column is read before the right column.
    """
    mid = page_width / 2
    full, cols = [], []
    for b in blocks:
        x0, y0, x1, y1 = b[:4]
        if x0 < mid - 10 and x1 > mid + 10:
            full.append(b)
        else:
            cols.append(b)
    if not any(b[0] >= mid - 10 for b in cols):
        # Single column page: plain top-to-bottom order.
        return sorted(blocks, key=lambda b: (round(b[1]), b[0]))

    full.sort(key=lambda b: b[1])
    ordered: list[tuple] = []
    prev_y = -1.0
    for sep in [*full, None]:
        limit = sep[1] if sep is not None else float("inf")
        segment = [b for b in cols if prev_y <= b[1] < limit]
        left = sorted((b for b in segment if b[0] < mid - 10), key=lambda b: b[1])
        right = sorted((b for b in segment if b[0] >= mid - 10), key=lambda b: b[1])
        ordered.extend(left + right)
        if sep is not None:
            ordered.append(sep)
            prev_y = sep[1]
    return ordered


def _page_blocks(page: pymupdf.Page) -> list[tuple]:
    """Text blocks as (x0, y0, x1, y1, text, max_font_size)."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        sizes = [s["size"] for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
        text = "\n".join("".join(s["text"] for s in ln["spans"]) for ln in b["lines"])
        if sizes:
            blocks.append((*b["bbox"], text, max(sizes)))
    return blocks


def _body_font_size(doc: pymupdf.Document) -> float:
    counts: Counter[float] = Counter()
    for page in doc:
        for b in _page_blocks(page):
            counts[round(b[5], 1)] += len(b[4])
    return counts.most_common(1)[0][0]


def _join_hyphenated(lines: list[Line]) -> list[Line]:
    out: list[Line] = []
    for line in lines:
        if out and out[-1].text.endswith("\xad"):  # soft hyphen: always a word break
            prev = out[-1]
            out[-1] = Line(prev.page, prev.text[:-1] + line.text)
        elif out and out[-1].text.endswith("-") and line.text[:1].islower():
            # A hard hyphen at a line end is part of a compound ("non-personal", "cross-border"):
            # these PDFs break words with soft hyphens, so the hyphen is kept.
            prev = out[-1]
            out[-1] = Line(prev.page, prev.text + line.text)
        else:
            out.append(line)
    return out


def extract_lines(path: Path) -> list[Line]:
    """Return cleaned text lines of a PDF in reading order."""
    lines: list[Line] = []
    with pymupdf.open(path) as doc:
        body_size = _body_font_size(doc)
        for page_no, page in enumerate(doc, start=1):
            blocks = _page_blocks(page)
            footnote_areas: list[tuple[float, float, float]] = []  # (x0, x1, y0)
            for block in _order_blocks(blocks, page.rect.width):
                x0, y0, x1 = block[0], block[1], block[2]
                block_lines = [t.strip() for t in block[4].splitlines()]
                block_lines = [t for t in block_lines if t]
                if not block_lines:
                    continue
                # Header blocks combine several noise lines; drop if all are noise.
                if all(_is_noise(t) for t in block_lines):
                    continue
                if _FOOTNOTE.match(block_lines[0]) and block[5] < body_size - 0.5:
                    # Footnotes run to the bottom of their column.
                    footnote_areas.append((x0, x1, y0))
                if any(y0 >= fy and x0 < fx1 and x1 > fx0 for fx0, fx1, fy in footnote_areas):
                    continue
                for t in block_lines:
                    if not _is_noise(t) and (t := _INLINE_MARKER.sub("", t).strip()):
                        lines.append(Line(page_no, t))
    return _join_hyphenated(lines)
