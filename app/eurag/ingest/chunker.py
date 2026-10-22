"""Split the text of an EU legal act into Recital / Article / Annex chunks."""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass, field

from eurag.ingest.pdf_extract import Line

# Bump on any change to text extraction or chunking that alters the indexed chunks;
# it is part of the ingest fingerprint and triggers a re-index.
CHUNKER_VERSION = 2
MAX_CHARS = 1800
_LETTERS, _DIGITS = string.ascii_lowercase, string.digits
_UNIT_NAMES = {"article": "Article", "recital": "Recital", "annex": "Annex"}

_RECITALS_START = re.compile(r"^Whereas:$")
_ENACTING_START = re.compile(r"^HAVE ADOPTED THIS (REGULATION|DIRECTIVE|DECISION):?$")
# Consolidated regulations omit the preamble; the enacting terms start right away.
_ENACTING_FIRST = re.compile(r"^(CHAPTER I|Article 1)$")
_SIGNATURE = re.compile(r"^Done at \w+, \d")
_RECITAL = re.compile(r"^\((\d+)\)\s*(.*)$")
_CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)$")
_SECTION = re.compile(r"^SECTION (\d+)$")
_ARTICLE = re.compile(r"^Article (\d+[a-z]?)$")
_ANNEX = re.compile(r"^ANNEX ([IVXLC]+)$")
_PARAGRAPH = re.compile(r"^(\d+)\.(?:\s+|$)")


@dataclass
class ActInfo:
    act: str  # short name, e.g. "GDPR"
    title: str
    celex: str  # the act as published in the OJ
    consolidated: str = ""  # CELEX of the consolidated version the text is taken from, if any

    def url(self, unit_type: str) -> str:
        # Recitals only exist in the OJ text; articles and annexes link to the consolidated version.
        celex = self.consolidated if self.consolidated and unit_type != "recital" else self.celex
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


@dataclass
class Unit:
    unit_type: str  # article | recital | annex
    number: str
    title: str = ""
    chapter: str = ""
    page_start: int = 0
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    act: str
    act_title: str
    celex: str
    unit_type: str
    number: str
    title: str
    chapter: str
    paragraphs: str  # e.g. "1-3", empty if whole unit
    part: int
    page_start: int
    url: str
    text: str  # body text including a context header

    @property
    def label(self) -> str:
        s = f"{self.act} – {_UNIT_NAMES[self.unit_type]} {self.number}"
        if self.paragraphs:
            s += f" (para. {self.paragraphs})"
        if self.title:
            s += f": {self.title}"
        return s

    def to_payload(self) -> dict:
        return {**asdict(self), "label": self.label}


def _follows(prev: str | None, number: str) -> bool:
    """True if article `number` can follow `prev`: N -> N+1, or an inserted article N -> Na -> Nb."""
    if prev is None:
        return number == "1"
    n, suffix = int(number.rstrip(_LETTERS)), number.lstrip(_DIGITS)
    pn, psuffix = int(prev.rstrip(_LETTERS)), prev.lstrip(_DIGITS)
    if n == pn + 1:
        return not suffix
    expected = chr(ord(psuffix) + 1) if psuffix else "a"
    return n == pn and suffix == expected


def parse_units(lines: list[Line]) -> list[Unit]:
    """State machine over the document lines producing structural units."""
    units: list[Unit] = []
    state = "preamble"
    chapter = ""
    current: Unit | None = None
    last_article: str | None = None
    # "chapter" | "section" | "article" | "annex": next line is its title;
    # "title": the title may continue on the next line (wrapped titles continue in lowercase)
    pending_heading: str | None = None

    def close() -> None:
        nonlocal current
        if current is not None and (current.lines or current.title):
            units.append(current)
        current = None

    for line in lines:
        text = line.text
        if state == "preamble":
            if _RECITALS_START.match(text):
                state = "recitals"
                continue
            if not _ENACTING_FIRST.match(text):
                continue
            state = "enacting"

        if state == "recitals":
            if _ENACTING_START.match(text):
                close()
                state = "enacting"
                continue
            m = _RECITAL.match(text)
            expected = str(int(current.number) + 1) if current else "1"
            if m and m.group(1) == expected:
                close()
                current = Unit("recital", m.group(1), page_start=line.page)
                if m.group(2):
                    current.lines.append(m.group(2))
            elif current is not None:
                current.lines.append(text)
            continue

        if state == "signature":
            if m := _ANNEX.match(text):
                state = "annexes"
                current = Unit("annex", m.group(1), page_start=line.page)
                pending_heading = "annex"
            continue

        # enacting terms or annexes
        if pending_heading:
            kind, pending_heading = pending_heading, None
            if kind == "chapter":
                chapter = f"{chapter} – {text}"
                continue
            if kind == "section":
                continue
            if kind in ("article", "annex") and current is not None:
                current.title = text
                pending_heading = "title"
                continue
            if kind == "title" and current is not None and text[:1].islower():
                current.title = f"{current.title} {text}"
                pending_heading = "title"
                continue

        if _SIGNATURE.match(text):
            close()
            state = "signature"
            continue
        if m := _ANNEX.match(text):
            close()
            state = "annexes"
            current = Unit("annex", m.group(1), page_start=line.page)
            pending_heading = "annex"
            continue
        if state == "enacting":
            if m := _CHAPTER.match(text):
                close()
                chapter = f"Chapter {m.group(1)}"
                pending_heading = "chapter"
                continue
            if _SECTION.match(text):
                close()
                pending_heading = "section"
                continue
            # A line that is just "Article N" can also be a wrapped cross-reference.
            if (m := _ARTICLE.match(text)) and _follows(last_article, m.group(1)):
                close()
                last_article = m.group(1)
                current = Unit("article", m.group(1), chapter=chapter, page_start=line.page)
                pending_heading = "article"
                continue
        if current is not None:
            current.lines.append(text)

    close()
    return units


def _join(lines: list[str]) -> str:
    """Join PDF lines into prose, keeping paragraph and point markers on new lines."""
    out: list[str] = []
    for ln in lines:
        starts_block = bool(_PARAGRAPH.match(ln) or re.match(r"^\(([a-z]{1,4}|[ivx]+)\)", ln))
        if out and not starts_block:
            out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    return "\n".join(re.sub(r"\s{2,}", " ", s).strip() for s in out)


def _split_paragraphs(text: str) -> list[tuple[str, str]]:
    """Split article text into (paragraph number, text) groups."""
    groups: list[tuple[str, str]] = []
    for block in re.split(r"\n(?=\d+\.\s)", text):
        m = _PARAGRAPH.match(block)
        groups.append((m.group(1) if m else "", block))
    return groups


def _pack(groups: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Greedily pack paragraph groups into chunks of at most MAX_CHARS."""
    packed: list[tuple[list[str], list[str]]] = []
    for num, block in groups:
        size = sum(len(b) for b in packed[-1][1]) if packed else 0
        if packed and size + len(block) <= MAX_CHARS:
            packed[-1][0].append(num)
            packed[-1][1].append(block)
        else:
            packed.append(([num], [block]))
    out = []
    for nums, blocks in packed:
        nums = [n for n in nums if n]
        rng = "" if not nums else nums[0] if len(nums) == 1 else f"{nums[0]}-{nums[-1]}"
        out.append((rng, "\n".join(blocks)))
    return out


def _hard_split(text: str) -> list[str]:
    """Split an over-long block on line, then sentence boundaries."""
    if len(text) <= MAX_CHARS * 1.5:
        return [text]
    pieces: list[tuple[str, str]] = []  # (separator, text)
    for line in text.split("\n"):
        sents = re.split(r"(?<=[.;:])\s+", line) if len(line) > MAX_CHARS else [line]
        pieces += [("\n" if i == 0 else " ", s) for i, s in enumerate(sents)]
    parts, buf = [], ""
    for sep, piece in pieces:
        if buf and len(buf) + len(piece) > MAX_CHARS:
            parts.append(buf)
            buf = ""
        buf = f"{buf}{sep}{piece}" if buf else piece
    if buf:
        parts.append(buf)
    return parts


def chunk_units(units: list[Unit], info: ActInfo) -> list[Chunk]:
    chunks: list[Chunk] = []
    for u in units:
        body = _join(u.lines)
        if len(body) <= MAX_CHARS:
            pieces = [("", body)]
        elif u.unit_type == "article":
            pieces = _pack(_split_paragraphs(body))
        else:
            pieces = [("", body)]
        pieces = [(rng, sub) for rng, text in pieces for sub in _hard_split(text)]

        anchor = {"article": f"#art_{u.number}", "annex": f"#anx_{u.number}", "recital": f"#rct_{u.number}"}
        for part, (rng, text) in enumerate(pieces):
            chunk = Chunk(
                act=info.act,
                act_title=info.title,
                celex=info.celex,
                unit_type=u.unit_type,
                number=u.number,
                title=u.title,
                chapter=u.chapter,
                paragraphs=rng if len(pieces) > 1 else "",
                part=part,
                page_start=u.page_start,
                url=info.url(u.unit_type) + anchor[u.unit_type],
                text="",
            )
            header = chunk.label + (f" [{u.chapter}]" if u.chapter else "")
            chunk.text = f"{header}\n{text}"
            chunks.append(chunk)
    return chunks


def merge_recitals(units: list[Unit], recital_source: list[Unit]) -> list[Unit]:
    """Replace the recitals of `units` with those of another text of the same act.

    Used for consolidated texts, which drop the preamble: recitals come from the OJ original.
    """
    recitals = [u for u in recital_source if u.unit_type == "recital"]
    return recitals + [u for u in units if u.unit_type != "recital"]
