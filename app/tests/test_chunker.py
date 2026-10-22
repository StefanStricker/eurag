from collections import Counter
from functools import cache

import pytest
import yaml

from eurag.ingest.chunker import MAX_CHARS, ActInfo, chunk_units, parse_units
from eurag.ingest.pdf_extract import Line, extract_lines

# (file, recitals, articles, annexes) as published in the Official Journal
EXPECTED = [
    ("GDPR.pdf", 173, 99, 0),
    ("AI-Act.pdf", 180, 113, 13),
    ("Data-Act.pdf", 119, 50, 0),
    ("EUR_lex.pdf", 63, 38, 0),
    ("ePrivacy.pdf", 49, 21, 0),
]


@cache
def _units(path):
    return parse_units(extract_lines(path))


@pytest.mark.parametrize(("filename", "recitals", "articles", "annexes"), EXPECTED)
def test_structure_counts(data_dir, filename, recitals, articles, annexes):
    units = _units(data_dir / filename)
    counts = Counter(u.unit_type for u in units)
    assert counts["recital"] == recitals
    assert counts["article"] == articles
    assert counts["annex"] == annexes
    numbers = [u.number for u in units if u.unit_type == "article"]
    assert numbers == [str(i) for i in range(1, articles + 1)]


def test_gdpr_article_6(data_dir):
    art6 = next(u for u in _units(data_dir / "GDPR.pdf") if u.unit_type == "article" and u.number == "6")
    assert art6.title == "Lawfulness of processing"
    assert art6.chapter.startswith("Chapter II")
    assert "consent" in " ".join(art6.lines)


def test_eprivacy_two_column_reading_order(data_dir):
    units = {u.number: u for u in _units(data_dir / "ePrivacy.pdf") if u.unit_type == "article"}
    assert units["5"].title == "Confidentiality of the communications"
    text = " ".join(units["5"].lines)
    assert "terminal equipment" in text


def test_no_footnotes_in_text(data_dir):
    for filename, *_ in EXPECTED:
        assert not any(line.text.startswith("(1) OJ") for line in extract_lines(data_dir / filename))


def test_chunks_have_context_header_and_bounded_size(data_dir):
    meta = yaml.safe_load((data_dir / "acts.yaml").read_text())["GDPR.pdf"]
    chunks = chunk_units(_units(data_dir / "GDPR.pdf"), ActInfo(**meta))
    for c in chunks:
        assert c.text.startswith(f"GDPR – {c.unit_type.capitalize()} {c.number}")
        assert len(c.text) < MAX_CHARS * 1.6
    art6 = [c for c in chunks if c.unit_type == "article" and c.number == "6"]
    assert len(art6) > 1 and art6[0].paragraphs.startswith("1")
    assert art6[0].url.endswith("CELEX:32016R0679#art_6")


def test_parse_units_synthetic():
    lines = [
        Line(1, t)
        for t in [
            "Whereas:", "(1)", "First recital.", "(2) Second recital", "continues.",
            "HAVE ADOPTED THIS REGULATION:", "CHAPTER I", "General provisions",
            "Article 1", "Subject matter", "1.", "Text of para one.", "(a) point a;",
            "Article 2", "Scope", "Some text referring to Article 1.",
            "Done at Brussels, 1 January 2024.", "For the Council",
            "ANNEX I", "List of things", "1. A thing",
        ]
    ]
    units = parse_units(lines)
    assert [(u.unit_type, u.number) for u in units] == [
        ("recital", "1"), ("recital", "2"), ("article", "1"), ("article", "2"), ("annex", "I"),
    ]
    assert units[1].lines == ["Second recital", "continues."]
    assert units[2].title == "Subject matter" and units[2].chapter == "Chapter I – General provisions"
    assert units[4].title == "List of things"
