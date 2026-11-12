import re
from collections import Counter
from functools import cache

import pytest
import yaml

from eurag.ingest.chunker import MAX_CHARS, chunk_units, parse_units
from eurag.ingest.pdf_extract import Line, _join_hyphenated, extract_lines
from eurag.ingest.run import load_act

# (file, recitals, articles, annexes, inserted articles) of the current texts (EUR-Lex consolidated
# versions where they exist); articles = last article number, not counting inserted "Na" articles
EXPECTED = [
    ("GDPR.pdf", 173, 99, 0, []),
    ("AI-Act.pdf", 180, 113, 14, ["4a", "60a", "75a", "75b", "75c", "75d"]),
    ("Data-Act.pdf", 119, 50, 0, []),
    ("DGA.pdf", 63, 38, 0, []),
    ("ePrivacy.pdf", 49, 21, 0, ["14a", "15a"]),
]


@cache
def _act(data_dir, filename):
    meta = yaml.safe_load((data_dir / "acts.yaml").read_text())[filename]
    return load_act(data_dir, filename, meta)


def _units(data_dir, filename):
    return _act(data_dir, filename)[1]


@pytest.mark.parametrize(("filename", "recitals", "articles", "annexes", "inserted"), EXPECTED)
def test_structure_counts(data_dir, filename, recitals, articles, annexes, inserted):
    units = _units(data_dir, filename)
    counts = Counter(u.unit_type for u in units)
    assert counts["recital"] == recitals
    assert counts["annex"] == annexes
    numbers = [u.number for u in units if u.unit_type == "article"]
    assert [n for n in numbers if not n.isdigit()] == inserted
    assert [n for n in numbers if n.isdigit()] == [str(i) for i in range(1, articles + 1)]
    assert numbers == sorted(numbers, key=lambda n: (int(re.match(r"\d+", n)[0]), n))


@pytest.mark.parametrize("filename", [e[0] for e in EXPECTED])
def test_no_amendment_markers(data_dir, filename):
    for u in _units(data_dir, filename):
        text = " ".join([u.title, *u.lines])
        assert not re.search(r"[►▼◄\xad]|— EN —", text), (u.unit_type, u.number)


def test_gdpr_article_6(data_dir):
    art6 = next(u for u in _units(data_dir, "GDPR.pdf") if u.unit_type == "article" and u.number == "6")
    assert art6.title == "Lawfulness of processing"
    assert art6.chapter.startswith("Chapter II")
    assert "consent" in " ".join(art6.lines)


def test_wrapped_titles_are_joined(data_dir):
    arts = {u.number: u for u in _units(data_dir, "GDPR.pdf") if u.unit_type == "article"}
    assert arts["10"].title == "Processing of personal data relating to criminal convictions and offences"
    assert arts["10"].lines[0][:1].isupper()


def test_eprivacy_two_column_reading_order(data_dir):
    units = {u.number: u for u in _units(data_dir, "ePrivacy.pdf") if u.unit_type == "article"}
    assert units["5"].title == "Confidentiality of the communications"
    text = " ".join(units["5"].lines)
    assert "terminal equipment" in text


def test_eprivacy_cookie_consent_amendment(data_dir):
    # Art. 5(3) as amended by Directive 2009/136/EC: consent, not just a right to refuse.
    art5 = next(u for u in _units(data_dir, "ePrivacy.pdf") if u.unit_type == "article" and u.number == "5")
    assert "has given his or her consent" in " ".join(art5.lines)


def test_wrapped_article_reference_is_not_a_heading(data_dir):
    # AI Act Art. 3(37) wraps so that a line reads just "Article 10"; definitions must stay in Art. 3.
    art3 = next(u for u in _units(data_dir, "AI-Act.pdf") if u.unit_type == "article" and u.number == "3")
    assert "(68)" in " ".join(art3.lines)


def test_no_footnotes_in_text(data_dir):
    for filename in [e[0] for e in EXPECTED] + ["GDPR-OJ.pdf", "AI-Act-OJ.pdf"]:
        assert not any(line.text.startswith("(1) OJ") for line in extract_lines(data_dir / filename))


def test_chunks_have_context_header_and_bounded_size(data_dir):
    info, units = _act(data_dir, "GDPR.pdf")
    chunks = chunk_units(units, info)
    for c in chunks:
        assert c.text.startswith(f"GDPR – {c.unit_type.capitalize()} {c.number}")
        assert len(c.text) < MAX_CHARS * 1.6
    art6 = [c for c in chunks if c.unit_type == "article" and c.number == "6"]
    assert len(art6) > 1 and art6[0].paragraphs.startswith("1")
    assert art6[0].url.endswith("CELEX:02016R0679-20160504#art_6")
    rct1 = next(c for c in chunks if c.unit_type == "recital" and c.number == "1")
    assert rct1.url.endswith("CELEX:32016R0679#rct_1")


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


def test_parse_units_consolidated_synthetic():
    # No preamble, an inserted article, and a cross-reference wrapped onto its own line.
    lines = [
        Line(1, t)
        for t in [
            "REGULATION (EU) 2099/1", "CHAPTER I", "General provisions",
            "Article 1", "Subject matter", "Text referring to", "Article 3", "of another act.",
            "Article 1a", "Inserted", "Text.",
            "Article 2", "Scope", "Text.",
        ]
    ]
    units = parse_units(lines)
    assert [u.number for u in units] == ["1", "1a", "2"]
    assert units[0].lines == ["Text referring to", "Article 3", "of another act."]


def test_join_hyphenated_keeps_compound_hyphens():
    lines = [Line(1, "non-"), Line(1, "personal data and per\xad"), Line(1, "sonal data, see"), Line(1, "Article-"), Line(1, "Article 5")]
    assert [ln.text for ln in _join_hyphenated(lines)] == ["non-personal data and personal data, see", "Article-", "Article 5"]


def test_no_merged_compounds(data_dir):
    text = " ".join(" ".join(u.lines) for u in _units(data_dir, "DGA.pdf"))
    assert "non-personal" in text
    assert not re.search(r"\b(nonpersonal|crossborder|thirdcountry)\b", text)
