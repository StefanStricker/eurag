from eurag.config import Settings
from eurag.generation import SYSTEM_PROMPT, build_citations, build_messages, clean_citations, fit_context
from eurag.retrieval import Hit, covers_paragraph, explicit_refs, mentioned_acts


def _hit(n: str, size: int = 0) -> Hit:
    return Hit(
        0.5,
        {
            "label": f"GDPR – Article {n}", "act": "GDPR", "act_title": "GDPR", "unit_type": "article",
            "number": n, "title": "", "chapter": "", "paragraphs": "", "page_start": 1,
            "url": "https://example", "text": f"GDPR – Article {n}\nBody {n}" + "x" * size, "celex": "x", "part": 0,
        },
    )


def test_clean_citations_drops_invalid_refs():
    answer, cited = clean_citations("Consent is a basis [1][3]. Also [7] and [2].", n_sources=3)
    assert answer == "Consent is a basis [1][3]. Also  and [2]."
    assert cited == [1, 2, 3]


def test_build_citations_only_cited_or_all():
    hits = [_hit("5"), _hit("6"), _hit("7")]
    assert [c["number"] for c in build_citations(hits, [2])] == ["6"]
    assert [c["id"] for c in build_citations(hits, [])] == [1, 2, 3]


def test_build_messages_numbers_sources():
    msgs = build_messages("What?", [_hit("5"), _hit("6")])
    assert "[1] GDPR – Article 5" in msgs[1]["content"]
    assert "[2] GDPR – Article 6" in msgs[1]["content"]


def test_act_detection():
    assert mentioned_acts("Does the Data Governance Act apply?") == ["Data Governance Act"]
    assert mentioned_acts("Under the Data Act, ...") == ["Data Act"]
    assert mentioned_acts("GDPR vs AI Act") == ["GDPR", "AI Act"]


def test_explicit_refs():
    assert explicit_refs("What does Article 6 GDPR say?", None) == [("GDPR", "article", "6", None)]
    assert explicit_refs("Explain Art. 5 and Article 50", ["AI Act"]) == [
        ("AI Act", "article", "5", None),
        ("AI Act", "article", "50", None),
    ]
    assert explicit_refs("Article 5 of GDPR or AI Act?", None) == []
    assert explicit_refs("What changed in Article 4a AI Act?", None) == [("AI Act", "article", "4a", None)]


def test_explicit_refs_with_paragraph():
    assert explicit_refs("What does Article 6(1) GDPR say?", None) == [("GDPR", "article", "6", "1")]
    assert explicit_refs("Art. 5 (3) ePrivacy and Article 5", None) == [
        ("ePrivacy Directive", "article", "5", "3"),
        ("ePrivacy Directive", "article", "5", None),
    ]


def test_explicit_refs_lists_and_ranges():
    numbers = lambda q: [(n, p) for _, _, n, p in explicit_refs(q, ["GDPR"])]
    assert numbers("What do Articles 13 and 14 require?") == [("13", None), ("14", None)]
    assert numbers("Art. 13, 14 and 15") == [("13", None), ("14", None), ("15", None)]
    assert numbers("Articles 6(1) or 9(2)") == [("6", "1"), ("9", "2")]
    assert numbers("Articles 15-17") == [("15", None), ("16", None), ("17", None)]
    assert numbers("Articles 12 to 14") == [("12", None), ("13", None), ("14", None)]
    assert numbers("Articles 1-99") == [("1", None), ("99", None)]
    assert len(numbers("Articles 12-22")) == 6
    assert numbers("Since 2018, under Article 17 and 2016 rules") == [("17", None), ("2016", None)]


def test_explicit_refs_annexes():
    assert explicit_refs("What does Annex III AI Act cover?", None) == [("AI Act", "annex", "III", None)]
    assert explicit_refs("Annexes I to IV AI Act", None) == [
        ("AI Act", "annex", n, None) for n in ("I", "II", "III", "IV")
    ]
    assert explicit_refs("Article 6 and annex iii of the AI Act", None) == [
        ("AI Act", "article", "6", None),
        ("AI Act", "annex", "III", None),
    ]
    assert explicit_refs("Annex III", None) == []


def test_covers_paragraph():
    assert covers_paragraph("", "7")
    assert covers_paragraph("1-3", "2") and covers_paragraph("1-3", "3")
    assert not covers_paragraph("1-3", "4")
    assert covers_paragraph("4", "4") and not covers_paragraph("4", "5")


def _settings(budget_chars: int) -> Settings:
    """Settings whose source budget is about `budget_chars` (chars_per_token=1, no answer reserve)."""
    fixed = len(SYSTEM_PROMPT) + len("Q?") + 200
    return Settings(num_ctx=fixed + budget_chars, answer_tokens=0, chars_per_token=1.0)


def test_fit_context_skips_oversized_hits_and_keeps_order():
    hits = [_hit("1", 500), _hit("2", 2000), _hit("3", 300)]
    kept = fit_context("Q?", hits, _settings(1000))
    assert [h.payload["number"] for h in kept] == ["1", "3"]


def test_fit_context_keeps_everything_that_fits():
    hits = [_hit("1", 100), _hit("2", 100)]
    assert fit_context("Q?", hits, _settings(1000)) == hits


def test_fit_context_always_keeps_first_hit():
    kept = fit_context("Q?", [_hit("1", 5000), _hit("2", 5000)], _settings(1000))
    assert [h.payload["number"] for h in kept] == ["1"]


def test_clean_citations_collapses_repeats_and_citation_only_lines():
    answer, cited = clean_citations("[1][3]\n\n- Consent [2][2] [2].\n- Contract [1].", n_sources=3)
    assert answer == "- Consent [2].\n- Contract [1]."
    assert cited == [1, 2, 3]


def test_clean_citations_dedupes_non_adjacent_repeats_in_run():
    answer, _ = clean_citations("Vital interests [2][1][2].", n_sources=3)
    assert answer == "Vital interests [2][1]."
