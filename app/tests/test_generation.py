import asyncio
import json

import httpx

from eurag.config import Settings
from eurag.generation import (
    DECLINE_FALLBACK,
    SYSTEM_PROMPT,
    DeclineFilter,
    build_citations,
    build_messages,
    clean_citations,
    debug_payload,
    fit_context,
)
from eurag.ollama_client import OllamaClient, query_prefix
from eurag.retrieval import Hit, covers_paragraph, explicit_refs, mentioned_acts, quota_acts, quota_merge


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
    answer, cited, declined = clean_citations("Consent is a basis [1][3]. Also [7] and [2].", n_sources=3)
    assert answer == "Consent is a basis [1][3]. Also and [2]."
    assert cited == [1, 2, 3]
    assert not declined


def test_clean_citations_fixes_spacing_left_by_removed_markers():
    answer, _, _ = clean_citations("It applies [9] . See Article 6 [8], too.\n  - nested [1]", n_sources=1)
    assert answer == "It applies. See Article 6, too.\n  - nested [1]"


def test_clean_citations_point_markers():
    answer, cited, _ = clean_citations("Advise the controller [1(a)]. Monitor compliance [1(b)][1(c)].", 3)
    assert (answer, cited) == ("Advise the controller [1]. Monitor compliance [1].", [1])
    answer, cited, _ = clean_citations("Tasks [2(1)(a)], [2(1)(c)-(g)] and [3(iii)-(vi)]; not [9(a)].", 3)
    assert (answer, cited) == ("Tasks [2], [2] and [3]; not.", [2, 3])
    # Bracketed text that is not a citation stays.
    assert clean_citations("Article 6 [(a) to (f)] and [see 2(1)].", 3)[:2] == (
        "Article 6 [(a) to (f)] and [see 2(1)].", [])


def test_clean_citations_decline():
    answer, cited, declined = clean_citations("NOT_COVERED: The sources do not cover tax law [2].", n_sources=3)
    assert (answer, cited, declined) == ("The sources do not cover tax law.", [], True)
    assert clean_citations("**NOT_COVERED**\n\nNothing on this.", 3)[::2] == ("Nothing on this.", True)
    assert clean_citations(" NOT_COVERED ", 3) == (DECLINE_FALLBACK, [], True)
    # Behind a bullet or a citation run (seen in eval run 20261028-074907_baseline).
    assert clean_citations("- NOT_COVERED [1][2]\n\nThe DMA is not in the sources.", 3) == (
        "The DMA is not in the sources.", [], True)
    assert clean_citations("[1][3] NOT_COVERED The sources lack this.", 3)[1:] == ([], True)
    # Only a leading marker counts as a decline.
    assert clean_citations("Consent [1]. NOT_COVERED", 3)[2] is False


def _stream(tokens: list[str]) -> tuple[str, bool]:
    f = DeclineFilter()
    out = "".join(f.feed(t) for t in tokens) + f.flush()
    return out, f.declined


def test_decline_filter_hides_marker():
    assert _stream(["NOT", "_CO", "VERED", ":", " The sources", " lack this."]) == ("The sources lack this.", True)
    assert _stream(["**NOT_COVERED**", "\n\n", "Missing."]) == ("Missing.", True)
    assert _stream(["-", " NOT", "_COVERED", " [1]", "\n\nMissing."])[1] is True
    assert _stream(["[", "1", "][5] NOT_COVERED", " Missing."])[1] is True


def test_decline_filter_passes_normal_answers():
    assert _stream(["Consent", " is required [1]."]) == ("Consent is required [1].", False)
    assert _stream(["NO", "T every case [1]."]) == ("NOT every case [1].", False)
    assert _stream(["No"]) == ("No", False)
    assert _stream(["- ", "Consent", " [1]."]) == ("- Consent [1].", False)
    assert _stream(["[1]", " The GDPR", " allows [1]."]) == ("[1] The GDPR allows [1].", False)


def test_build_citations_only_cited():
    hits = [_hit("5"), _hit("6"), _hit("7")]
    assert [c["number"] for c in build_citations(hits, [2])] == ["6"]
    assert build_citations(hits, []) == []


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


def test_quota_acts_respects_filter():
    q = "How does the AI Act relate to the GDPR?"
    assert quota_acts(q, None) == ["GDPR", "AI Act"]
    assert quota_acts(q, ["GDPR"]) == ["GDPR"]


def test_quota_merge_reserves_slots_per_act():
    ai = [Hit(s, {"act": "AI Act", "n": i}) for i, s in enumerate([0.9, 0.85, 0.8, 0.79, 0.78])]
    gdpr = [Hit(s, {"act": "GDPR", "n": i}) for i, s in enumerate([0.6, 0.5, 0.4])]
    merged = quota_merge([ai, gdpr], k=4)
    # 2 slots each: GDPR's best two get in although AI Act's 3rd-5th score higher
    assert [(h.payload["act"], h.score) for h in merged[:4]] == [
        ("AI Act", 0.9), ("AI Act", 0.85), ("GDPR", 0.6), ("GDPR", 0.5)]
    assert [h.score for h in merged[4:]] == [0.8, 0.79, 0.78, 0.4]


def test_quota_merge_fills_unused_share_by_score():
    ai = [Hit(s, {"act": "AI Act"}) for s in [0.9, 0.8, 0.7]]
    gdpr = [Hit(0.5, {"act": "GDPR"})]  # fewer hits than its share
    assert [h.score for h in quota_merge([ai, gdpr], k=4)[:4]] == [0.9, 0.8, 0.7, 0.5]


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
    answer, cited, _ = clean_citations("[1][3]\n\n- Consent [2][2] [2].\n- Contract [1].", n_sources=3)
    assert answer == "- Consent [2].\n- Contract [1]."
    assert cited == [1, 2, 3]


def test_clean_citations_dedupes_non_adjacent_repeats_in_run():
    answer, _, _ = clean_citations("Vital interests [2][1][2].", n_sources=3)
    assert answer == "Vital interests [2][1]."


def test_debug_payload_marks_context_and_cited_texts():
    retrieved = [_hit("6"), _hit("7"), _hit("9")]
    kept = [retrieved[0], retrieved[2]]  # Article 7 cut by the context budget
    debug = debug_payload(retrieved, kept, cited=[2])
    assert [(r["number"], r["in_context"]) for r in debug["retrieved"]] == [("6", True), ("7", False), ("9", True)]
    assert debug["sources"] == [{"id": 2, "text": "GDPR – Article 9\nBody 9"}]


def test_query_prefix_by_model_name():
    assert query_prefix("bge-m3") == ""
    assert query_prefix("snowflake-arctic-embed2:latest") == "query: "
    assert query_prefix("qwen3-embedding:0.6b").startswith("Instruct: ")
    assert query_prefix("qwen3-embedding:0.6b").endswith("\nQuery:")


def test_embed_prefixes_queries_only():
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.0]]})

    async def embed_both() -> None:
        client = OllamaClient(Settings(embed_model="snowflake-arctic-embed2"))
        client._http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
        await client.embed(["chunk"])
        await client.embed(["question?"], query=True)
        await client.aclose()

    asyncio.run(embed_both())
    assert sent == [["chunk"], ["query: question?"]]
