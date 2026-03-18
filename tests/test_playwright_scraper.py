"""Tests for PlaywrightLLMScraper and related helpers.

All external calls (Playwright browser, LLM API) are mocked.
Tests cover:
- PortalConfig parsing
- _parse_all_career_entries (three-column format)
- build_careers_scrapers factory routing
- _build_snapshot (HTML → plain text)
- _resolve_url helper
- _inject_keywords_into_url helper
- Pydantic model validation (JobItem, ExtractJobsResponse, ExploreAction, FilterJobsResponse)
- _llm_extract_jobs / _llm_filter_jobs / _llm_explore_portal (mocked LLM)
- PlaywrightLLMScraper.fetch() (mocked asyncio.run + LLM)
- _ExploreCache helpers
- build_careers_scrapers returns correct scraper types
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.ingest.careers_page import (
    CareerEntry,
    PortalConfig,
    _parse_all_career_entries,
    _parse_career_pages_file,
    build_careers_scrapers,
)
from agent.ingest.playwright_scraper import (
    ExtractJobsResponse,
    ExploreAction,
    FilterJobsResponse,
    JobItem,
    PlaywrightLLMScraper,
    _ExploreCache,
    _build_snapshot,
    _inject_keywords_into_url,
    _llm_extract_jobs,
    _llm_explore_portal,
    _llm_filter_jobs,
    _resolve_url,
    _explore_cache,
)


# ---------------------------------------------------------------------------
# PortalConfig parsing
# ---------------------------------------------------------------------------


def test_portal_config_from_column_basic():
    cfg = PortalConfig.from_column("portal=workday")
    assert cfg.portal_type == "workday"
    assert cfg.keywords == []


def test_portal_config_from_column_with_keywords():
    cfg = PortalConfig.from_column("portal=eightfold;keywords=machine learning,data engineer")
    assert cfg.portal_type == "eightfold"
    assert cfg.keywords == ["machine learning", "data engineer"]


def test_portal_config_from_column_unknown_portal():
    cfg = PortalConfig.from_column("portal=unknown;keywords=AI,ML")
    assert cfg.portal_type == "unknown"
    assert "AI" in cfg.keywords


def test_portal_config_from_column_lowercase_normalizes():
    cfg = PortalConfig.from_column("portal=WORKDAY")
    assert cfg.portal_type == "workday"


def test_portal_config_from_column_empty():
    cfg = PortalConfig.from_column("")
    assert cfg.portal_type == "unknown"
    assert cfg.keywords == []


# ---------------------------------------------------------------------------
# _parse_all_career_entries
# ---------------------------------------------------------------------------


def test_parse_all_entries_simple_line(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text("https://example.com/jobs | Acme Corp\n", encoding="utf-8")
    entries = _parse_all_career_entries(str(f))
    assert len(entries) == 1
    assert entries[0].portal is None
    assert entries[0].company == "Acme Corp"


def test_parse_all_entries_portal_line(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://cmu.wd5.myworkdayjobs.com/CMU | Carnegie Mellon | portal=workday\n",
        encoding="utf-8",
    )
    entries = _parse_all_career_entries(str(f))
    assert len(entries) == 1
    assert entries[0].portal is not None
    assert entries[0].portal.portal_type == "workday"
    assert entries[0].company == "Carnegie Mellon"


def test_parse_all_entries_portal_with_keywords(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://careers.pnc.com | PNC | portal=phenom;keywords=machine learning,data engineer\n",
        encoding="utf-8",
    )
    entries = _parse_all_career_entries(str(f))
    assert entries[0].portal.portal_type == "phenom"
    assert "machine learning" in entries[0].portal.keywords


def test_parse_all_entries_skips_comments(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "# comment\nhttps://a.com | A\nhttps://b.com | B | portal=workday\n",
        encoding="utf-8",
    )
    entries = _parse_all_career_entries(str(f))
    assert len(entries) == 2


def test_parse_all_entries_missing_file():
    entries = _parse_all_career_entries("/nonexistent/path.txt")
    assert entries == []


def test_parse_career_pages_file_excludes_portal_lines(tmp_path):
    """Backward-compat: _parse_career_pages_file should NOT return portal entries."""
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://simple.com | Simple\nhttps://portal.com | Portal | portal=workday\n",
        encoding="utf-8",
    )
    entries = _parse_career_pages_file(str(f))
    # Only simple entry returned
    assert len(entries) == 1
    assert entries[0][0] == "https://simple.com"


# ---------------------------------------------------------------------------
# build_careers_scrapers factory
# ---------------------------------------------------------------------------


def test_build_careers_scrapers_returns_simple_and_playwright(tmp_path):
    from agent.ingest.careers_page import _SimpleCareersScraperFromEntries

    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://simple.com | Simple\nhttps://portal.com | Portal | portal=workday\n",
        encoding="utf-8",
    )
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 2048}

    scrapers = build_careers_scrapers(str(f), llm_config)
    scraper_types = [type(s).__name__ for s in scrapers]

    assert "_SimpleCareersScraperFromEntries" in scraper_types
    assert "PlaywrightLLMScraper" in scraper_types


def test_build_careers_scrapers_all_simple(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://a.com | A\nhttps://b.com | B\n",
        encoding="utf-8",
    )
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 2048}
    scrapers = build_careers_scrapers(str(f), llm_config)
    assert len(scrapers) == 1  # one _SimpleCareersScraperFromEntries for all simple entries


def test_build_careers_scrapers_all_portal(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://a.com | A | portal=workday\nhttps://b.com | B | portal=eightfold\n",
        encoding="utf-8",
    )
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 2048}
    scrapers = build_careers_scrapers(str(f), llm_config)
    assert len(scrapers) == 2
    assert all(type(s).__name__ == "PlaywrightLLMScraper" for s in scrapers)


# ---------------------------------------------------------------------------
# _build_snapshot
# ---------------------------------------------------------------------------


def test_build_snapshot_strips_scripts():
    html = "<html><head><script>var x=1;</script></head><body><h1>Jobs</h1></body></html>"
    snapshot = _build_snapshot(html)
    assert "var x=1" not in snapshot
    assert "Jobs" in snapshot


def test_build_snapshot_truncates():
    html = "<html><body>" + "<p>word</p>" * 5000 + "</body></html>"
    snapshot = _build_snapshot(html, max_chars=100)
    assert len(snapshot) <= 130  # truncated + "[truncated]" suffix
    assert "[truncated]" in snapshot


def test_build_snapshot_collapses_whitespace():
    html = "<html><body><p>hello    world</p></body></html>"
    snapshot = _build_snapshot(html)
    assert "hello world" in snapshot
    # No run of 3+ spaces
    assert "   " not in snapshot


def test_build_snapshot_empty_html():
    snapshot = _build_snapshot("")
    assert snapshot == "" or snapshot is not None


# ---------------------------------------------------------------------------
# _resolve_url
# ---------------------------------------------------------------------------


def test_resolve_url_absolute():
    assert _resolve_url("https://other.com/job/1", "https://base.com") == "https://other.com/job/1"


def test_resolve_url_root_relative():
    result = _resolve_url("/jobs/123", "https://careers.example.com/search")
    assert result == "https://careers.example.com/jobs/123"


def test_resolve_url_protocol_relative():
    result = _resolve_url("//careers.example.com/job/1", "https://base.com")
    assert result == "https://careers.example.com/job/1"


def test_resolve_url_empty_href_returns_base():
    result = _resolve_url("", "https://base.com/careers")
    assert result == "https://base.com/careers"


# ---------------------------------------------------------------------------
# _inject_keywords_into_url
# ---------------------------------------------------------------------------


def test_inject_keywords_replaces_placeholder():
    url = "https://careers.example.com/jobs?keyword=__KEYWORDS__&location=Pittsburgh"
    result = _inject_keywords_into_url(url, "machine learning")
    assert result == "https://careers.example.com/jobs?keyword=machine+learning&location=Pittsburgh"


def test_inject_keywords_no_placeholder():
    url = "https://careers.example.com/jobs?filter=tech"
    result = _inject_keywords_into_url(url, "ML engineer")
    assert result == url  # unchanged


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


def test_job_item_valid():
    item = JobItem(title="ML Engineer", url="https://example.com/job/1", location="Pittsburgh, PA")
    assert item.title == "ML Engineer"


def test_job_item_empty_title_raises():
    with pytest.raises(Exception):
        JobItem(title="   ")


def test_job_item_empty_url_becomes_none():
    item = JobItem(title="Engineer", url="")
    assert item.url is None


def test_extract_jobs_response_defaults():
    resp = ExtractJobsResponse()
    assert resp.jobs == []
    assert resp.has_next_page is False


def test_extract_jobs_response_from_dict():
    data = {
        "jobs": [{"title": "Data Engineer", "url": "https://x.com/1", "location": "Remote", "remote": True}],
        "has_next_page": False,
    }
    resp = ExtractJobsResponse.model_validate(data)
    assert len(resp.jobs) == 1
    assert resp.jobs[0].remote is True


def test_explore_action_valid():
    action = ExploreAction(action="extract", reasoning="jobs visible")
    assert action.action == "extract"


def test_explore_action_invalid_action_raises():
    with pytest.raises(Exception):
        ExploreAction(action="fly_to_moon")


def test_filter_jobs_response_defaults():
    resp = FilterJobsResponse()
    assert resp.relevant_indices == []


# ---------------------------------------------------------------------------
# _ExploreCache helpers
# ---------------------------------------------------------------------------


def test_explore_cache_put_and_get():
    cache = _ExploreCache()
    actions = [ExploreAction(action="click", selector="button.search")]
    cache.put("workday", actions)
    assert cache.has("workday")
    assert cache.get("workday") == actions


def test_explore_cache_miss_returns_none():
    cache = _ExploreCache()
    assert cache.get("eightfold") is None
    assert not cache.has("eightfold")


def test_explore_cache_overwrite():
    cache = _ExploreCache()
    a1 = [ExploreAction(action="click")]
    a2 = [ExploreAction(action="extract")]
    cache.put("workday", a1)
    cache.put("workday", a2)
    assert cache.get("workday") == a2


# ---------------------------------------------------------------------------
# _llm_extract_jobs (mocked LLM)
# ---------------------------------------------------------------------------


def _make_mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = MagicMock(return_value=response)
    return llm


def test_llm_extract_jobs_valid_response():
    payload = json.dumps({
        "jobs": [
            {"title": "ML Engineer", "url": "https://x.com/1", "location": "Pittsburgh", "remote": None}
        ],
        "has_next_page": False,
    })
    llm = _make_mock_llm(payload)
    resp = _llm_extract_jobs(llm, "snapshot text", "Acme", "workday", "https://acme.com", ["ML"])
    assert len(resp.jobs) == 1
    assert resp.jobs[0].title == "ML Engineer"


def test_llm_extract_jobs_invalid_json_returns_empty():
    llm = _make_mock_llm("not json at all {{{")
    resp = _llm_extract_jobs(llm, "snap", "Acme", "workday", "https://x.com", ["ML"])
    assert resp.jobs == []


def test_llm_extract_jobs_passes_keywords_to_prompt():
    payload = json.dumps({"jobs": [], "has_next_page": False})
    llm = _make_mock_llm(payload)
    _llm_extract_jobs(llm, "snap", "Acme", "workday", "https://x.com", ["data engineer", "MLOps"])
    call_args = llm.complete.call_args
    # keywords should appear in the user prompt
    assert "data engineer" in call_args[1]["user"] or "data engineer" in str(call_args)


# ---------------------------------------------------------------------------
# _llm_explore_portal (mocked LLM)
# ---------------------------------------------------------------------------


def test_llm_explore_portal_returns_extract_action():
    payload = json.dumps({
        "action": "extract",
        "selector": None,
        "value": None,
        "url": None,
        "reasoning": "jobs visible on page",
    })
    llm = _make_mock_llm(payload)
    action = _llm_explore_portal(llm, "snap", "Acme", "workday", "https://x.com", ["ML"])
    assert action.action == "extract"


def test_llm_explore_portal_bad_json_returns_done():
    llm = _make_mock_llm("INVALID")
    action = _llm_explore_portal(llm, "snap", "Acme", "unknown", "https://x.com", ["ML"])
    assert action.action == "done"


def test_llm_explore_portal_uses_workday_prompt():
    """Workday portal should use the Workday-specific system prompt."""
    from agent.ingest.portal_prompts import WORKDAY_EXPLORE_SYSTEM_PROMPT

    payload = json.dumps({"action": "extract", "reasoning": "ok"})
    llm = _make_mock_llm(payload)
    _llm_explore_portal(llm, "snap", "CMU", "workday", "https://cmu.wd5.myworkdayjobs.com", ["ML"])
    call_kwargs = llm.complete.call_args[1]
    assert call_kwargs["system"] == WORKDAY_EXPLORE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _llm_filter_jobs (mocked LLM)
# ---------------------------------------------------------------------------


def test_llm_filter_jobs_returns_relevant_indices():
    payload = json.dumps({"relevant_indices": [0, 2]})
    llm = _make_mock_llm(payload)
    result = _llm_filter_jobs(llm, ["ML Engineer", "Janitor", "Data Engineer"])
    assert result == [0, 2]


def test_llm_filter_jobs_empty_titles():
    llm = _make_mock_llm("{}")
    result = _llm_filter_jobs(llm, [])
    assert result == []
    llm.complete.assert_not_called()


def test_llm_filter_jobs_bad_json_returns_all_indices():
    llm = _make_mock_llm("BAD")
    result = _llm_filter_jobs(llm, ["A", "B", "C"])
    # On failure, return all indices (inclusive fallback)
    assert set(result) == {0, 1, 2}


# ---------------------------------------------------------------------------
# PlaywrightLLMScraper.fetch() — mocked asyncio.run
# ---------------------------------------------------------------------------


def _make_raw_posting_data():
    from datetime import datetime
    from agent.ingest.base import RawPosting
    return RawPosting(
        posting_id="https://portal.com/job/1",
        source="careers_page",
        company="Acme",
        title="ML Engineer",
        location="Pittsburgh, PA",
        remote=None,
        description="",
        url="https://portal.com/job/1",
        scraped_at=datetime.utcnow(),
    )


def test_playwright_scraper_fetch_returns_postings():
    """fetch() should return postings produced by asyncio.run()."""
    portal_config = PortalConfig(portal_type="workday", keywords=["ML"])
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 512}

    expected = [_make_raw_posting_data()]

    scraper = PlaywrightLLMScraper(
        url="https://example.wd5.myworkdayjobs.com/JOBS",
        company="Acme",
        portal_config=portal_config,
        llm_config=llm_config,
    )

    with patch("agent.ingest.playwright_scraper.asyncio.run", return_value=expected) as mock_run:
        with patch("agent.ingest.playwright_scraper.build_llm_backend") as mock_build:
            mock_build.return_value = MagicMock()
            postings = scraper.fetch()

    assert postings == expected
    mock_run.assert_called_once()


def test_playwright_scraper_fetch_returns_empty_on_llm_build_failure():
    """If build_llm_backend raises, fetch() must return [] without raising."""
    portal_config = PortalConfig(portal_type="workday", keywords=["ML"])
    llm_config = {"provider": "claude", "model": "bad-model", "max_tokens": 512}

    scraper = PlaywrightLLMScraper(
        url="https://example.com",
        company="Acme",
        portal_config=portal_config,
        llm_config=llm_config,
    )

    with patch(
        "agent.ingest.playwright_scraper.build_llm_backend",
        side_effect=Exception("no api key"),
    ):
        postings = scraper.fetch()

    assert postings == []


def test_playwright_scraper_fetch_returns_empty_on_asyncio_error():
    """If asyncio.run raises, fetch() must return [] without raising."""
    portal_config = PortalConfig(portal_type="unknown", keywords=["ML"])
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 512}

    scraper = PlaywrightLLMScraper(
        url="https://example.com",
        company="Acme",
        portal_config=portal_config,
        llm_config=llm_config,
    )

    with patch("agent.ingest.playwright_scraper.build_llm_backend") as mock_build:
        mock_build.return_value = MagicMock()
        with patch(
            "agent.ingest.playwright_scraper.asyncio.run",
            side_effect=Exception("browser launch failed"),
        ):
            postings = scraper.fetch()

    assert postings == []


def test_playwright_scraper_uses_asyncio_run_bridge():
    """fetch() must use asyncio.run() (not await) — synchronous interface contract."""
    portal_config = PortalConfig(portal_type="workday")
    llm_config = {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 512}

    scraper = PlaywrightLLMScraper(
        url="https://example.com",
        company="Test",
        portal_config=portal_config,
        llm_config=llm_config,
    )

    with patch("agent.ingest.playwright_scraper.asyncio.run", return_value=[]) as mock_run:
        with patch("agent.ingest.playwright_scraper.build_llm_backend", return_value=MagicMock()):
            scraper.fetch()

    # asyncio.run was called (not await) — confirming synchronous bridge pattern
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: agent/main.py wiring
# ---------------------------------------------------------------------------


def test_main_uses_build_careers_scrapers_when_playwright_enabled(tmp_path):
    """run_pipeline should call build_careers_scrapers when sources.playwright=True."""
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "https://simple.com | Simple\nhttps://portal.com | Portal | portal=workday\n",
        encoding="utf-8",
    )
    config = {
        "sources": {
            "linkedin": False,
            "indeed": False,
            "careers_pages_file": str(f),
            "playwright": True,
        },
        "llm": {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 512},
        "embedding": {"model": "all-MiniLM-L6-v2", "similarity_threshold": 0.28, "extraction_chars": 2000},
        "scoring": {"weights": {"role_type": 0.5, "location": 0.3, "stack": 0.2}},
        "db_path": str(tmp_path / "jobs.db"),
    }

    with patch("agent.main.build_careers_scrapers", return_value=[]) as mock_bcs:
        with patch("agent.main.build_llm_backend") as mock_llm:
            with patch("agent.main.ScoringPipeline") as mock_sp:
                mock_llm.return_value = MagicMock()
                mock_sp.return_value = MagicMock()
                mock_sp.return_value.score = MagicMock(return_value=None)
                from agent.main import run_pipeline
                run_pipeline(config)

    mock_bcs.assert_called_once_with(
        careers_pages_file=str(f),
        llm_config=config["llm"],
    )


def test_main_falls_back_to_careers_page_scraper_when_playwright_disabled(tmp_path):
    """When sources.playwright=False, CareersPageScraper is used directly."""
    f = tmp_path / "career_pages.txt"
    f.write_text("https://simple.com | Simple\n", encoding="utf-8")
    config = {
        "sources": {
            "linkedin": False,
            "indeed": False,
            "careers_pages_file": str(f),
            "playwright": False,
        },
        "llm": {"provider": "claude", "model": "claude-sonnet-4-20250514", "max_tokens": 512},
        "embedding": {"model": "all-MiniLM-L6-v2", "similarity_threshold": 0.28, "extraction_chars": 2000},
        "scoring": {"weights": {"role_type": 0.5, "location": 0.3, "stack": 0.2}},
        "db_path": str(tmp_path / "jobs.db"),
    }

    with patch("agent.main.build_careers_scrapers") as mock_bcs:
        with patch("agent.main.build_llm_backend") as mock_llm:
            with patch("agent.main.ScoringPipeline") as mock_sp:
                with patch("agent.main.CareersPageScraper") as mock_cps:
                    mock_llm.return_value = MagicMock()
                    mock_sp.return_value = MagicMock()
                    mock_sp.return_value.score = MagicMock(return_value=None)
                    mock_cps.return_value = MagicMock()
                    mock_cps.return_value.fetch = MagicMock(return_value=[])
                    from agent.main import run_pipeline
                    run_pipeline(config)

    # build_careers_scrapers should NOT be called when playwright=False
    mock_bcs.assert_not_called()
    # CareersPageScraper should be used instead
    mock_cps.assert_called_once_with(str(f))
