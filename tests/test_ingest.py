"""Ingest tests — mock HTTP responses, verify RawPosting output + dedup logic."""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import unicodedata

from agent.ingest.base import RawPosting, Scraper, clean_text
from agent.ingest.careers_page import CareersPageScraper, _parse_career_pages_file


# ---------- clean_text ----------

def test_clean_text_strips_html():
    result = clean_text("<strong>Engineer</strong> &amp; <br/>Developer")
    assert result == "Engineer & Developer"


def test_clean_text_normalizes_unicode():
    # bullet U+2022, em-dash U+2014, smart quotes U+201C / U+201D
    raw = "\u2022 bullet \u2014 dash \u201chello\u201d"
    result = clean_text(raw)
    expected = unicodedata.normalize("NFKC", raw).strip()
    assert result == expected


def test_clean_text_collapses_whitespace():
    result = clean_text("foo   bar\n\n\n\n\nbaz")
    assert result == "foo bar\n\nbaz"


# ---------- RawPosting dataclass ----------

def test_raw_posting_defaults():
    p = RawPosting(
        posting_id="abc",
        source="indeed",
        company="Acme",
        title="ML Engineer",
        location="Pittsburgh",
        remote=True,
        description="desc",
        url="https://example.com",
    )
    assert isinstance(p.scraped_at, datetime)


# ---------- Abstract Scraper ----------

def test_scraper_is_abstract():
    with pytest.raises(TypeError):
        Scraper()  # type: ignore


# ---------- careers_pages.txt parser ----------

def test_parse_career_pages_file(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text(
        "# comment\nhttps://example.com/jobs | Acme Corp\nhttps://other.io/careers | Other\n",
        encoding="utf-8",
    )
    entries = _parse_career_pages_file(str(f))
    assert len(entries) == 2
    assert entries[0] == ("https://example.com/jobs", "Acme Corp")
    assert entries[1] == ("https://other.io/careers", "Other")


def test_parse_career_pages_skips_comments(tmp_path):
    f = tmp_path / "career_pages.txt"
    f.write_text("# just a comment\n\nhttps://a.com | A\n", encoding="utf-8")
    entries = _parse_career_pages_file(str(f))
    assert len(entries) == 1


def test_parse_career_pages_missing_file():
    entries = _parse_career_pages_file("/nonexistent/path.txt")
    assert entries == []


# ---------- CareersPageScraper — mock HTTP ----------

SAMPLE_HTML = """
<html><body>
  <a href="/jobs/ml-engineer">Machine Learning Engineer</a>
  <a href="/jobs/ds">Data Scientist - remote</a>
  <a href="/about">About Us</a>
</body></html>
"""


def test_careers_page_scraper_filters_relevant_titles(tmp_path):
    career_file = tmp_path / "career_pages.txt"
    career_file.write_text("https://acme.com/careers | Acme\n", encoding="utf-8")

    mock_response = MagicMock()
    mock_response.text = SAMPLE_HTML
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(return_value=mock_response)

    with patch("agent.ingest.careers_page.httpx.Client", return_value=mock_client):
        scraper = CareersPageScraper(str(career_file))
        postings = scraper.fetch()

    # Should find "Machine Learning Engineer" and "Data Scientist"
    titles = [p.title for p in postings]
    assert any("Machine Learning" in t for t in titles)
    assert any("Data Scientist" in t for t in titles)
    # Should NOT include "About Us"
    assert not any("About" in t for t in titles)


def test_careers_page_scraper_handles_http_error(tmp_path):
    career_file = tmp_path / "career_pages.txt"
    career_file.write_text("https://acme.com/careers | Acme\n", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(side_effect=Exception("connection refused"))

    with patch("agent.ingest.careers_page.httpx.Client", return_value=mock_client):
        scraper = CareersPageScraper(str(career_file))
        # Must not raise — returns empty list
        postings = scraper.fetch()

    assert postings == []


# ---------- Dedup logic (tested via repository in test_db.py) ----------

def test_dedup_skips_seen_ids():
    """Verify that duplicate posting_ids are skipped in the pipeline loop."""
    from agent.db.repository import JobRepository, ScoreResult

    repo = JobRepository(":memory:")
    posting = RawPosting(
        posting_id="dup-001",
        source="indeed",
        company="Co",
        title="Engineer",
        location="Pittsburgh",
        remote=False,
        description="",
        url="",
    )
    scores = ScoreResult(
        role_score=0.8,
        location_score=1.0,
        stack_score=0.6,
        composite_score=0.82,
        rationale="",
        skill_gaps=[],
    )
    repo.insert_job(posting, scores)
    seen = repo.get_seen_ids()
    assert "dup-001" in seen
    # Simulating dedup: second posting with same ID would be filtered out
    postings = [posting]
    new_postings = [p for p in postings if p.posting_id not in seen]
    assert len(new_postings) == 0
