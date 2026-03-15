"""Company career pages scraper.

Reads the curated URL list from config (one 'URL | Company Name' entry per line).
Uses httpx + BeautifulSoup to extract job listings from each page.
Results are best-effort; per-URL failures are caught and logged at WARNING.
"""
import logging
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from agent.ingest.base import RawPosting, Scraper, clean_text

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Keywords that must appear in a job title to be considered relevant.
_RELEVANT_TITLE_KEYWORDS = re.compile(
    r"machine.?learning|ml engineer|data engineer|ai engineer|"
    r"mlops|llm|applied.?ai|quant|risk.model|safety|alignment|"
    r"research.?engineer|data.?scientist",
    re.IGNORECASE,
)


def _parse_career_pages_file(file_path: str) -> list[tuple[str, str]]:
    """Return list of (url, company_name) from the curated file."""
    entries: list[tuple[str, str]] = []
    path = Path(file_path)
    if not path.exists():
        logger.warning("career_pages file not found: %s", file_path)
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            url, company = line.split("|", 1)
            entries.append((url.strip(), company.strip()))
        else:
            entries.append((line, ""))
    return entries


def _extract_postings(html: str, base_url: str, company: str) -> list[RawPosting]:
    """Best-effort extraction of job postings from an HTML careers page."""
    soup = BeautifulSoup(html, "lxml")
    postings: list[RawPosting] = []
    seen: set[str] = set()

    # Generic heuristic: look for <a> tags that look like job links
    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]
        text = a_tag.get_text(strip=True)

        if not _RELEVANT_TITLE_KEYWORDS.search(text):
            continue

        # Build absolute URL
        if href.startswith("http"):
            job_url = href
        elif href.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            job_url = f"{parsed.scheme}://{parsed.netloc}{href}"
        else:
            job_url = f"{base_url.rstrip('/')}/{href}"

        job_url = job_url.split("?")[0]  # strip query params for stable ID
        if job_url in seen:
            continue
        seen.add(job_url)

        # Try to find a location nearby in the DOM
        parent = a_tag.find_parent()
        location_text = ""
        if parent:
            sibling_text = parent.get_text(" ", strip=True)
            loc_match = re.search(
                r"(remote|pittsburgh|new york|san francisco|austin|chicago|boston)", sibling_text, re.I
            )
            if loc_match:
                location_text = loc_match.group(0)

        remote: bool | None = "remote" in (text + location_text).lower() or None

        postings.append(
            RawPosting(
                posting_id=job_url,
                source="careers_page",
                company=company,
                title=text[:200],
                location=location_text,
                remote=remote,
                description=clean_text(""),  # full JD not fetched here
                url=job_url,
                scraped_at=datetime.utcnow(),
            )
        )

    return postings


class CareersPageScraper(Scraper):
    def __init__(self, careers_pages_file: str) -> None:
        self._file = careers_pages_file

    def fetch(self) -> list[RawPosting]:
        entries = _parse_career_pages_file(self._file)
        postings: list[RawPosting] = []

        for url, company in entries:
            try:
                with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                found = _extract_postings(resp.text, url, company)
                logger.info("CareersPageScraper: %s → %d postings", company or url, len(found))
                postings.extend(found)
            except Exception as exc:
                logger.warning("CareersPageScraper failed for %s: %s", url, exc)

        logger.info("CareersPageScraper: total %d postings", len(postings))
        return postings
