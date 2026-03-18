"""Company career pages scraper.

Reads the curated URL list from config (one entry per line).

Two line formats are supported:

  Simple (two columns):
      URL | Company Name

  Portal (three columns):
      URL | Company Name | portal=TYPE;keywords=kw1,kw2

Lines without a third column are routed to CareersPageScraper (httpx +
BeautifulSoup heuristic extraction). Lines with a third column are routed to
PlaywrightLLMScraper (JS-rendered, LLM-assisted extraction).

Results are best-effort; per-URL failures are caught and logged at WARNING.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import certifi
import httpx
from bs4 import BeautifulSoup

from agent.ingest.base import RawPosting, Scraper, clean_text

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Keywords that must appear in a job title to be considered relevant.
_RELEVANT_TITLE_KEYWORDS = re.compile(
    r"machine.?learning|ml engineer|data engineer|ai engineer|"
    r"mlops|llm|applied.?ai|quant|risk.model|safety|alignment|"
    r"research.?engineer|data.?scientist",
    re.IGNORECASE,
)


@dataclass
class PortalConfig:
    """Parsed portal metadata from the third column of career_pages.txt."""

    portal_type: str  # workday | eightfold | greenhouse | phenom | brassring |
    #                   taleo | talentbrew | custom_url_params | unknown
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_column(cls, column: str) -> "PortalConfig":
        """Parse 'portal=workday;keywords=ml,data engineer' into PortalConfig."""
        portal_type = "unknown"
        keywords: list[str] = []

        for segment in column.split(";"):
            segment = segment.strip()
            if segment.startswith("portal="):
                portal_type = segment[len("portal=") :].strip().lower()
            elif segment.startswith("keywords="):
                raw = segment[len("keywords=") :].strip()
                keywords = [k.strip() for k in raw.split(",") if k.strip()]

        return cls(portal_type=portal_type, keywords=keywords)


@dataclass
class CareerEntry:
    """A single parsed line from career_pages.txt."""

    url: str
    company: str
    portal: PortalConfig | None  # None → simple scraper; not-None → playwright scraper


def _parse_career_pages_file(file_path: str) -> list[tuple[str, str]]:
    """Return list of (url, company_name) for SIMPLE entries only.

    This is the backward-compatible interface used by existing tests and by
    CareersPageScraper. Portal entries (three columns) are excluded.
    """
    entries: list[tuple[str, str]] = []
    path = Path(file_path)
    if not path.exists():
        logger.warning("career_pages file not found: %s", file_path)
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) == 2:
            url, company = parts
            entries.append((url.strip(), company.strip()))
        elif len(parts) == 1:
            entries.append((line.strip(), ""))
        # Three-column entries (portal lines) are intentionally skipped here.
    return entries


def _parse_all_career_entries(file_path: str) -> list[CareerEntry]:
    """Parse all entries including portal lines. Returns CareerEntry objects."""
    entries: list[CareerEntry] = []
    path = Path(file_path)
    if not path.exists():
        logger.warning("career_pages file not found: %s", file_path)
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            url, company, portal_col = parts
            portal = PortalConfig.from_column(portal_col)
            entries.append(CareerEntry(url=url, company=company, portal=portal))
        elif len(parts) == 2:
            url, company = parts
            entries.append(CareerEntry(url=url, company=company, portal=None))
        elif len(parts) == 1:
            entries.append(CareerEntry(url=parts[0], company="", portal=None))
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
                r"(remote|pittsburgh|new york|san francisco|austin|chicago|boston)",
                sibling_text,
                re.I,
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
    """Scrapes simple (non-JS) career pages via httpx + BeautifulSoup.

    Handles only two-column entries from career_pages.txt. Portal entries
    (three-column lines) are handled by PlaywrightLLMScraper.
    """

    def __init__(self, careers_pages_file: str) -> None:
        self._file = careers_pages_file

    def fetch(self) -> list[RawPosting]:
        entries = _parse_career_pages_file(self._file)
        postings: list[RawPosting] = []

        for url, company in entries:
            try:
                with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20, verify=certifi.where()) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                found = _extract_postings(resp.text, url, company)
                logger.info("CareersPageScraper: %s → %d postings", company or url, len(found))
                postings.extend(found)
            except Exception as exc:
                logger.warning("CareersPageScraper failed for %s: %s", url, exc)

        logger.info("CareersPageScraper: total %d postings", len(postings))
        return postings


def build_careers_scrapers(
    careers_pages_file: str,
    llm_config: dict[str, Any],
) -> list[Scraper]:
    """Factory: parse career_pages.txt and return appropriate Scraper instances.

    Simple entries → one CareersPageScraper instance.
    Portal entries → one PlaywrightLLMScraper per entry.

    Import of PlaywrightLLMScraper is deferred to avoid forcing playwright as
    a hard import at module load time.
    """
    from agent.ingest.playwright_scraper import PlaywrightLLMScraper  # deferred

    entries = _parse_all_career_entries(careers_pages_file)

    simple_entries = [e for e in entries if e.portal is None]
    portal_entries = [e for e in entries if e.portal is not None]

    scrapers: list[Scraper] = []

    if simple_entries:
        # Reconstruct a temporary in-memory "file" by building a list directly
        # so CareersPageScraper can be passed just the simple entries.
        # We do this by writing a temp-file-equivalent in-memory path trick —
        # simpler: subclass with direct entry list.
        scrapers.append(_SimpleCareersScraperFromEntries(simple_entries))

    for entry in portal_entries:
        scrapers.append(
            PlaywrightLLMScraper(
                url=entry.url,
                company=entry.company,
                portal_config=entry.portal,
                llm_config=llm_config,
            )
        )

    return scrapers


class _SimpleCareersScraperFromEntries(Scraper):
    """CareersPageScraper variant that works from already-parsed CareerEntry list."""

    def __init__(self, entries: list[CareerEntry]) -> None:
        self._entries = entries

    def fetch(self) -> list[RawPosting]:
        postings: list[RawPosting] = []
        for entry in self._entries:
            try:
                with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20, verify=certifi.where()) as client:
                    resp = client.get(entry.url)
                    resp.raise_for_status()
                found = _extract_postings(resp.text, entry.url, entry.company)
                logger.info(
                    "CareersPageScraper: %s → %d postings",
                    entry.company or entry.url,
                    len(found),
                )
                postings.extend(found)
            except Exception as exc:
                logger.warning("CareersPageScraper failed for %s: %s", entry.url, exc)
        logger.info("CareersPageScraper: total %d postings", len(postings))
        return postings
