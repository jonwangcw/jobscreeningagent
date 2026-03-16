"""LinkedIn scraper — uses the public job search page with httpx + BeautifulSoup.

LinkedIn aggressively blocks scraping, so this implementation uses the
public /jobs/search URL (no login required) with randomised User-Agent
headers and a short delay between pages.  Results are best-effort; failures
are caught and logged at WARNING.
"""
import logging
import time
from datetime import datetime
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup

from agent.ingest.base import RawPosting, Scraper, clean_text

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.linkedin.com/jobs/search/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_SEARCHES: list[tuple[str, str]] = [
    ("machine learning engineer", "Pittsburgh, PA"),
    ("data engineer", "Pittsburgh, PA"),
    ("AI engineer", "Pittsburgh, PA"),
    ("data scientist", "Pittsburgh, PA"),
    ("machine learning engineer", "United States"),
    ("ML engineer remote", "United States"),
    ("AI safety", "United States"),
]


def _build_search_url(keywords: str, location: str, start: int = 0) -> str:
    params = {
        "keywords": keywords,
        "location": location,
        "f_WT": "2",   # remote filter
        "sortBy": "DD",
        "start": start,
    }
    return f"{_BASE_URL}?{urlencode(params)}"


def _detect_remote(title: str, location: str) -> bool | None:
    combined = f"{title} {location}".lower()
    if "remote" in combined:
        return True
    return None


class LinkedInScraper(Scraper):
    def fetch(self) -> list[RawPosting]:
        postings: list[RawPosting] = []
        seen: set[str] = set()

        for keywords, location in _SEARCHES:
            for start in [0, 25]:
                url = _build_search_url(keywords, location, start)
                try:
                    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=15) as client:
                        resp = client.get(url)
                        resp.raise_for_status()

                    soup = BeautifulSoup(resp.text, "lxml")
                    cards = soup.select("div.base-card")
                    if not cards:
                        # Try alternate selector used by newer LinkedIn markup
                        cards = soup.select("li.jobs-search__results-list > div")

                    for card in cards:
                        link_tag = card.select_one("a.base-card__full-link, a[data-tracking-control-name]")
                        title_tag = card.select_one("h3.base-search-card__title, span.sr-only")
                        company_tag = card.select_one("h4.base-search-card__subtitle, a.hidden-nested-link")
                        location_tag = card.select_one("span.job-search-card__location")

                        if not link_tag:
                            continue

                        job_url = link_tag.get("href", "").split("?")[0].strip()
                        if not job_url or job_url in seen:
                            continue
                        seen.add(job_url)

                        title = (title_tag.get_text(strip=True) if title_tag else keywords)
                        company = (company_tag.get_text(strip=True) if company_tag else "")
                        raw_location = (location_tag.get_text(strip=True) if location_tag else location)

                        postings.append(
                            RawPosting(
                                posting_id=job_url,
                                source="linkedin",
                                company=company,
                                title=title,
                                location=raw_location,
                                remote=_detect_remote(title, raw_location),
                                description=clean_text(""),  # full JD fetched lazily if posting passes gates
                                url=job_url,
                                scraped_at=datetime.utcnow(),
                            )
                        )

                    time.sleep(1.5)

                except Exception as exc:
                    logger.warning(
                        "LinkedInScraper failed for keywords=%r location=%r start=%d: %s",
                        keywords, location, start, exc,
                    )

        logger.info("LinkedInScraper: fetched %d postings", len(postings))
        return postings
