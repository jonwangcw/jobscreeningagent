"""Indeed scraper — uses the public RSS feed for job searches."""
import logging
from datetime import datetime
from urllib.parse import urlencode

import feedparser

from agent.ingest.base import RawPosting, Scraper, clean_text

logger = logging.getLogger(__name__)

# RSS endpoint for Indeed job search
_RSS_BASE = "https://www.indeed.com/rss"

# Searches to run. Each tuple is (query, location).
_SEARCHES: list[tuple[str, str]] = [
    ("machine learning engineer", "Pittsburgh, PA"),
    ("ML engineer", "Pittsburgh, PA"),
    ("data engineer", "Pittsburgh, PA"),
    ("data scientist", "Pittsburgh, PA"),
    ("applied AI engineer", "Pittsburgh, PA"),
    ("MLOps engineer", "Pittsburgh, PA"),
    ("machine learning engineer", "remote"),
    ("ML engineer", "remote"),
    ("AI safety engineer", "remote"),
    ("applied AI research", "remote"),
]


def _build_rss_url(query: str, location: str) -> str:
    params = {"q": query, "l": location, "sort": "date", "limit": "50"}
    return f"{_RSS_BASE}?{urlencode(params)}"


def _detect_remote(title: str, location: str, summary: str) -> bool | None:
    combined = f"{title} {location} {summary}".lower()
    if "remote" in combined:
        return True
    return None


class IndeedScraper(Scraper):
    def fetch(self) -> list[RawPosting]:
        postings: list[RawPosting] = []
        seen: set[str] = set()

        for query, location in _SEARCHES:
            url = _build_rss_url(query, location)
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    posting_id = entry.get("id") or entry.get("link", "")
                    if not posting_id or posting_id in seen:
                        continue
                    seen.add(posting_id)

                    title = entry.get("title", "").strip()
                    company = entry.get("author", "").strip()
                    link = entry.get("link", "").strip()
                    raw_location = entry.get("indeed_city", location).strip()
                    summary = clean_text(entry.get("summary", ""))

                    postings.append(
                        RawPosting(
                            posting_id=posting_id,
                            source="indeed",
                            company=company,
                            title=title,
                            location=raw_location,
                            remote=_detect_remote(title, raw_location, summary),
                            description=summary,
                            url=link,
                            scraped_at=datetime.utcnow(),
                        )
                    )
            except Exception as exc:
                logger.warning("IndeedScraper failed for query=%r location=%r: %s", query, location, exc)

        logger.info("IndeedScraper: fetched %d postings", len(postings))
        return postings
