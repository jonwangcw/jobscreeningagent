"""Abstract base classes and shared dataclasses for the ingest layer."""
import html
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


def clean_text(s: str) -> str:
    """Normalize raw scraper text for storage in RawPosting.description.

    All scrapers must call this on description before appending to results.
    New scrapers must do the same — this is the ingest contract.

    Operations (in order):
    1. html.unescape  — decode HTML entities
    2. Strip HTML tags (replace with space to avoid word-merging)
    3. NFKC Unicode normalization
    4. Collapse space/tab runs to single space
    5. Collapse 3+ newlines to 2
    6. strip()
    """
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


@dataclass
class RawPosting:
    posting_id: str          # stable unique ID (URL or platform ID)
    source: str              # linkedin | indeed | careers_page
    company: str
    title: str
    location: str
    remote: bool | None
    description: str
    url: str
    scraped_at: datetime = field(default_factory=datetime.utcnow)


class Scraper(ABC):
    """All scrapers implement this interface."""

    @abstractmethod
    def fetch(self) -> list[RawPosting]:
        """Fetch and return raw postings. Must not raise — return [] on failure."""
        ...
