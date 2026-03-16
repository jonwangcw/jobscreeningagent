"""Playwright + LLM agentic scraper for JavaScript-rendered job portals.

Architecture
------------
PlaywrightLLMScraper implements the Scraper ABC. fetch() is synchronous and
bridges to async Playwright via asyncio.run().

The scraper uses an LLM to:
1. Decide how to navigate/interact with an unknown portal (explore phase)
2. Extract structured job listings from rendered page snapshots (extract phase)
3. Filter extracted titles for relevance (filter phase)

All prompt strings live in agent/ingest/portal_prompts.py — none here.

Portal types (from career_pages.txt third column):
    workday, eightfold, greenhouse, phenom, brassring, taleo,
    talentbrew, custom_url_params, unknown

Config keys read from llm_config dict (passed from config.yml llm section):
    provider, model, max_tokens

No hardcoded model names, thresholds, or paths.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator

from agent.ingest.base import RawPosting, Scraper, clean_text
from agent.ingest.careers_page import PortalConfig
from agent.ingest.portal_prompts import (
    EIGHTFOLD_EXTRACT_SYSTEM_PROMPT,
    EIGHTFOLD_EXTRACT_USER_PROMPT,
    EXPLORE_PORTAL_SYSTEM_PROMPT,
    EXPLORE_PORTAL_USER_PROMPT,
    EXTRACT_JOBS_SYSTEM_PROMPT,
    EXTRACT_JOBS_USER_PROMPT,
    FILTER_JOBS_SYSTEM_PROMPT,
    FILTER_JOBS_USER_PROMPT,
    WORKDAY_EXPLORE_SYSTEM_PROMPT,
    WORKDAY_EXPLORE_USER_PROMPT,
)
from agent.scoring.llm_scorer import LLMBackend, build_llm_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default keyword list — used when portal entry has no keywords= column
# ---------------------------------------------------------------------------
_DEFAULT_KEYWORDS = [
    "machine learning",
    "data engineer",
    "AI engineer",
    "data scientist",
    "MLOps",
    "applied AI",
    "quantitative",
    "LLM",
]

# Maximum pages to paginate through per portal entry
_MAX_PAGES = 10

# Maximum exploration steps before giving up on unknown portals
_MAX_EXPLORE_STEPS = 8

# Playwright timeout (ms) for navigation and element waits
_NAV_TIMEOUT_MS = 30_000
_WAIT_TIMEOUT_MS = 10_000

# Snapshot character limit sent to LLM (keeps token costs bounded)
_SNAPSHOT_MAX_CHARS = 12_000


# ---------------------------------------------------------------------------
# Pydantic models for LLM response validation
# ---------------------------------------------------------------------------


class JobItem(BaseModel):
    """A single job listing extracted by the LLM."""

    title: str
    url: str | None = None
    location: str | None = None
    remote: bool | None = None

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty")
        return v

    @field_validator("url")
    @classmethod
    def url_nonempty_or_none(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            return v if v else None
        return None


class ExtractJobsResponse(BaseModel):
    """Validated response from EXTRACT_JOBS_SYSTEM_PROMPT."""

    jobs: list[JobItem] = Field(default_factory=list)
    has_next_page: bool = False


class ExploreAction(BaseModel):
    """Validated response from EXPLORE_PORTAL_SYSTEM_PROMPT."""

    action: str  # click | type_and_search | navigate | extract | pagination_next | done
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    reasoning: str = ""

    @field_validator("action")
    @classmethod
    def action_valid(cls, v: str) -> str:
        valid = {"click", "type_and_search", "navigate", "extract", "pagination_next", "done"}
        if v not in valid:
            raise ValueError(f"action must be one of {valid}, got {v!r}")
        return v


class FilterJobsResponse(BaseModel):
    """Validated response from FILTER_JOBS_SYSTEM_PROMPT."""

    relevant_indices: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Exploration result cache (in-process, lives for the duration of a run)
# ---------------------------------------------------------------------------


@dataclass
class _ExploreCache:
    """Caches per-portal-type exploration paths discovered during a run.

    Key: portal_type string.
    Value: list of ExploreAction that successfully led to job listings.

    If a portal type was already explored successfully, the cached action
    sequence is replayed instead of asking the LLM to re-explore. This saves
    LLM calls when multiple entries share the same portal type (e.g. two
    Workday tenants).
    """

    _store: dict[str, list[ExploreAction]] = field(default_factory=dict)

    def get(self, portal_type: str) -> list[ExploreAction] | None:
        return self._store.get(portal_type)

    def put(self, portal_type: str, actions: list[ExploreAction]) -> None:
        self._store[portal_type] = actions

    def has(self, portal_type: str) -> bool:
        return portal_type in self._store


# Module-level cache shared across all PlaywrightLLMScraper instances in a run
_explore_cache = _ExploreCache()


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _build_snapshot(html: str, max_chars: int = _SNAPSHOT_MAX_CHARS) -> str:
    """Convert rendered HTML to a compact plain-text snapshot for the LLM.

    Strategy:
    1. Strip script/style/noscript/svg/img tags completely
    2. Replace block-level elements with newlines for structure
    3. Extract remaining text
    4. Collapse whitespace
    5. Truncate to max_chars

    This keeps the snapshot token-efficient while preserving the structural
    cues (headings, links, list items) the LLM needs to understand the page.
    """
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags entirely
    for tag in soup.find_all(["script", "style", "noscript", "svg", "img", "meta", "link"]):
        tag.decompose()

    # Inject newlines around block elements so words don't merge
    block_tags = {
        "div", "p", "li", "tr", "td", "th", "section", "article",
        "header", "footer", "nav", "aside", "main", "h1", "h2",
        "h3", "h4", "h5", "h6", "br", "hr",
    }
    for tag in soup.find_all(block_tags):
        if isinstance(tag, Tag):
            tag.insert_before("\n")
            tag.insert_after("\n")

    text = soup.get_text(separator=" ")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    return text


# ---------------------------------------------------------------------------
# URL manipulation helpers
# ---------------------------------------------------------------------------


def _resolve_url(href: str, base_url: str) -> str:
    """Make href absolute using base_url."""
    if not href:
        return base_url
    if href.startswith("http"):
        return href
    parsed_base = urlparse(base_url)
    if href.startswith("//"):
        return f"{parsed_base.scheme}:{href}"
    if href.startswith("/"):
        return f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
    # Relative path
    base_path = parsed_base.path.rsplit("/", 1)[0]
    return f"{parsed_base.scheme}://{parsed_base.netloc}{base_path}/{href}"


def _inject_keywords_into_url(url: str, keyword: str) -> str:
    """Replace __KEYWORDS__ placeholder in URL with the given keyword."""
    return url.replace("__KEYWORDS__", keyword)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


def _llm_extract_jobs(
    llm: LLMBackend,
    snapshot: str,
    company: str,
    portal_type: str,
    base_url: str,
    keywords: list[str],
) -> ExtractJobsResponse:
    """Call LLM to extract jobs from a page snapshot. Returns validated response."""
    user = EXTRACT_JOBS_USER_PROMPT.format(
        company=company,
        portal_type=portal_type,
        base_url=base_url,
        keywords=", ".join(keywords),
        snapshot=snapshot,
    )
    try:
        raw = llm.complete(system=EXTRACT_JOBS_SYSTEM_PROMPT, user=user)
        data = json.loads(raw)
        return ExtractJobsResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "PlaywrightLLMScraper: LLM extract_jobs parse error for %s: %s", company, exc
        )
        return ExtractJobsResponse()


def _llm_explore_portal(
    llm: LLMBackend,
    snapshot: str,
    company: str,
    portal_type: str,
    current_url: str,
    keywords: list[str],
) -> ExploreAction:
    """Call LLM to decide the next navigation action. Returns validated action."""
    if portal_type == "workday":
        system = WORKDAY_EXPLORE_SYSTEM_PROMPT
        user = WORKDAY_EXPLORE_USER_PROMPT.format(
            company=company,
            current_url=current_url,
            keywords=", ".join(keywords),
            snapshot=snapshot,
        )
    elif portal_type == "eightfold":
        # Eightfold goes straight to extraction (REST API is exposed)
        system = EIGHTFOLD_EXTRACT_SYSTEM_PROMPT
        user = EIGHTFOLD_EXTRACT_USER_PROMPT.format(
            company=company,
            base_url=current_url,
            keywords=", ".join(keywords),
            snapshot=snapshot,
        )
    else:
        system = EXPLORE_PORTAL_SYSTEM_PROMPT
        user = EXPLORE_PORTAL_USER_PROMPT.format(
            company=company,
            portal_type=portal_type,
            current_url=current_url,
            keywords=", ".join(keywords),
            snapshot=snapshot,
        )
    try:
        raw = llm.complete(system=system, user=user)
        data = json.loads(raw)
        return ExploreAction.model_validate(data)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "PlaywrightLLMScraper: LLM explore_portal parse error for %s: %s", company, exc
        )
        return ExploreAction(action="done", reasoning=f"parse error: {exc}")


def _llm_filter_jobs(
    llm: LLMBackend,
    titles: list[str],
) -> list[int]:
    """Call LLM to filter job titles for relevance. Returns list of relevant indices."""
    if not titles:
        return []
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(titles))
    user = FILTER_JOBS_USER_PROMPT.format(titles_numbered=numbered)
    try:
        raw = llm.complete(system=FILTER_JOBS_SYSTEM_PROMPT, user=user)
        data = json.loads(raw)
        resp = FilterJobsResponse.model_validate(data)
        return resp.relevant_indices
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("PlaywrightLLMScraper: LLM filter_jobs parse error: %s", exc)
        # On failure, return all indices (be inclusive)
        return list(range(len(titles)))


# ---------------------------------------------------------------------------
# Per-mechanism query executors
# ---------------------------------------------------------------------------


async def _execute_workday(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Workday ATS portals via LLM-guided exploration."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    for _ in range(_MAX_EXPLORE_STEPS):
        html = await page.content()
        snapshot = _build_snapshot(html)
        current_url = page.url

        action = _llm_explore_portal(llm, snapshot, company, "workday", current_url, keywords)
        logger.debug("Workday explore action for %s: %s — %s", company, action.action, action.reasoning)

        if action.action == "extract":
            jobs_resp = _llm_extract_jobs(llm, snapshot, company, "workday", current_url, keywords)
            all_jobs.extend(jobs_resp.jobs)
            if not jobs_resp.has_next_page:
                break
            # Attempt to click next page
            try:
                next_btn = await page.query_selector("button[aria-label*='next' i], a[aria-label*='next' i]")
                if next_btn:
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                else:
                    break
            except Exception:
                break

        elif action.action == "done":
            break

        elif action.action == "navigate" and action.url:
            await page.goto(action.url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

        elif action.action == "click" and action.selector:
            try:
                await page.click(action.selector, timeout=_WAIT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            except Exception as exc:
                logger.warning("Workday click failed for %s selector=%r: %s", company, action.selector, exc)
                break

        elif action.action == "type_and_search" and action.selector and action.value:
            try:
                await page.fill(action.selector, action.value, timeout=_WAIT_TIMEOUT_MS)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            except Exception as exc:
                logger.warning("Workday search failed for %s: %s", company, exc)
                break

        else:
            break

    return all_jobs


async def _execute_eightfold(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Eightfold AI portals. Eightfold exposes a REST API.

    Strategy: navigate to the portal, intercept the JSON from the Eightfold
    API endpoint, then extract from there. Fall back to LLM snapshot extraction
    if the API response isn't captured.
    """
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []
    api_responses: list[dict] = []

    async def handle_response(response: Any) -> None:
        """Intercept Eightfold API responses."""
        if "api/apply/v2/jobs" in response.url or "position_list" in response.url:
            try:
                body = await response.json()
                if isinstance(body, dict):
                    api_responses.append(body)
            except Exception:
                pass

    page.on("response", handle_response)

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    # Try to use search if available
    for keyword in keywords[:3]:  # limit to 3 keyword searches to control cost
        try:
            search_input = await page.query_selector(
                "input[placeholder*='search' i], input[aria-label*='search' i], input[type='search']"
            )
            if search_input:
                await search_input.fill(keyword)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                await page.wait_for_timeout(2000)
        except Exception:
            break

    # If we captured API responses, parse them
    for resp_body in api_responses:
        positions = resp_body.get("positions") or resp_body.get("data") or []
        for pos in positions:
            title = pos.get("name") or pos.get("title") or ""
            job_id = pos.get("id") or pos.get("job_id") or ""
            loc = pos.get("location") or pos.get("city") or ""
            if not title:
                continue
            parsed = urlparse(url)
            job_url = f"{parsed.scheme}://{parsed.netloc}/careers/{job_id}" if job_id else None
            remote_flag = "remote" in str(loc).lower() or None
            all_jobs.append(JobItem(title=title, url=job_url, location=loc, remote=remote_flag))

    # If no API data captured, fall back to LLM snapshot
    if not all_jobs:
        html = await page.content()
        snapshot = _build_snapshot(html)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "eightfold", url, keywords)
        all_jobs.extend(jobs_resp.jobs)

    return all_jobs


async def _execute_taleo(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Taleo portals. Taleo renders a table of jobs after page load."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    # Taleo typically renders a job results table — extract directly
    for _page_num in range(_MAX_PAGES):
        html = await page.content()
        snapshot = _build_snapshot(html)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "taleo", url, keywords)
        all_jobs.extend(jobs_resp.jobs)

        if not jobs_resp.has_next_page:
            break

        # Try next page button
        try:
            next_btn = await page.query_selector(
                "a[title*='Next' i], button[title*='Next' i], a[aria-label*='next' i]"
            )
            if next_btn:
                await next_btn.click()
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            else:
                break
        except Exception:
            break

    return all_jobs


async def _execute_greenhouse(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Greenhouse ATS portals.

    Greenhouse board pages are mostly static — full job list is in the HTML.
    """
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("domcontentloaded", timeout=_WAIT_TIMEOUT_MS)

    html = await page.content()
    snapshot = _build_snapshot(html)
    jobs_resp = _llm_extract_jobs(llm, snapshot, company, "greenhouse", url, keywords)
    all_jobs.extend(jobs_resp.jobs)

    return all_jobs


async def _execute_phenom(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Phenom People portals. Phenom is a JS-heavy SPA."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    for keyword in keywords[:4]:
        try:
            search_input = await page.query_selector(
                "input[placeholder*='keyword' i], input[placeholder*='search' i], input[type='search']"
            )
            if search_input:
                await search_input.fill("")
                await search_input.type(keyword, delay=50)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                await page.wait_for_timeout(2000)

                html = await page.content()
                snapshot = _build_snapshot(html)
                jobs_resp = _llm_extract_jobs(llm, snapshot, company, "phenom", url, keywords)
                all_jobs.extend(jobs_resp.jobs)
        except Exception as exc:
            logger.warning("Phenom search failed for %s keyword=%r: %s", company, keyword, exc)

    return all_jobs


async def _execute_brassring(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle Brassring (Kenexa) ATS portals — Angular SPA."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
    await page.wait_for_timeout(3000)  # Brassring needs extra settle time

    html = await page.content()
    snapshot = _build_snapshot(html)

    # Use LLM to figure out how to search
    action = _llm_explore_portal(llm, snapshot, company, "brassring", page.url, keywords)

    if action.action == "type_and_search" and action.selector and action.value:
        try:
            await page.fill(action.selector, action.value)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
        except Exception as exc:
            logger.warning("Brassring search failed for %s: %s", company, exc)

    html = await page.content()
    snapshot = _build_snapshot(html)
    jobs_resp = _llm_extract_jobs(llm, snapshot, company, "brassring", page.url, keywords)
    all_jobs.extend(jobs_resp.jobs)

    return all_jobs


async def _execute_talentbrew(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle TalentBrew portals. URL is typically pre-filtered; just extract."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    for _page_num in range(_MAX_PAGES):
        html = await page.content()
        snapshot = _build_snapshot(html)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "talentbrew", page.url, keywords)
        all_jobs.extend(jobs_resp.jobs)

        if not jobs_resp.has_next_page:
            break

        try:
            next_btn = await page.query_selector("a.next-page, a[aria-label*='Next' i], button.next")
            if next_btn:
                await next_btn.click()
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            else:
                break
        except Exception:
            break

    return all_jobs


async def _execute_custom_url_params(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle portals where keyword filtering works via URL parameters.

    If the URL contains __KEYWORDS__ placeholder, iterate over each keyword,
    replacing the placeholder and loading each filtered page.
    Otherwise load the URL as-is and extract.
    """
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []
    seen_titles: set[str] = set()

    has_placeholder = "__KEYWORDS__" in url

    search_targets = keywords if has_placeholder else [""]

    for keyword in search_targets:
        target_url = _inject_keywords_into_url(url, keyword) if has_placeholder else url
        try:
            await page.goto(target_url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            await page.wait_for_timeout(1500)

            html = await page.content()
            snapshot = _build_snapshot(html)
            jobs_resp = _llm_extract_jobs(
                llm, snapshot, company, "custom_url_params", page.url, keywords
            )
            for job in jobs_resp.jobs:
                if job.title not in seen_titles:
                    seen_titles.add(job.title)
                    all_jobs.append(job)
        except Exception as exc:
            logger.warning(
                "custom_url_params failed for %s url=%r: %s", company, target_url, exc
            )

    return all_jobs


async def _execute_unknown(
    page: Any,
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[JobItem]:
    """Handle unknown portals via fully LLM-guided exploration."""
    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    all_jobs: list[JobItem] = []
    explore_actions: list[ExploreAction] = []

    # Check cache for previously successful exploration paths
    cached = _explore_cache.get(portal_config.portal_type + ":" + url)
    if cached:
        logger.info("PlaywrightLLMScraper: using cached explore path for %s", company)
        actions_to_replay = cached
    else:
        actions_to_replay = []

    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    step = 0
    while step < _MAX_EXPLORE_STEPS:
        html = await page.content()
        snapshot = _build_snapshot(html)

        # Use cached action if available for this step
        if actions_to_replay and step < len(actions_to_replay):
            action = actions_to_replay[step]
        else:
            action = _llm_explore_portal(
                llm, snapshot, company, portal_config.portal_type, page.url, keywords
            )

        explore_actions.append(action)
        logger.debug(
            "Unknown portal explore step %d for %s: %s — %s",
            step, company, action.action, action.reasoning,
        )

        if action.action == "extract":
            jobs_resp = _llm_extract_jobs(
                llm, snapshot, company, portal_config.portal_type, page.url, keywords
            )
            all_jobs.extend(jobs_resp.jobs)
            # Cache the successful exploration path
            _explore_cache.put(portal_config.portal_type + ":" + url, explore_actions)
            if not jobs_resp.has_next_page:
                break
            # Try to find next page
            step += 1
            try:
                next_btn = await page.query_selector(
                    "a[aria-label*='next' i], button[aria-label*='next' i], a.next, button.next"
                )
                if next_btn:
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                else:
                    break
            except Exception:
                break
            continue

        elif action.action == "done":
            break

        elif action.action == "navigate" and action.url:
            await page.goto(action.url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

        elif action.action == "click" and action.selector:
            try:
                await page.click(action.selector, timeout=_WAIT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            except Exception as exc:
                logger.warning(
                    "Unknown portal click failed for %s selector=%r: %s",
                    company, action.selector, exc
                )
                break

        elif action.action == "type_and_search" and action.selector and action.value:
            try:
                await page.fill(action.selector, action.value, timeout=_WAIT_TIMEOUT_MS)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            except Exception as exc:
                logger.warning(
                    "Unknown portal search failed for %s: %s", company, exc
                )
                break

        elif action.action == "pagination_next":
            try:
                next_btn = await page.query_selector(
                    "a[aria-label*='next' i], button[aria-label*='next' i]"
                )
                if next_btn:
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                else:
                    break
            except Exception:
                break

        step += 1

    return all_jobs


# ---------------------------------------------------------------------------
# Mechanism dispatcher
# ---------------------------------------------------------------------------

_MECHANISM_MAP = {
    "workday": _execute_workday,
    "eightfold": _execute_eightfold,
    "taleo": _execute_taleo,
    "greenhouse": _execute_greenhouse,
    "phenom": _execute_phenom,
    "brassring": _execute_brassring,
    "talentbrew": _execute_talentbrew,
    "custom_url_params": _execute_custom_url_params,
    "unknown": _execute_unknown,
}


async def _scrape_portal_async(
    url: str,
    company: str,
    portal_config: PortalConfig,
    llm: LLMBackend,
) -> list[RawPosting]:
    """Async core: launch Playwright, dispatch to the correct mechanism, return postings."""
    from playwright.async_api import async_playwright

    keywords = portal_config.keywords or _DEFAULT_KEYWORDS
    execute_fn = _MECHANISM_MAP.get(portal_config.portal_type, _execute_unknown)

    postings: list[RawPosting] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        try:
            job_items = await execute_fn(page, url, company, portal_config, llm)
        except Exception as exc:
            logger.warning(
                "PlaywrightLLMScraper: %s portal execution failed for %s: %s",
                portal_config.portal_type, company, exc,
            )
            job_items = []
        finally:
            await browser.close()

    if not job_items:
        return postings

    # LLM relevance filter on titles
    titles = [j.title for j in job_items]
    relevant_indices = _llm_filter_jobs(llm, titles)
    relevant_set = set(relevant_indices)

    for i, item in enumerate(job_items):
        if i not in relevant_set:
            continue

        job_url = _resolve_url(item.url or "", url) if item.url else url
        # Strip query params for stable dedup ID, unless URL has no path info
        parsed = urlparse(job_url)
        if parsed.path and parsed.path not in ("/", ""):
            posting_id = urlunparse(parsed._replace(query="", fragment=""))
        else:
            posting_id = job_url

        remote_val: bool | None = item.remote
        if remote_val is None and item.location:
            remote_val = True if "remote" in item.location.lower() else None

        postings.append(
            RawPosting(
                posting_id=posting_id,
                source="careers_page",
                company=company,
                title=clean_text(item.title)[:200],
                location=item.location or "",
                remote=remote_val,
                description=clean_text(""),  # full JD fetched lazily if posting passes gates
                url=job_url,
                scraped_at=datetime.utcnow(),
            )
        )

    return postings


# ---------------------------------------------------------------------------
# Public Scraper class
# ---------------------------------------------------------------------------


class PlaywrightLLMScraper(Scraper):
    """Scrapes a single JS-rendered career portal using Playwright + LLM guidance.

    One instance per portal entry in career_pages.txt. The fetch() method is
    synchronous (required by the Scraper ABC) and uses asyncio.run() internally
    to drive the async Playwright logic.

    Parameters
    ----------
    url:
        The starting URL for this portal (from career_pages.txt).
    company:
        Company name string (from career_pages.txt).
    portal_config:
        Parsed PortalConfig (portal_type + keywords).
    llm_config:
        Dict from config.yml llm section — passed to build_llm_backend().
    """

    def __init__(
        self,
        url: str,
        company: str,
        portal_config: PortalConfig,
        llm_config: dict[str, Any],
    ) -> None:
        self._url = url
        self._company = company
        self._portal_config = portal_config
        self._llm_config = llm_config

    def fetch(self) -> list[RawPosting]:
        """Synchronous fetch — bridges to async via asyncio.run()."""
        try:
            llm = build_llm_backend(self._llm_config)
        except Exception as exc:
            logger.warning(
                "PlaywrightLLMScraper: failed to build LLM backend for %s: %s",
                self._company, exc,
            )
            return []

        try:
            postings = asyncio.run(
                _scrape_portal_async(
                    url=self._url,
                    company=self._company,
                    portal_config=self._portal_config,
                    llm=llm,
                )
            )
            logger.info(
                "PlaywrightLLMScraper: %s (%s) → %d postings",
                self._company, self._portal_config.portal_type, len(postings),
            )
            return postings
        except Exception as exc:
            logger.warning(
                "PlaywrightLLMScraper: %s raised an unexpected exception: %s",
                self._company, exc,
            )
            return []
