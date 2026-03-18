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
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator

from agent.ingest.base import RawPosting, Scraper, clean_text
from agent.ingest.careers_page import PortalConfig
from agent.ingest.portal_prompts import (
    BRASSRING_EXPLORE_SYSTEM_PROMPT,
    BRASSRING_EXPLORE_USER_PROMPT,
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
# Structured trace logging
# ---------------------------------------------------------------------------


class ScrapeTrace:
    """Thread-safe JSONL trace writer for the Playwright scraper.

    Each event is one JSON object per line with a 'ts' timestamp, 'company',
    'portal', 'event' fields, and event-specific data.

    Disabled (all methods are no-ops) when path is None.

    Event types:
        navigate  — before every page.goto()
        snapshot  — full rendered text after every _build_snapshot() call
        llm_call  — full prompt + raw response after every LLM .complete() call
        action    — each action attempted (type_and_search / click / navigate)
        error     — exceptions in executor functions with context
        result    — jobs_raw / jobs_after_filter at end of _scrape_portal_async
    """

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def _write(self, entry: dict) -> None:
        if not self._path:
            return
        entry["ts"] = datetime.utcnow().isoformat()
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def navigate(self, company: str, portal: str, url: str) -> None:
        self._write({"event": "navigate", "company": company, "portal": portal, "url": url})

    def snapshot(self, company: str, portal: str, url: str, text: str) -> None:
        self._write({
            "event": "snapshot",
            "company": company,
            "portal": portal,
            "url": url,
            "chars": len(text),
            "text": text,
        })

    def llm_call(
        self,
        company: str,
        portal: str,
        call_type: str,
        url: str,
        user_prompt: str,
        raw_response: str,
        valid_json: bool,
        parsed: Any = None,
    ) -> None:
        self._write({
            "event": "llm_call",
            "company": company,
            "portal": portal,
            "call_type": call_type,
            "url": url,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "valid_json": valid_json,
            "parsed": parsed,
        })

    def action(
        self,
        company: str,
        portal: str,
        url: str,
        action_type: str,
        selector: str | None,
        value: str | None,
        reasoning: str,
        status: str,
        error: str | None = None,
    ) -> None:
        self._write({
            "event": "action",
            "company": company,
            "portal": portal,
            "url": url,
            "action_type": action_type,
            "selector": selector,
            "value": value,
            "reasoning": reasoning,
            "status": status,
            "error": error,
        })

    def error(self, company: str, portal: str, url: str, error_type: str, **ctx: Any) -> None:
        self._write({
            "event": "error",
            "company": company,
            "portal": portal,
            "url": url,
            "error_type": error_type,
            **ctx,
        })

    def result(self, company: str, portal: str, jobs_raw: int, jobs_after_filter: int) -> None:
        self._write({
            "event": "result",
            "company": company,
            "portal": portal,
            "jobs_raw": jobs_raw,
            "jobs_after_filter": jobs_after_filter,
        })


# Module-level trace instance — no-op by default, configured from main.py
_trace: ScrapeTrace = ScrapeTrace(None)


def configure_trace(path: str) -> None:
    """Enable structured JSONL trace logging to the given file path.

    Called from main.py when playwright.trace_log is set in config.yml.
    The file is created (or appended to) on first write.
    Safe to call multiple times — replaces the previous instance.
    """
    global _trace
    _trace = ScrapeTrace(path)


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


def _is_css_selector(s: str) -> bool:
    """Return True if s looks like a CSS selector rather than a visible text label.

    CSS selectors start with #, ., [, :, *, or a tag name (letter). A string
    that starts with a letter followed by spaces or dashes is most likely a
    visible text label returned by the LLM — those must be handled with
    get_by_text() instead of page.click(selector).
    """
    if not s:
        return False
    # If it contains spaces or dashes early on, it's a text label
    first_word = s.split()[0] if " " in s else s
    if re.search(r'[\s]', s) or (re.match(r'^[a-zA-Z]', s) and "-" in first_word):
        return False
    return bool(re.match(r'^[#.\[:\*a-zA-Z]', s))


async def _smart_click(page: Any, selector: str, timeout: int) -> None:
    """Click an element identified by a CSS selector or visible text label."""
    if _is_css_selector(selector):
        await page.click(selector, timeout=timeout)
    else:
        await page.get_by_text(selector, exact=False).first.click(timeout=timeout)


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
    """Replace __KEYWORDS__ placeholder in URL with the URL-encoded keyword."""
    return url.replace("__KEYWORDS__", quote_plus(keyword))


# ---------------------------------------------------------------------------
# LLM JSON parse helper
# ---------------------------------------------------------------------------


def _parse_llm_json(raw: str, context: str) -> Any:
    """Parse a JSON string returned by the LLM, stripping markdown fences if present.

    Returns the parsed object, or raises json.JSONDecodeError on failure.
    Logs the first 300 chars of the raw response on parse failure so the caller
    can see exactly what the LLM sent.

    context: a short label for warning messages (e.g. "extract_jobs/UPMC").
    """
    if not raw or not raw.strip():
        raise json.JSONDecodeError("empty response", "", 0)

    text = raw.strip()

    # Strip markdown code fences: ```json\n...\n``` or ```\n...\n```
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "PlaywrightLLMScraper: JSON parse failure [%s] — raw response (first 300 chars): %r",
            context,
            raw[:300],
        )
        raise


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
    current_url: str = "",
) -> ExtractJobsResponse:
    """Call LLM to extract jobs from a page snapshot. Returns validated response."""
    trace_url = current_url or base_url
    user = EXTRACT_JOBS_USER_PROMPT.format(
        company=company,
        portal_type=portal_type,
        base_url=base_url,
        keywords=", ".join(keywords),
        snapshot=snapshot,
    )
    raw = ""
    try:
        raw = llm.complete(system=EXTRACT_JOBS_SYSTEM_PROMPT, user=user, prefill="{")
        data = _parse_llm_json(raw, f"extract_jobs/{company}")
        result = ExtractJobsResponse.model_validate(data)
        _trace.llm_call(company, portal_type, "extract", trace_url, user, raw, True, data)
        return result
    except (json.JSONDecodeError, ValueError, KeyError):
        _trace.llm_call(company, portal_type, "extract", trace_url, user, raw, False, None)
        return ExtractJobsResponse()


def _llm_explore_portal(
    llm: LLMBackend,
    snapshot: str,
    company: str,
    portal_type: str,
    current_url: str,
    keywords: list[str],
    failed_selectors: list[str] | None = None,
) -> ExploreAction:
    """Call LLM to decide the next navigation action. Returns validated action."""
    if portal_type == "workday":
        system = WORKDAY_EXPLORE_SYSTEM_PROMPT
        failed_str = ", ".join(failed_selectors) if failed_selectors else "none"
        user = WORKDAY_EXPLORE_USER_PROMPT.format(
            company=company,
            current_url=current_url,
            keywords=", ".join(keywords),
            failed_selectors=failed_str,
            snapshot=snapshot,
        )
    elif portal_type == "brassring":
        system = BRASSRING_EXPLORE_SYSTEM_PROMPT
        failed_str = ", ".join(failed_selectors) if failed_selectors else "none"
        user = BRASSRING_EXPLORE_USER_PROMPT.format(
            company=company,
            current_url=current_url,
            keywords=", ".join(keywords),
            failed_selectors=failed_str,
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
    raw = ""
    try:
        raw = llm.complete(system=system, user=user, prefill="{")
        data = _parse_llm_json(raw, f"explore_portal/{company}")
        result = ExploreAction.model_validate(data)
        _trace.llm_call(company, portal_type, "explore", current_url, user, raw, True, data)
        return result
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        _trace.llm_call(company, portal_type, "explore", current_url, user, raw, False, None)
        return ExploreAction(action="done", reasoning=f"parse error: {exc}")


def _llm_filter_jobs(
    llm: LLMBackend,
    titles: list[str],
    company: str = "",
    portal_type: str = "",
    current_url: str = "",
) -> list[int]:
    """Call LLM to filter job titles for relevance. Returns list of relevant indices."""
    if not titles:
        return []
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(titles))
    user = FILTER_JOBS_USER_PROMPT.format(titles_numbered=numbered)
    raw = ""
    try:
        raw = llm.complete(system=FILTER_JOBS_SYSTEM_PROMPT, user=user, prefill="{")
        data = _parse_llm_json(raw, "filter_jobs")
        resp = FilterJobsResponse.model_validate(data)
        _trace.llm_call(company, portal_type, "filter", current_url, user, raw, True, data)
        return resp.relevant_indices
    except (json.JSONDecodeError, ValueError, KeyError):
        _trace.llm_call(company, portal_type, "filter", current_url, user, raw, False, None)
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
    failed_selectors: list[str] = []

    _trace.navigate(company, "workday", url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    for _ in range(_MAX_EXPLORE_STEPS):
        html = await page.content()
        snapshot = _build_snapshot(html)
        current_url = page.url
        _trace.snapshot(company, "workday", current_url, snapshot)

        action = _llm_explore_portal(
            llm, snapshot, company, "workday", current_url, keywords,
            failed_selectors=failed_selectors,
        )
        logger.info("Workday [%s] step: %s — %s", company, action.action, action.reasoning)

        if action.action == "extract":
            jobs_resp = _llm_extract_jobs(
                llm, snapshot, company, "workday", current_url, keywords,
                current_url=current_url,
            )
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
            _trace.navigate(company, "workday", action.url)
            await page.goto(action.url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

        elif action.action == "click" and action.selector:
            try:
                await _smart_click(page, action.selector, _WAIT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, "workday", current_url, "click", action.selector, None, action.reasoning, "ok")
            except Exception as exc:
                logger.warning("Workday click failed for %s selector=%r: %s", company, action.selector, exc)
                _trace.action(company, "workday", current_url, "click", action.selector, None, action.reasoning, "exception", error=str(exc))
                break

        elif action.action == "type_and_search" and action.selector and action.value:
            try:
                el = await page.query_selector(action.selector)
                if el is None:
                    logger.warning(
                        "Workday selector not found for %s: %r — adding to failed list",
                        company, action.selector,
                    )
                    _trace.action(company, "workday", current_url, "type_and_search", action.selector, action.value, action.reasoning, "selector_not_found")
                    failed_selectors.append(action.selector)
                    continue
                await page.fill(action.selector, action.value, timeout=_WAIT_TIMEOUT_MS)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, "workday", current_url, "type_and_search", action.selector, action.value, action.reasoning, "ok")
            except Exception as exc:
                logger.warning("Workday search failed for %s: %s", company, exc)
                _trace.action(company, "workday", current_url, "type_and_search", action.selector, action.value, action.reasoning, "exception", error=str(exc))
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

    _trace.navigate(company, "eightfold", url)
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
                _trace.action(company, "eightfold", page.url, "type_and_search", "search_input", keyword, "keyword search", "ok")
        except Exception as exc:
            _trace.error(company, "eightfold", page.url, "search_failed", keyword=keyword, exception=str(exc))
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
        _trace.snapshot(company, "eightfold", page.url, snapshot)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "eightfold", url, keywords, current_url=page.url)
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

    _trace.navigate(company, "taleo", url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    # Taleo typically renders a job results table — extract directly
    for _page_num in range(_MAX_PAGES):
        html = await page.content()
        snapshot = _build_snapshot(html)
        _trace.snapshot(company, "taleo", page.url, snapshot)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "taleo", url, keywords, current_url=page.url)
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

    _trace.navigate(company, "greenhouse", url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("domcontentloaded", timeout=_WAIT_TIMEOUT_MS)

    html = await page.content()
    snapshot = _build_snapshot(html)
    _trace.snapshot(company, "greenhouse", page.url, snapshot)
    jobs_resp = _llm_extract_jobs(llm, snapshot, company, "greenhouse", url, keywords, current_url=page.url)
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

    _trace.navigate(company, "phenom", url)
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
                _trace.action(company, "phenom", page.url, "type_and_search", "search_input", keyword, "keyword search", "ok")

                html = await page.content()
                snapshot = _build_snapshot(html)
                _trace.snapshot(company, "phenom", page.url, snapshot)
                jobs_resp = _llm_extract_jobs(llm, snapshot, company, "phenom", page.url, keywords, current_url=page.url)
                all_jobs.extend(jobs_resp.jobs)
        except Exception as exc:
            logger.warning("Phenom search failed for %s keyword=%r: %s", company, keyword, exc)
            _trace.error(company, "phenom", page.url, "search_failed", keyword=keyword, exception=str(exc))

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
    failed_selectors: list[str] = []

    _trace.navigate(company, "brassring", url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
    await page.wait_for_timeout(3000)  # Brassring needs extra settle time

    html = await page.content()
    snapshot = _build_snapshot(html)
    _trace.snapshot(company, "brassring", page.url, snapshot)

    # Use LLM to decide how to search; retry if selector not found
    for _ in range(_MAX_EXPLORE_STEPS):
        action = _llm_explore_portal(llm, snapshot, company, "brassring", page.url, keywords, failed_selectors=failed_selectors)

        if action.action in ("extract", "done"):
            break

        if action.action == "type_and_search" and action.selector and action.value:
            try:
                el = await page.query_selector(action.selector)
                if el is None:
                    logger.warning("Brassring selector not found for %s: %r", company, action.selector)
                    _trace.action(company, "brassring", page.url, "type_and_search", action.selector, action.value, action.reasoning, "selector_not_found")
                    if action.selector not in failed_selectors:
                        failed_selectors.append(action.selector)
                    continue
                await page.fill(action.selector, action.value, timeout=_WAIT_TIMEOUT_MS)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, "brassring", page.url, "type_and_search", action.selector, action.value, action.reasoning, "ok")
                break
            except Exception as exc:
                logger.warning("Brassring search failed for %s: %s", company, exc)
                _trace.action(company, "brassring", page.url, "type_and_search", action.selector, action.value, action.reasoning, "exception", error=str(exc))
                break

        elif action.action == "click" and action.selector:
            try:
                el = await page.query_selector(action.selector)
                if el is None:
                    _trace.action(company, "brassring", page.url, "click", action.selector, None, action.reasoning, "selector_not_found")
                    if action.selector not in failed_selectors:
                        failed_selectors.append(action.selector)
                    continue
                await page.click(action.selector, timeout=_WAIT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, "brassring", page.url, "click", action.selector, None, action.reasoning, "ok")
                html = await page.content()
                snapshot = _build_snapshot(html)
                _trace.snapshot(company, "brassring", page.url, snapshot)
            except Exception as exc:
                logger.warning("Brassring click failed for %s: %s", company, exc)
                _trace.action(company, "brassring", page.url, "click", action.selector, None, action.reasoning, "exception", error=str(exc))
                break
        else:
            break

    html = await page.content()
    snapshot = _build_snapshot(html)
    _trace.snapshot(company, "brassring", page.url, snapshot)
    jobs_resp = _llm_extract_jobs(llm, snapshot, company, "brassring", page.url, keywords, current_url=page.url)
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

    _trace.navigate(company, "talentbrew", url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

    for _page_num in range(_MAX_PAGES):
        html = await page.content()
        snapshot = _build_snapshot(html)
        _trace.snapshot(company, "talentbrew", page.url, snapshot)
        jobs_resp = _llm_extract_jobs(llm, snapshot, company, "talentbrew", page.url, keywords, current_url=page.url)
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
            _trace.navigate(company, "custom_url_params", target_url)
            await page.goto(target_url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
            await page.wait_for_timeout(1500)

            html = await page.content()
            snapshot = _build_snapshot(html)
            _trace.snapshot(company, "custom_url_params", page.url, snapshot)
            jobs_resp = _llm_extract_jobs(
                llm, snapshot, company, "custom_url_params", page.url, keywords,
                current_url=page.url,
            )
            for job in jobs_resp.jobs:
                # When the LLM doesn't return a per-job URL, use the keyword-substituted
                # search URL as fallback — never the template URL with __KEYWORDS__.
                if not job.url:
                    job.url = target_url
                if job.title not in seen_titles:
                    seen_titles.add(job.title)
                    all_jobs.append(job)
        except Exception as exc:
            logger.warning(
                "custom_url_params failed for %s url=%r: %s", company, target_url, exc
            )
            _trace.error(company, "custom_url_params", target_url, "navigation_failed", keyword=keyword, exception=str(exc))

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

    _trace.navigate(company, portal_config.portal_type, url)
    await page.goto(url, timeout=_NAV_TIMEOUT_MS)
    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
    
    # Handle cookie consent if present
    try:
        cookie_accept = await page.query_selector("button:has-text('Accept'), button:has-text('Accept all'), #hs-eu-confirmation-button")
        if cookie_accept:
            await cookie_accept.click()
            await page.wait_for_timeout(1000)
    except Exception:
        pass

    step = 0
    while step < _MAX_EXPLORE_STEPS:
        html = await page.content()
        snapshot = _build_snapshot(html)
        _trace.snapshot(company, portal_config.portal_type, page.url, snapshot)

        # Use cached action if available for this step
        if actions_to_replay and step < len(actions_to_replay):
            action = actions_to_replay[step]
        else:
            action = _llm_explore_portal(
                llm, snapshot, company, portal_config.portal_type, page.url, keywords
            )

        explore_actions.append(action)
        logger.info(
            "Unknown portal [%s] step %d: %s — %s",
            company, step, action.action, action.reasoning,
        )

        if action.action == "extract":
            jobs_resp = _llm_extract_jobs(
                llm, snapshot, company, portal_config.portal_type, page.url, keywords,
                current_url=page.url,
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
            _trace.navigate(company, portal_config.portal_type, action.url)
            _trace.action(company, portal_config.portal_type, page.url, "navigate", None, action.url, action.reasoning, "ok")
            await page.goto(action.url, timeout=_NAV_TIMEOUT_MS)
            await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)

        elif action.action == "click" and action.selector:
            try:
                await _smart_click(page, action.selector, _WAIT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, portal_config.portal_type, page.url, "click", action.selector, None, action.reasoning, "ok")
            except Exception as exc:
                logger.warning(
                    "Unknown portal click failed for %s selector=%r: %s",
                    company, action.selector, exc
                )
                _trace.action(company, portal_config.portal_type, page.url, "click", action.selector, None, action.reasoning, "exception", error=str(exc))
                break

        elif action.action == "type_and_search" and action.selector and action.value:
            try:
                el = await page.query_selector(action.selector)
                if el is None:
                    logger.warning(
                        "Unknown portal selector not found for %s: %r — re-snapshotting",
                        company, action.selector,
                    )
                    _trace.action(company, portal_config.portal_type, page.url, "type_and_search", action.selector, action.value, action.reasoning, "selector_not_found")
                    step += 1
                    continue
                await page.fill(action.selector, action.value, timeout=_WAIT_TIMEOUT_MS)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                _trace.action(company, portal_config.portal_type, page.url, "type_and_search", action.selector, action.value, action.reasoning, "ok")
            except Exception as exc:
                logger.warning(
                    "Unknown portal search failed for %s: %s", company, exc
                )
                _trace.action(company, portal_config.portal_type, page.url, "type_and_search", action.selector, action.value, action.reasoning, "exception", error=str(exc))
                break

        elif action.action == "pagination_next":
            try:
                next_btn = await page.query_selector(
                    "a[aria-label*='next' i], button[aria-label*='next' i]"
                )
                if next_btn:
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=_WAIT_TIMEOUT_MS)
                    _trace.action(company, portal_config.portal_type, page.url, "pagination_next", None, None, action.reasoning, "ok")
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
    relevant_indices = _llm_filter_jobs(
        llm, titles,
        company=company,
        portal_type=portal_config.portal_type,
        current_url=url,
    )
    relevant_set = set(relevant_indices)

    for i, item in enumerate(job_items):
        if i not in relevant_set:
            continue

        job_url = _resolve_url(item.url or "", url) if item.url else url
        # Strip query params for stable dedup ID.
        # If the URL has no unique path, append a title hash so distinct jobs
        # from the same portal don't collide on the same posting_id.
        parsed = urlparse(job_url)
        if parsed.path and parsed.path not in ("/", ""):
            posting_id = urlunparse(parsed._replace(query="", fragment=""))
        else:
            title_hash = hashlib.md5(clean_text(item.title).encode()).hexdigest()[:8]
            posting_id = f"{job_url.rstrip('/')}#t{title_hash}"

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

    _trace.result(company, portal_config.portal_type, len(job_items), len(postings))
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
