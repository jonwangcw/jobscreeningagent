"""Pipeline orchestrator.

Execution order:
  1. Run all scrapers in parallel (ThreadPoolExecutor)
  2. Dedup against SQLite (skip seen posting_id)
  2.5 Enrich descriptions — best-effort httpx fetch for postings with empty descriptions
  3. Location gate — discard non-Pittsburgh / non-remote postings immediately
  4. Embed new postings + profile.md → cosine similarity pre-filter
  5. LLM deep score → structured JSON
  6. Persist to DB with status='new'
"""
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

from agent.db.repository import JobRepository
from agent.ingest.base import RawPosting
from agent.ingest.careers_page import CareersPageScraper, build_careers_scrapers
from agent.ingest.indeed import IndeedScraper
from agent.ingest.linkedin import LinkedInScraper
from agent.scoring.llm_scorer import build_llm_backend
from agent.scoring.pipeline import ScoringPipeline

logger = logging.getLogger(__name__)

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_job_description(url: str) -> str:
    """Best-effort httpx fetch of a job detail page; returns plain text or ''."""
    import certifi
    import httpx
    from bs4 import BeautifulSoup

    try:
        with httpx.Client(
            headers=_FETCH_HEADERS, follow_redirects=True, timeout=15, verify=certifi.where()
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception as exc:
        logger.debug("Description fetch failed for %s: %s", url, exc)
        return ""


def _write_error_log(trace_path: str, stats: dict[str, int]) -> str | None:
    """Parse a JSONL trace file and write a compact per-portal error summary.

    Returns the path of the written error log, or None if the trace is unreadable
    or contains no failures/partials.
    """
    try:
        with open(trace_path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        logger.warning("Could not read trace for error log: %s", exc)
        return None

    # Group events by (company, portal)
    by_portal: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ev in events:
        key = (ev.get("company", ""), ev.get("portal", ""))
        by_portal[key].append(ev)

    portals_out = []
    for (company, portal_type), evs in by_portal.items():
        selector_failures = []
        exceptions = []
        llm_failures = []
        first_snapshot: str | None = None
        portal_url: str | None = None
        jobs_raw: int | None = None
        jobs_after_filter: int | None = None

        for ev in evs:
            etype = ev.get("event")
            if etype == "navigate" and portal_url is None:
                portal_url = ev.get("url")
            elif etype == "snapshot" and first_snapshot is None:
                first_snapshot = (ev.get("text") or "")[:2000]
            elif etype == "action":
                if ev.get("status") == "selector_not_found":
                    selector_failures.append({
                        "action": ev.get("action_type"),
                        "selector": ev.get("selector"),
                        "url": ev.get("url"),
                    })
                elif ev.get("status") == "exception":
                    exceptions.append({
                        "action": ev.get("action_type"),
                        "selector": ev.get("selector"),
                        "error": ev.get("error"),
                        "url": ev.get("url"),
                    })
            elif etype == "error":
                ctx = {k: v for k, v in ev.items() if k not in ("event", "company", "portal", "ts")}
                exceptions.append(ctx)
            elif etype == "llm_call" and not ev.get("valid_json"):
                llm_failures.append({
                    "call_type": ev.get("call_type"),
                    "url": ev.get("url"),
                    "raw_response": (ev.get("raw_response") or "")[:500],
                })
            elif etype == "result":
                jobs_raw = ev.get("jobs_raw", 0)
                jobs_after_filter = ev.get("jobs_after_filter", 0)

        # Classify
        has_issues = bool(selector_failures or exceptions or llm_failures)
        if jobs_raw is None:
            status = "missing"
        elif jobs_raw == 0:
            status = "failure"
        elif has_issues:
            status = "partial"
        else:
            status = "ok"

        if status == "ok":
            continue

        portals_out.append({
            "company": company,
            "portal_type": portal_type,
            "url": portal_url,
            "status": status,
            "jobs_raw": jobs_raw if jobs_raw is not None else 0,
            "jobs_after_filter": jobs_after_filter if jobs_after_filter is not None else 0,
            "selector_failures": selector_failures,
            "exceptions": exceptions,
            "llm_failures": llm_failures,
            "first_snapshot": first_snapshot or "",
        })

    if not portals_out:
        return None

    log_dir = Path(trace_path).parent
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    error_path = str(log_dir / f"errors_{ts}.json")
    payload = {
        "run_ts": datetime.utcnow().isoformat(),
        "trace_file": trace_path,
        "pipeline_stats": stats,
        "portals": portals_out,
    }
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Always overwrite errors_latest.json for easy access
    latest_path = str(log_dir / "errors_latest.json")
    shutil.copy(error_path, latest_path)

    return error_path


def load_config(config_path: str = "config.yml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict[str, Any]) -> dict[str, int]:
    """Run the full ingestion + scoring pipeline. Returns a stats dict."""
    db_path = os.environ.get("DB_PATH", config.get("db_path", "./data/jobs.db"))
    # Ensure parent dir exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Configure Playwright trace logging if enabled
    trace_path: str | None = None
    if config.get("playwright", {}).get("trace_log"):
        from agent.ingest.playwright_scraper import configure_trace
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        trace_path = str(log_dir / f"scrape_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl")
        configure_trace(trace_path)
        logger.info("Playwright trace log: %s", trace_path)

    repo = JobRepository(db_path)
    llm = build_llm_backend(config["llm"])
    scoring = ScoringPipeline(config, llm)

    # 1. Build scrapers
    scrapers = []
    if config.get("sources", {}).get("indeed", True):
        scrapers.append(IndeedScraper())
    if config.get("sources", {}).get("linkedin", True):
        scrapers.append(LinkedInScraper())

    careers_file = config.get("sources", {}).get("careers_pages_file")
    use_playwright = config.get("sources", {}).get("playwright", True)

    if careers_file:
        if use_playwright:
            # build_careers_scrapers returns CareersPageScraper instances for simple
            # entries and PlaywrightLLMScraper instances for portal entries.
            careers_scrapers = build_careers_scrapers(
                careers_pages_file=careers_file,
                llm_config=config["llm"],
            )
            scrapers.extend(careers_scrapers)
        else:
            # Playwright disabled — fall back to simple HTTP scraper only
            scrapers.append(CareersPageScraper(careers_file))

    all_postings: list[RawPosting] = []
    with ThreadPoolExecutor(max_workers=max(len(scrapers), 1)) as pool:
        futures = {pool.submit(s.fetch): type(s).__name__ for s in scrapers}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                logger.info("%s returned %d postings", name, len(results))
                all_postings.extend(results)
            except Exception as exc:
                logger.warning("%s raised an unexpected exception: %s", name, exc)

    # 2. Dedup
    seen_ids = repo.get_seen_ids()
    new_postings = [p for p in all_postings if p.posting_id not in seen_ids]
    logger.info(
        "Dedup: %d total → %d new (skipped %d seen)",
        len(all_postings), len(new_postings), len(all_postings) - len(new_postings),
    )

    # 2.5. Enrich descriptions for postings that came back with no description text.
    # Playwright portal scrapers only capture titles/URLs; the full JD must be fetched
    # separately so the embedding and LLM scoring steps have meaningful content.
    enriched = 0
    for posting in new_postings:
        if not posting.description and posting.url and "__KEYWORDS__" not in posting.url:
            desc = _fetch_job_description(posting.url)
            if desc:
                posting.description = desc
                enriched += 1
    if enriched:
        logger.info("Description enrichment: fetched %d/%d postings", enriched, len(new_postings))

    # 3–6. Score and persist
    stats = {"fetched": len(all_postings), "new": len(new_postings), "scored": 0, "discarded": 0}

    for posting in new_postings:
        result = scoring.score(posting)
        if result is None:
            stats["discarded"] += 1
            continue
        repo.insert_job(posting, result)
        stats["scored"] += 1

    logger.info(
        "Pipeline complete: fetched=%d new=%d scored=%d discarded=%d",
        stats["fetched"], stats["new"], stats["scored"], stats["discarded"],
    )

    # Write compact error log for failing/partial portals
    if trace_path:
        error_log = _write_error_log(trace_path, stats)
        if error_log:
            logger.info("Portal error log: %s", error_log)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    run_pipeline(cfg)
