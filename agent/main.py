"""Pipeline orchestrator.

Execution order:
  1. Run all scrapers in parallel (ThreadPoolExecutor)
  2. Dedup against SQLite (skip seen posting_id)
  3. Location gate — discard non-Pittsburgh / non-remote postings immediately
  4. Embed new postings + profile.md → cosine similarity pre-filter
  5. LLM deep score → structured JSON
  6. Persist to DB with status='new'
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from agent.db.repository import JobRepository
from agent.ingest.base import RawPosting
from agent.ingest.careers_page import CareersPageScraper, build_careers_scrapers
from agent.ingest.indeed import IndeedScraper
from agent.ingest.linkedin import LinkedInScraper
from agent.scoring.llm_scorer import build_llm_backend
from agent.scoring.pipeline import ScoringPipeline

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict[str, Any]) -> dict[str, int]:
    """Run the full ingestion + scoring pipeline. Returns a stats dict."""
    db_path = os.environ.get("DB_PATH", config.get("db_path", "./data/jobs.db"))
    # Ensure parent dir exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

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
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    run_pipeline(cfg)
