"""APScheduler wrapper that runs the pipeline on a cron schedule.

Also exposes a manual trigger used by POST /run.
A threading.Lock prevents overlapping runs.
"""
import logging
import threading
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None


def _run_with_lock(config: dict[str, Any]) -> dict[str, int] | None:
    """Attempt to acquire lock and run pipeline. Returns stats or None if already running."""
    if not _run_lock.acquire(blocking=False):
        logger.warning("Pipeline run skipped — a run is already in progress")
        return None
    try:
        from agent.main import run_pipeline  # local import to avoid circular deps
        return run_pipeline(config)
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc, exc_info=True)
        return None
    finally:
        _run_lock.release()


def start_scheduler(config: dict[str, Any]) -> None:
    """Start the background scheduler. Call once at app startup."""
    global _scheduler

    cron_expr = config.get("schedule", {}).get("cron", "0 2 * * *")
    cron_parts = cron_expr.split()
    if len(cron_parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expr!r}")

    minute, hour, day, month, day_of_week = cron_parts

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _run_with_lock,
        CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week),
        args=[config],
        id="nightly_pipeline",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Scheduler started. Next run at: %s", _scheduler.get_jobs()[0].next_run_time)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def trigger_run(config: dict[str, Any]) -> dict[str, int] | None:
    """Manually trigger a pipeline run (from POST /run endpoint)."""
    logger.info("Manual pipeline run triggered")
    return _run_with_lock(config)
