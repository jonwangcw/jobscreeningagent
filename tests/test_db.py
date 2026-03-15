"""DB layer tests — use in-memory SQLite."""
import json
from datetime import datetime

import pytest

from agent.db.repository import JobRepository, ScoreResult
from agent.ingest.base import RawPosting


def make_repo() -> JobRepository:
    return JobRepository(":memory:")


def make_posting(**kwargs) -> RawPosting:
    defaults = dict(
        posting_id="test-001",
        source="indeed",
        company="Acme Corp",
        title="ML Engineer",
        location="Pittsburgh, PA",
        remote=False,
        description="Build ML pipelines.",
        url="https://example.com/job/1",
        scraped_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return RawPosting(**defaults)


def make_scores(**kwargs) -> ScoreResult:
    defaults = dict(
        role_score=0.8,
        location_score=1.0,
        stack_score=0.6,
        composite_score=0.82,
        rationale="Good fit.",
        skill_gaps=["Kubernetes"],
    )
    defaults.update(kwargs)
    return ScoreResult(**defaults)


def test_insert_and_get():
    repo = make_repo()
    posting = make_posting()
    scores = make_scores()
    job_id = repo.insert_job(posting, scores)
    assert job_id == 1

    job = repo.get_job(job_id)
    assert job is not None
    assert job.title == "ML Engineer"
    assert job.company == "Acme Corp"
    assert job.composite_score == pytest.approx(0.82)
    assert job.status == "new"
    gaps = json.loads(job.skill_gaps)
    assert "Kubernetes" in gaps


def test_insert_parse_error_status():
    repo = make_repo()
    posting = make_posting()
    scores = make_scores(composite_score=None)
    job_id = repo.insert_job(posting, scores)
    job = repo.get_job(job_id)
    assert job.status == "parse_error"


def test_get_seen_ids():
    repo = make_repo()
    assert repo.get_seen_ids() == set()
    repo.insert_job(make_posting(posting_id="abc"), make_scores())
    repo.insert_job(make_posting(posting_id="xyz"), make_scores())
    seen = repo.get_seen_ids()
    assert "abc" in seen
    assert "xyz" in seen


def test_update_status():
    repo = make_repo()
    job_id = repo.insert_job(make_posting(), make_scores())
    repo.update_status(job_id, "reviewed")
    job = repo.get_job(job_id)
    assert job.status == "reviewed"


def test_update_status_invalid():
    repo = make_repo()
    job_id = repo.insert_job(make_posting(), make_scores())
    with pytest.raises(ValueError, match="Invalid status"):
        repo.update_status(job_id, "banana")


def test_update_status_not_found():
    repo = make_repo()
    with pytest.raises(KeyError):
        repo.update_status(999, "reviewed")


def test_list_jobs_filter_by_status():
    repo = make_repo()
    repo.insert_job(make_posting(posting_id="a"), make_scores())
    repo.insert_job(make_posting(posting_id="b"), make_scores())
    job_id = repo.insert_job(make_posting(posting_id="c"), make_scores())
    repo.update_status(job_id, "reviewed")

    all_jobs = repo.list_jobs()
    assert len(all_jobs) == 3

    new_jobs = repo.list_jobs(status="new")
    assert len(new_jobs) == 2

    reviewed = repo.list_jobs(status="reviewed")
    assert len(reviewed) == 1


def test_list_jobs_sorted_by_composite():
    repo = make_repo()
    repo.insert_job(make_posting(posting_id="low"), make_scores(composite_score=0.3))
    repo.insert_job(make_posting(posting_id="high"), make_scores(composite_score=0.9))
    repo.insert_job(make_posting(posting_id="mid"), make_scores(composite_score=0.6))

    jobs = repo.list_jobs()
    scores = [j.composite_score for j in jobs]
    assert scores == sorted(scores, reverse=True)
