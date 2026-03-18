"""All DB read/write logic. No raw SQL elsewhere in the codebase."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agent.db.models import Base, Job

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    role_score: Optional[float]
    location_score: Optional[float]
    stack_score: Optional[float]
    composite_score: Optional[float]
    rationale: Optional[str]
    skill_gaps: list[str]


def _make_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


class JobRepository:
    def __init__(self, db_path: str) -> None:
        self._engine = _make_engine(db_path)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    # ------------------------------------------------------------------
    # Ingest helpers
    # ------------------------------------------------------------------

    def get_seen_ids(self) -> set[str]:
        """Return all posting_ids already in the DB (for dedup)."""
        with self._Session() as session:
            rows = session.query(Job.posting_id).all()
            return {r[0] for r in rows}

    def insert_job(self, posting, scores: ScoreResult) -> int:
        """Insert a new job row; returns the new row id."""
        now = datetime.utcnow()
        job = Job(
            posting_id=posting.posting_id,
            source=posting.source,
            company=posting.company,
            title=posting.title,
            location=posting.location,
            remote=posting.remote,
            description=posting.description,
            url=posting.url,
            role_score=scores.role_score,
            location_score=scores.location_score,
            stack_score=scores.stack_score,
            composite_score=scores.composite_score,
            rationale=scores.rationale,
            skill_gaps=json.dumps(scores.skill_gaps),
            status="new" if scores.composite_score is not None else "parse_error",
            created_at=now,
            updated_at=now,
        )
        with self._Session() as session:
            session.add(job)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                logger.warning("insert_job: duplicate posting_id %r — skipped", posting.posting_id)
                return -1
            session.refresh(job)
            return job.id

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
        sort: str = "score",
    ) -> list[Job]:
        with self._Session() as session:
            q = session.query(Job)
            if status:
                q = q.filter(Job.status == status)
            if sort == "recent":
                q = q.order_by(Job.created_at.desc())
            else:
                q = q.order_by(Job.composite_score.desc().nullslast())
            q = q.limit(limit).offset(offset)
            jobs = q.all()
            session.expunge_all()
            return jobs

    def get_job(self, job_id: int) -> Optional[Job]:
        with self._Session() as session:
            job = session.get(Job, job_id)
            if job:
                session.expunge(job)
            return job

    def update_status(self, job_id: int, status: str) -> None:
        valid = {"new", "reviewed", "applied", "rejected", "parse_error"}
        if status not in valid:
            raise ValueError(f"Invalid status: {status!r}. Must be one of {valid}")
        with self._Session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            job.status = status
            job.updated_at = datetime.utcnow()
            session.commit()
            logger.info("Job %d status → %s", job_id, status)
