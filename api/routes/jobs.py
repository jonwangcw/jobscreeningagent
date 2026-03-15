"""Job listing and status update routes."""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.db.models import Job
from agent.db.repository import JobRepository
from api.deps import get_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------- Pydantic response schemas ----------

class JobResponse(BaseModel):
    id: int
    posting_id: str
    source: str
    company: str
    title: str
    location: Optional[str]
    remote: Optional[bool]
    url: Optional[str]
    role_score: Optional[float]
    location_score: Optional[float]
    stack_score: Optional[float]
    composite_score: Optional[float]
    rationale: Optional[str]
    skill_gaps: list[str]
    status: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class StatusUpdateRequest(BaseModel):
    status: str


# ---------- Helpers ----------

def _job_to_response(job: Job) -> JobResponse:
    gaps = []
    if job.skill_gaps:
        try:
            gaps = json.loads(job.skill_gaps)
        except (ValueError, TypeError):
            gaps = []
    return JobResponse(
        id=job.id,
        posting_id=job.posting_id,
        source=job.source,
        company=job.company,
        title=job.title,
        location=job.location,
        remote=job.remote,
        url=job.url,
        role_score=job.role_score,
        location_score=job.location_score,
        stack_score=job.stack_score,
        composite_score=job.composite_score,
        rationale=job.rationale,
        skill_gaps=gaps,
        status=job.status,
        created_at=str(job.created_at),
        updated_at=str(job.updated_at),
    )


# ---------- Routes ----------

@router.get("", response_model=list[JobResponse])
def list_jobs(
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    repo: JobRepository = Depends(get_repo),
):
    jobs = repo.list_jobs(status=status, limit=limit, offset=offset)
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, repo: JobRepository = Depends(get_repo)):
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.patch("/{job_id}/status", response_model=JobResponse)
def update_status(job_id: int, body: StatusUpdateRequest, repo: JobRepository = Depends(get_repo)):
    try:
        repo.update_status(job_id, body.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    job = repo.get_job(job_id)
    return _job_to_response(job)
