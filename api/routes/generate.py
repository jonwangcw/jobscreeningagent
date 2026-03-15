"""Generation endpoint: POST /jobs/{id}/generate"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.db.repository import JobRepository
from api.deps import get_repo
from agent.generation.cover_letter import generate_cover_letter
from agent.generation.resume_tailor import tailor_resume
from agent.scoring.llm_scorer import build_llm_backend

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["generate"])


class GenerateRequest(BaseModel):
    cover_letter: bool = False


class GenerateResponse(BaseModel):
    output_path: str


def _sanitize(text: str) -> str:
    """Make a string filesystem-safe."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)[:40]


def get_generate_router(config: dict[str, Any]) -> APIRouter:
    """Returns the router with config injected via closure."""

    @router.post("/{job_id}/generate", response_model=GenerateResponse)
    def generate(
        job_id: int,
        body: GenerateRequest,
        repo: JobRepository = Depends(get_repo),
    ):
        job = repo.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status == "new":
            raise HTTPException(
                status_code=422,
                detail="Job must be reviewed before generating materials. Update status to 'reviewed' first.",
            )

        # Create output folder
        date_str = datetime.utcnow().strftime("%Y%m%d")
        folder_name = f"{_sanitize(job.company)}_{_sanitize(job.title)}_{date_str}"
        output_dir = Path(config.get("output_dir", "./outputs")) / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        resume_output = str(output_dir / "resume.docx")
        master_resume_path = config["master_resume_path"]
        profile_path = config["profile_path"]
        profile_text = Path(profile_path).read_text(encoding="utf-8")

        llm = build_llm_backend(config["llm"])

        # Tailor resume (copies master + applies changes)
        changes = tailor_resume(
            master_resume_path=master_resume_path,
            output_path=resume_output,
            job=job,
            profile_text=profile_text,
            llm=llm,
        )

        # Write notes.md
        skill_gaps = []
        if job.skill_gaps:
            try:
                skill_gaps = json.loads(job.skill_gaps)
            except (ValueError, TypeError):
                skill_gaps = []

        notes_lines = [
            f"# {job.title} @ {job.company}",
            "",
            f"**URL:** {job.url}",
            f"**Composite score:** {job.composite_score:.2f}" if job.composite_score else "**Composite score:** N/A",
            f"**Status:** {job.status}",
            "",
            "## Scorer rationale",
            job.rationale or "",
            "",
            "## Skill gaps flagged",
            *[f"- {gap}" for gap in skill_gaps],
            "",
            "## Resume changes",
            f"**Keywords injected (skills):** {', '.join(changes.get('skills_keywords', []))}",
            f"**Summary additions:** {', '.join(changes.get('summary_additions', []))}",
            f"**Section reorder:** {' → '.join(changes.get('section_reorder', []))}",
            "",
            "## Mapping notes",
            *[f"- {note}" for note in changes.get("mapping_notes", [])],
        ]
        notes_path = output_dir / "notes.md"
        notes_path.write_text("\n".join(notes_lines), encoding="utf-8")

        # Cover letter (optional)
        if body.cover_letter:
            cl_path = str(output_dir / "cover_letter.docx")
            generate_cover_letter(
                output_path=cl_path,
                job=job,
                profile_text=profile_text,
                llm=llm,
            )

        logger.info("Generation complete for job %d → %s", job_id, output_dir)
        return GenerateResponse(output_path=str(output_dir))

    return router
