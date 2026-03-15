"""Shared FastAPI dependencies."""
import os

from agent.db.repository import JobRepository


def get_repo() -> JobRepository:
    db_path = os.environ.get("DB_PATH", "./data/jobs.db")
    return JobRepository(db_path)
