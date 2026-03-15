"""SQLAlchemy ORM models."""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    posting_id = Column(Text, nullable=False, unique=True)
    source = Column(Text, nullable=False)
    company = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    location = Column(Text)
    remote = Column(Boolean)
    description = Column(Text)
    url = Column(Text)
    role_score = Column(Float)
    location_score = Column(Float)
    stack_score = Column(Float)
    composite_score = Column(Float)
    rationale = Column(Text)
    skill_gaps = Column(Text)          # JSON array stored as string
    status = Column(Text, nullable=False, default="new")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_composite", "composite_score"),
    )
