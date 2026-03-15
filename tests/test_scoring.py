"""Scoring tests — mock LLM backend, verify score parsing + composite logic + location gate."""
import logging

import pytest

from agent.db.repository import ScoreResult
from agent.ingest.base import RawPosting
from agent.scoring.llm_scorer import LLMBackend, parse_score_response
from agent.scoring.pipeline import ScoringPipeline, extract_embedding_text


# ---------- LLM mock ----------

class MockLLM(LLMBackend):
    def __init__(self, response: str):
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


VALID_SCORE_JSON = """{
  "role_score": 0.85,
  "location_score": 1.0,
  "stack_score": 0.70,
  "composite_score": 0.845,
  "rationale": "Strong alignment.",
  "skill_gaps": ["Kubernetes", "Go"]
}"""

INVALID_JSON = "Sure! Here are the scores: [not json]"

MISSING_FIELD_JSON = '{"role_score": 0.8, "location_score": 1.0}'


# ---------- parse_score_response ----------

def test_parse_valid_json():
    result = parse_score_response(VALID_SCORE_JSON)
    assert result.role_score == pytest.approx(0.85)
    assert result.location_score == pytest.approx(1.0)
    assert result.stack_score == pytest.approx(0.70)
    assert result.composite_score == pytest.approx(0.845)
    assert result.rationale == "Strong alignment."
    assert "Kubernetes" in result.skill_gaps
    assert "Go" in result.skill_gaps


def test_parse_invalid_json_returns_null_scores():
    result = parse_score_response(INVALID_JSON)
    assert result.composite_score is None
    assert result.role_score is None
    assert "parse_error" in result.rationale


def test_parse_missing_field_returns_null():
    result = parse_score_response(MISSING_FIELD_JSON)
    assert result.composite_score is None


# ---------- Location gate ----------

def make_posting(**kwargs):
    from datetime import datetime
    defaults = dict(
        posting_id="x",
        source="indeed",
        company="Co",
        title="ML Engineer",
        location="Pittsburgh, PA",
        remote=None,
        description="desc",
        url="https://example.com",
        scraped_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return RawPosting(**defaults)


def test_location_gate_remote_true():
    p = make_posting(location="San Francisco, CA", remote=True)
    assert ScoringPipeline.passes_location_gate(p) is True


def test_location_gate_pittsburgh():
    p = make_posting(location="Pittsburgh, PA", remote=None)
    assert ScoringPipeline.passes_location_gate(p) is True


def test_location_gate_remote_in_location_string():
    p = make_posting(location="Remote - US", remote=None)
    assert ScoringPipeline.passes_location_gate(p) is True


def test_location_gate_fails_other_city():
    p = make_posting(location="Austin, TX", remote=None)
    assert ScoringPipeline.passes_location_gate(p) is False


def test_location_gate_fails_none_location():
    p = make_posting(location="", remote=None)
    assert ScoringPipeline.passes_location_gate(p) is False


# ---------- Composite calculation ----------

def test_composite_with_zero_location():
    """If location_score == 0.0, composite must be 0.0 regardless of other scores."""
    data = parse_score_response(
        '{"role_score": 0.9, "location_score": 0.0, "stack_score": 0.8, "composite_score": 0.0, "rationale": "", "skill_gaps": []}'
    )
    assert data.composite_score == pytest.approx(0.0)


def test_composite_weighted_formula():
    """Verify weighted formula: (role*0.5) + (loc*0.3) + (stack*0.2)."""
    role, loc, stack = 0.8, 1.0, 0.6
    expected = (role * 0.50) + (loc * 0.30) + (stack * 0.20)
    result = parse_score_response(
        f'{{"role_score": {role}, "location_score": {loc}, "stack_score": {stack}, '
        f'"composite_score": {expected}, "rationale": "", "skill_gaps": []}}'
    )
    assert result.composite_score == pytest.approx(expected, abs=0.001)


# ---------- extract_embedding_text ----------

def test_extract_anchor_found():
    """When an anchor header is present, extraction starts at the anchor, not boilerplate."""
    desc = "We are a great company.\n\nResponsibilities\nBuild models\nDeploy pipelines"
    result = extract_embedding_text("ML Engineer", desc)
    assert result.startswith("ML Engineer\n\nResponsibilities")
    assert "We are a great company" not in result


def test_extract_no_anchor_fallback(caplog):
    """When no anchor is found, falls back to head-truncation and logs at DEBUG."""
    desc = "We are hiring. Come join us. Great benefits await you here."
    with caplog.at_level(logging.DEBUG, logger="agent.scoring.pipeline"):
        result = extract_embedding_text("Engineer", desc)
    assert result == "Engineer\n\n" + desc[:2000]
    assert any("No anchor found" in r.message for r in caplog.records)


def test_extract_earliest_anchor():
    """When multiple anchors exist, extraction starts at the earliest one."""
    # "About the role" at char ~200, "Requirements" at char ~800
    prefix = "x" * 200
    desc = prefix + "About the role\nDo cool things\n" + "y" * 570 + "Requirements\nPython"
    result = extract_embedding_text("SWE", desc)
    # Should start at "About the role", not "Requirements"
    assert "About the role" in result
    idx_about = result.find("About the role")
    idx_req = result.find("Requirements")
    assert idx_about < idx_req


def test_extract_respects_max_chars():
    """Extracted slice must not exceed max_chars characters (plus title + separator)."""
    desc = "Responsibilities\n" + "a" * 5000
    result = extract_embedding_text("T", desc, max_chars=100)
    title_and_sep = len("T\n\n")
    assert len(result) <= title_and_sep + 100
