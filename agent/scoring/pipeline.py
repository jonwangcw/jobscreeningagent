"""Scoring pipeline: embed → threshold filter → LLM deep score."""
import logging
import re
import os
from pathlib import Path
from typing import Any

from agent.db.repository import ScoreResult
from agent.ingest.base import RawPosting
from agent.scoring.embedder import EmbeddingBackend, LocalSentenceTransformer, cosine_similarity
from agent.scoring.llm_scorer import LLMBackend, parse_score_response
from agent.scoring.prompts import SCORING_SYSTEM_PROMPT, SCORING_USER_PROMPT

logger = logging.getLogger(__name__)

# Section headers that signal the start of signal-dense job description content.
# Defined once at module level so _ANCHOR_PATTERNS is compiled only once.
ANCHOR_HEADERS = [
    r"responsibilities",
    r"what you('ll| will) do",
    r"the role",
    r"about the role",
    r"job description",
    r"what we('re| are) looking for",
    r"requirements",
    r"qualifications",
    r"minimum qualifications",
    r"preferred qualifications",
    r"you (will|would|should)",
    r"in this role",
    r"key responsibilities",
    r"essential (duties|functions|responsibilities)",
]

_ANCHOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ANCHOR_HEADERS]


def extract_embedding_text(title: str, description: str, max_chars: int = 2000) -> str:
    """Return the most signal-dense window of a description for embedding.

    Finds the earliest anchor header (responsibilities, requirements, etc.) and
    slices from there. Falls back to head-truncation if no anchor is found.

    Receives already-cleaned text — do NOT call clean_text here.
    """
    earliest: int | None = None
    for pattern in _ANCHOR_PATTERNS:
        m = pattern.search(description)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()

    if earliest is not None:
        extracted = description[earliest: earliest + max_chars]
    else:
        logger.debug("No anchor found in description for: %s", title)
        extracted = description[:max_chars]

    return title + "\n\n" + extracted


class ScoringPipeline:
    def __init__(self, config: dict[str, Any], llm: LLMBackend) -> None:
        self._config = config
        self._llm = llm
        self._embedder: EmbeddingBackend = LocalSentenceTransformer(
            config["embedding"]["model"]
        )
        self._profile_path: str = config["profile_path"]
        self._threshold: float = float(config["embedding"]["similarity_threshold"])
        self._extraction_chars: int = int(config["embedding"].get("extraction_chars", 2000))
        self._profile_vector: list[float] | None = None
        self._profile_mtime: float = 0.0
        self._profile_text: str = ""

    # ------------------------------------------------------------------
    # Profile embedding (cached; refreshed if file mtime changes)
    # ------------------------------------------------------------------

    def _get_profile_vector(self) -> list[float]:
        path = Path(self._profile_path)
        mtime = path.stat().st_mtime
        if self._profile_vector is None or mtime != self._profile_mtime:
            self._profile_text = path.read_text(encoding="utf-8")
            self._profile_vector = self._embedder.embed([self._profile_text])[0]
            self._profile_mtime = mtime
            logger.info("Profile embedding refreshed from %s", self._profile_path)
        return self._profile_vector

    def _get_profile_text(self) -> str:
        self._get_profile_vector()  # ensure loaded
        return self._profile_text

    # ------------------------------------------------------------------
    # Location gate (cheap string match — runs before embedding)
    # ------------------------------------------------------------------

    @staticmethod
    def passes_location_gate(posting: RawPosting) -> bool:
        if posting.remote is True:
            return True
        loc = (posting.location or "").lower()
        if "pittsburgh" in loc:
            return True
        if "remote" in loc:
            return True
        return False

    # ------------------------------------------------------------------
    # Anti-target check (quick regex against profile Anti-targets section)
    # ------------------------------------------------------------------

    def _matches_anti_target(self, posting: RawPosting) -> bool:
        import re

        profile_text = self._get_profile_text()
        # Extract the Anti-targets section
        match = re.search(r"## Anti-targets(.*?)(?:^##|\Z)", profile_text, re.S | re.M)
        if not match:
            return False
        anti_text = match.group(1).lower()
        posting_combined = f"{posting.title} {posting.description} {posting.company}".lower()

        # Check for weapons/defense
        if "weapons" in anti_text and re.search(r"weapon|targeting system|lethal", posting_combined):
            return True
        # Pure data analyst
        if "pure data analyst" in anti_text and re.search(
            r"\bdata analyst\b", posting.title.lower()
        ) and not re.search(r"scientist|engineer|model", posting.title.lower()):
            return True
        # Exam-track actuarial
        if "exam-track actuarial" in anti_text and re.search(
            r"actuar", posting.title.lower()
        ) and not re.search(r"model|quant|risk", posting.title.lower()):
            return True
        # MLM / crypto / NFT
        if "mlm" in anti_text and re.search(r"\bcrypto\b|nft|multi.?level", posting_combined):
            return True
        # Pure frontend
        if "pure frontend" in anti_text and re.search(
            r"front.?end developer|mobile developer|ios|android", posting.title.lower()
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Main scoring entrypoint
    # ------------------------------------------------------------------

    def score(self, posting: RawPosting) -> ScoreResult | None:
        """
        Returns ScoreResult or None if the posting should be discarded.
        None means don't write to DB at all.
        """
        # Location gate
        if not self.passes_location_gate(posting):
            logger.debug("Location gate discarded: %s @ %s", posting.title, posting.location)
            return None

        # Embedding pre-filter
        profile_vector = self._get_profile_vector()
        posting_text = extract_embedding_text(
            title=posting.title,
            description=posting.description,
            max_chars=self._extraction_chars,
        )
        posting_vector = self._embedder.embed([posting_text])[0]
        similarity = cosine_similarity(profile_vector, posting_vector)

        if similarity < self._threshold:
            logger.debug(
                "Embedding filter discarded (%.3f < %.3f): %s", similarity, self._threshold, posting.title
            )
            return None

        # Anti-target check
        if self._matches_anti_target(posting):
            logger.info("Anti-target matched, discarding: %s @ %s", posting.title, posting.company)
            return None

        # LLM deep score
        profile_text = self._get_profile_text()
        system_prompt = SCORING_SYSTEM_PROMPT.format(profile_text=profile_text)
        user_prompt = SCORING_USER_PROMPT.format(
            title=posting.title,
            company=posting.company,
            location=posting.location,
            remote=posting.remote,
            description=posting.description[:4000],
        )

        try:
            raw_response = self._llm.complete(system_prompt, user_prompt, prefill="{")
        except Exception as exc:
            logger.error("LLM scoring failed for %s: %s", posting.title, exc)
            return ScoreResult(
                role_score=None, location_score=None, stack_score=None,
                composite_score=None, rationale=f"llm_error: {exc}", skill_gaps=[]
            )

        result = parse_score_response(raw_response)
        logger.info(
            "Scored: %s @ %s → composite=%.2f",
            posting.title,
            posting.company,
            result.composite_score or 0,
        )
        return result
