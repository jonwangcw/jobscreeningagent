# Job Application Agent — CLAUDE.md

This file is the authoritative guide for Claude Code working on this codebase.
Read it in full before touching any file. Re-read it when resuming a session.

---

## Project overview

An autonomous job application agent that runs nightly, scrapes postings from
LinkedIn, Indeed, and a curated list of company career pages, scores them for
fit against a hand-maintained candidate profile, stores results in SQLite, and
surfaces them through a local web dashboard (FastAPI + React). When the user
approves a job in the dashboard, the agent tailors a resume clone and optionally
generates a cover letter on demand.

---

## Repository layout

```
job-agent/
├── CLAUDE.md                  ← this file
├── SPEC.md                    ← full product specification
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── config.yml                 ← runtime config (thresholds, schedule, backend)
│
├── profile/
│   ├── profile.md             ← HAND-MAINTAINED candidate profile (never auto-edited)
│   ├── career_pages.txt       ← HAND-MAINTAINED list of company career URLs (never auto-edited)
│   └── master_resume.docx     ← CANONICAL resume (never mutated — always cloned)
│
├── agent/
│   ├── __init__.py
│   ├── main.py                ← entrypoint; orchestrates pipeline stages
│   ├── scheduler.py           ← APScheduler cron wrapper
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── base.py            ← abstract Scraper(ABC)
│   │   ├── linkedin.py
│   │   ├── indeed.py
│   │   └── careers_page.py    ← reads curated URL list from config
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── embedder.py        ← abstract EmbeddingBackend + LocalSentenceTransformer impl
│   │   ├── llm_scorer.py      ← abstract LLMBackend + concrete impls (Claude, OpenAI, Ollama)
│   │   ├── pipeline.py        ← orchestrates embed → threshold → LLM deep score
│   │   └── prompts.py         ← all scoring prompt templates (single source of truth)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── resume_tailor.py   ← clones master_resume.docx, applies keyword/section changes
│   │   └── cover_letter.py    ← generates cover_letter.docx on demand
│   │
│   └── db/
│       ├── __init__.py
│       ├── models.py          ← SQLAlchemy models
│       └── repository.py      ← all DB read/write logic; no raw SQL elsewhere
│
├── api/
│   ├── __init__.py
│   ├── main.py                ← FastAPI app factory
│   └── routes/
│       ├── jobs.py            ← GET /jobs, PATCH /jobs/{id}/status
│       └── generate.py        ← POST /jobs/{id}/generate
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── components/
│       └── api.ts             ← typed fetch wrappers for all backend routes
│
├── outputs/                   ← gitignored; per-job output folders land here
│   └── {company}_{role}_{date}/
│       ├── resume.docx
│       ├── notes.md
│       └── cover_letter.docx  ← only if requested
│
└── tests/
    ├── test_scoring.py
    ├── test_ingest.py
    └── test_db.py
```

---

## Sacred files — never auto-modify

| File | Rule |
|---|---|
| `profile/profile.md` | Read-only for the agent. Only the human edits this. |
| `profile/master_resume.docx` | Never written to. Always `shutil.copy()` to `outputs/` first. |
| `profile/career_pages.txt` | Read-only for the agent. Only the human edits this. |

Violating either rule corrupts the baseline and breaks every downstream run.

---

## Key abstractions

### EmbeddingBackend (agent/scoring/embedder.py)

```python
class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Active implementation: `LocalSentenceTransformer` using `all-MiniLM-L6-v2`.
The model name is read from `config.yml` (`embedding.model`), not hardcoded.
Swap to a hosted backend by adding a new impl and changing config — no other
files should need to change.

### LLMBackend (agent/scoring/llm_scorer.py)

```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...
```

Concrete impls: `ClaudeBackend`, `OpenAIBackend`, `OllamaBackend`.
Active backend selected via `config.yml` (`llm.provider`).
All prompt strings live in `agent/scoring/prompts.py` — never inline them
inside backend classes.

### Scraper (agent/ingest/base.py)

```python
class Scraper(ABC):
    @abstractmethod
    def fetch(self) -> list[RawPosting]: ...
```

Each scraper returns `RawPosting` dataclasses. Dedup and normalization happen
in `agent/main.py` after all scrapers run, not inside scrapers.

---

## Pipeline execution order

```
scheduler.py  →  main.py
    │
    ├── 1. Run all scrapers in parallel (ThreadPoolExecutor)
    ├── 2. Dedup against SQLite (skip seen posting_id)
    ├── 3. Location gate — discard non-Pittsburgh / non-remote postings immediately
    ├── 4. Embed new postings + profile.md → cosine similarity pre-filter
    │        threshold from config.yml (embedding.similarity_threshold, default 0.28)
    ├── 5. LLM deep score → structured JSON (role_score, location_score, stack_score,
    │        composite, rationale, skill_gaps[])
    ├── 6. Persist to DB with status='new'
    └── 7. Done — dashboard polls DB for display
```

Generation pipeline (triggered from dashboard, not scheduler):

```
POST /jobs/{id}/generate
    │
    ├── shutil.copy(master_resume.docx → outputs/{job_folder}/resume.docx)
    ├── resume_tailor.py  →  keyword injection + section reorder
    ├── Write notes.md  (what changed, why, skill gaps flagged by scorer)
    └── If cover_letter=true: cover_letter.py → cover_letter.docx
```

---

## Database schema

Table: `jobs`

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| posting_id | TEXT UNIQUE | source URL or platform ID — used for dedup |
| source | TEXT | linkedin / indeed / careers_page |
| company | TEXT | |
| title | TEXT | |
| location | TEXT | raw string from posting |
| remote | BOOLEAN | |
| description | TEXT | full JD text |
| url | TEXT | |
| role_score | REAL | 0–1 |
| location_score | REAL | 0–1 |
| stack_score | REAL | 0–1 |
| composite_score | REAL | weighted composite |
| rationale | TEXT | LLM-generated 2–3 sentence explanation |
| skill_gaps | TEXT | JSON array of strings |
| status | TEXT | new / reviewed / applied / rejected |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

All DB access goes through `agent/db/repository.py`. No raw SQL in any other file.

---

## Scoring rubric

Weights (configurable in `config.yml` under `scoring.weights`):

| Dimension | Default weight |
|---|---|
| role_type | 0.50 |
| location | 0.30 |
| stack | 0.20 |

Location scoring logic:
- Fully remote → 1.0
- Hybrid with Pittsburgh office → 0.9
- Pittsburgh onsite only → 1.0
- Requires relocation → 0.0 (hard fail — also applied as pre-filter in step 3)

The location gate in step 3 runs **before** embedding to avoid wasting compute
on clearly ineligible postings.

---

## Configuration (config.yml)

```yaml
schedule:
  cron: "0 2 * * *"           # 2am nightly

embedding:
  model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.28   # tune after first few runs

llm:
  provider: "claude"           # claude | openai | ollama
  model: "claude-sonnet-4-20250514"
  max_tokens: 2048

scoring:
  weights:
    role_type: 0.50
    location: 0.30
    stack: 0.20

sources:
  linkedin: true
  indeed: true
  careers_pages_file: "./profile/career_pages.txt"

output_dir: "./outputs"
profile_path: "./profile/profile.md"
master_resume_path: "./profile/master_resume.docx"
```

---

## Docker setup

- Single `Dockerfile`, multi-stage: builder (npm build) + runtime (Python + built frontend)
- `docker-compose.yml` runs one service with `--gpus all` for sentence-transformers GPU inference
- SQLite DB lives at `/data/jobs.db` — mount a host volume so it persists across container restarts
- Frontend is served as static files from FastAPI (`/` route), no separate container
- Env vars for API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) injected via `.env` file
- `profile/` volume mount contains three hand-maintained files: `profile.md`,
  `master_resume.docx`, and `career_pages.txt` — all read-only inside the container

---

## profile.md structure

The agent reads this file as a single string and passes it to both the
embedding step (to generate the profile embedding vector) and the LLM scorer
(as context in the system prompt). It must contain these sections:

```markdown
## Target roles
## Hard constraints
## Skills
## Experience anchors
## Preferred stack
## Anti-targets
```

`## Anti-targets` is a list of roles, industries, companies, or role patterns
the candidate explicitly does not want — regardless of how well they score.
Examples: defense contractors, "data analyst" roles that are Excel work in
disguise, companies already rejected, roles requiring any relocation.
The agent checks this section before scoring and discards matching postings.

If any section is missing, the scoring step logs a warning and substitutes an
empty string — it does not crash.

---

## Code style and conventions

- Python 3.11+, type hints everywhere, dataclasses for data transfer objects
- `ruff` for linting, `black` for formatting — both enforced in CI
- No business logic in API route handlers — route handlers do exactly three
  things: validate the incoming request, call a function from `agent/` or
  `agent/db/repository.py`, and return the response. Score calculation, file
  generation, status transitions, and all domain logic live in `agent/`. This
  keeps logic testable without spinning up the web stack.
- Logging via stdlib `logging`, not `print`. All pipeline stages log at INFO.
  Scraper errors log at WARNING and continue — one failed scraper must not abort the run.
- Tests use `pytest`. Mock all external HTTP calls and LLM API calls.
  DB tests use an in-memory SQLite instance.
- Never import from `api/` inside `agent/` — the scheduler runs
  `agent/main.py` directly (no web server involved), and circular imports
  between the agent and API layers will break headless execution. All shared
  state goes through the DB, not module-level references.

---

## Things Claude Code must never do

1. Write to `profile/profile.md` or `profile/master_resume.docx`
2. Delete rows from the `jobs` table (set status='rejected' instead)
3. Hardcode API keys, model names, thresholds, or file paths — all go in `config.yml`
4. Put prompt strings anywhere except `agent/scoring/prompts.py`
5. Put raw SQL anywhere except `agent/db/repository.py`
6. Make the generation pipeline run automatically — it is always user-triggered
