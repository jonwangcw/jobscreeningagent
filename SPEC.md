# Job Application Agent — Product Specification

Version: 0.1  
Status: Draft — for Claude Code implementation

---

## 1. Purpose

Build a Dockerized agent that autonomously discovers job postings, scores them
for fit against a candidate profile, persists results, and surfaces them through
a local web dashboard. Resume tailoring and cover letter generation are triggered
manually from the dashboard — the agent never applies to jobs autonomously.

---

## 2. System components

### 2.1 Ingest layer

**Sources:**
- LinkedIn (scrape or unofficial API)
- Indeed (scrape or RSS feed)
- Curated company career pages (URL list in config.yml)

**Behavior:**
- All scrapers run concurrently via `ThreadPoolExecutor`
- The careers page scraper reads `profile/career_pages.txt` — one entry per
  line in the format `https://example.com/careers | Company Name`. Lines
  starting with `#` are treated as comments and skipped. The file path is
  read from `config.sources.careers_pages_file`.
- Each returns a list of `RawPosting` dataclasses:
  ```python
  @dataclass
  class RawPosting:
      posting_id: str      # stable unique ID (URL or platform ID)
      source: str
      company: str
      title: str
      location: str
      remote: bool | None
      description: str
      url: str
      scraped_at: datetime
  ```
- Scraper failures are caught, logged at WARNING, and do not abort the pipeline
- A failed scraper returns an empty list

**Deduplication:**
- After all scrapers complete, postings are checked against `jobs.posting_id`
- Postings already in the DB are skipped entirely — no re-scoring
- New postings proceed to the location gate

---

### 2.2 Location gate

Runs immediately after dedup, before embedding. Cheap string matching on
`posting.location` and `posting.remote`.

**Rules (applied in order):**
1. If `remote == True` → pass
2. If location string contains "Pittsburgh" (case-insensitive) → pass
3. If location string contains "Remote" (case-insensitive) → pass
4. Otherwise → discard (do not write to DB, do not embed)

Rationale: Pittsburgh is a hard geographic constraint. Discarding here avoids
wasting embedding and LLM calls on ineligible postings.

---

### 2.3 Embedding pre-filter

**Model:** `all-MiniLM-L6-v2` via `sentence-transformers` (local GPU inference)

**Process:**
1. At startup (or on first run), embed the full text of `profile/profile.md`
   and cache the vector in memory. Re-embed if the file's mtime changes.
2. For each new posting, call `extract_embedding_text(title, description, max_chars=config.embedding.extraction_chars)`
   (see §11.2). This finds the earliest anchor header in the cleaned description and slices
   from there, falling back to head-truncation if no anchor is found. Title is prepended
   for signal density. Descriptions are already cleaned by the ingest layer (see §11.1).
3. Compute cosine similarity between posting embedding and profile embedding
4. Postings below `config.embedding.similarity_threshold` (default: 0.28) are
   discarded — do not proceed to LLM scoring
5. Similarity score is stored for diagnostics but not surfaced in the dashboard

**EmbeddingBackend interface:**
```python
class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class LocalSentenceTransformer(EmbeddingBackend):
    def __init__(self, model_name: str): ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Model name is read from config — never hardcoded.

---

### 2.4 LLM deep scorer

**Interface:**
```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

class ClaudeBackend(LLMBackend): ...
class OpenAIBackend(LLMBackend): ...
class OllamaBackend(LLMBackend): ...
```

Active backend selected via `config.llm.provider`.

**Scoring dimensions:**

| Dimension | Weight | What it evaluates |
|---|---|---|
| `role_type` | 0.50 | Does the job's actual responsibilities map to the candidate's target role families? Reasons about JD content, not just title. |
| `location` | 0.30 | Is the location compatible? See location scoring table below. |
| `stack` | 0.20 | Overlap between required/preferred skills in JD and candidate's declared stack. |

**Location scoring:**

| Scenario | Score |
|---|---|
| Fully remote | 1.0 |
| Pittsburgh onsite | 1.0 |
| Hybrid, Pittsburgh office | 0.9 |
| Hybrid, other city | 0.0 |
| Onsite, requires relocation | 0.0 |

**Composite score:**
```
if location_score == 0.0:
    composite = 0.0  # hard gate — location failure zeroes the entire posting
else:
    composite = (role_score * 0.50) + (location_score * 0.30) + (stack_score * 0.20)
```
Weights are read from `config.scoring.weights`.

**Scorer output (JSON):**
```json
{
  "role_score": 0.82,
  "location_score": 1.0,
  "stack_score": 0.65,
  "composite_score": 0.84,
  "rationale": "Strong alignment with ML Engineering responsibilities — JD emphasizes production model deployment and Python data pipelines, both core strengths. Stack overlap is good; Go experience listed as preferred is a gap.",
  "skill_gaps": ["Go", "Kubernetes production ops", "A/B experimentation frameworks"]
}
```

**Prompt contract:**
- System prompt: candidate profile.md content + scoring rubric + output format instructions
- User prompt: job title + company + full description
- The LLM is instructed to return **only valid JSON** with no preamble
- Response is parsed with `json.loads()`. On parse failure: log error, store
  `composite_score = null`, set status to `'parse_error'` for manual review.
- All prompts live in `agent/scoring/prompts.py` — no prompt strings elsewhere.

---

### 2.5 Persistence

**ORM:** SQLAlchemy (sync, not async — keeps the agent side simple)

**Table: `jobs`**

```sql
CREATE TABLE jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    posting_id      TEXT    NOT NULL UNIQUE,
    source          TEXT    NOT NULL,
    company         TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    location        TEXT,
    remote          BOOLEAN,
    description     TEXT,
    url             TEXT,
    role_score      REAL,
    location_score  REAL,
    stack_score     REAL,
    composite_score REAL,
    rationale       TEXT,
    skill_gaps      TEXT,   -- JSON array stored as string
    status          TEXT    NOT NULL DEFAULT 'new',
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_composite ON jobs(composite_score DESC);
```

**Status state machine:**

```
new → reviewed → applied
          ↓
       rejected
```

Rows are never deleted. Status transitions are append-only in spirit.

**Repository pattern:**
All DB access goes through `agent/db/repository.py`. No raw SQL or ORM queries
in any other module. The repository exposes typed methods:

```python
def get_seen_ids() -> set[str]: ...
def insert_job(posting: RawPosting, scores: ScoreResult) -> int: ...
def update_status(job_id: int, status: str) -> None: ...
def list_jobs(status: str | None = None, limit: int = 100) -> list[Job]: ...
def get_job(job_id: int) -> Job | None: ...
```

---

### 2.6 Generation pipeline

Triggered only by explicit user action in the dashboard. Never runs automatically.

**Endpoint:** `POST /jobs/{id}/generate`

**Request body:**
```json
{ "cover_letter": false }
```

**Process:**
1. Load job record from DB
2. Validate `status` is `'reviewed'` (reject with 422 if still `'new'`)
3. Create output folder: `outputs/{company}_{title}_{YYYYMMDD}/`
   (sanitize company + title to filesystem-safe string)
4. `shutil.copy(master_resume.docx, output_folder/resume.docx)`
5. Run `resume_tailor.py`:
   - Parse `resume.docx` (python-docx)
   - Inject role-relevant keywords into the Skills section and Summary
   - Reorder sections if role family calls for it (e.g., Quant roles: move
     quantitative skills section above software skills section)
   - Write modified doc back to `output_folder/resume.docx`
6. Write `output_folder/notes.md`:
   - Job title, company, URL, composite score
   - Rationale from scorer
   - List of keywords injected
   - List of skill gaps flagged
   - Section reorders applied (if any)
7. If `cover_letter == true`:
   - Run `cover_letter.py` → LLM call with profile.md + JD → write
     `output_folder/cover_letter.docx`
8. Return `{ "output_path": "outputs/..." }` to dashboard

**Resume tailoring rules:**
- Keywords are injected into existing content — never fabricated
  (i.e., if the candidate doesn't have Kubernetes experience, "Kubernetes" is
  not added to the resume)
- The LLM is given the profile.md, the JD, and the current resume text and
  asked to identify which existing skills/phrases in the profile map to JD
  keywords, then the tailor inserts those mappings into the resume
- Bullets are never rewritten — only the Summary paragraph and Skills section
  are modified
- Section reorder means changing the order of top-level sections only;
  content within sections is untouched

---

### 2.7 Dashboard (FastAPI + React)

**Backend:** FastAPI, served at port 8000  
**Frontend:** React (Vite), built to `frontend/dist/`, served as static files from `/`  
**No separate frontend container** — FastAPI mounts the built React app

**API routes:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/jobs` | List jobs. Query params: `status`, `limit`, `offset` |
| `GET` | `/jobs/{id}` | Single job detail |
| `PATCH` | `/jobs/{id}/status` | Update status: `{status: "reviewed"}` |
| `POST` | `/jobs/{id}/generate` | Trigger generation: `{cover_letter: bool}` |
| `GET` | `/health` | Health check |

**Dashboard UI behavior:**
- Default view: all jobs sorted by `composite_score DESC`
- Status filter tabs: All / New / Reviewed / Applied / Rejected
- Each job card shows: title, company, location, composite score (0–100),
  role/location/stack sub-scores as a mini bar, status badge, date found
- Clicking a card expands to show: full rationale, skill gaps, JD link
- "Mark reviewed" button → PATCH status
- "Generate materials" button → POST generate (checkbox for cover letter)
- "Mark applied" / "Reject" buttons on reviewed jobs
- No pagination initially — limit 200, add pagination if needed later

---

## 3. Scheduler

**Library:** APScheduler (AsyncIOScheduler)  
**Schedule:** Configurable via `config.yml` cron expression, default `0 2 * * *` (2am)  
**Trigger:** Also triggerable via `POST /run` endpoint for manual runs  
**Concurrency:** A lock prevents overlapping runs (if a run is still in progress
when the cron fires, the new run is skipped and logged)

---

## 4. Docker

**Dockerfile (multi-stage):**

```
Stage 1 (node:20-slim):
  - Copy frontend/
  - npm ci && npm run build
  - Output: frontend/dist/

Stage 2 (python:3.11-slim):
  - Install Python dependencies (pyproject.toml)
  - Copy agent/, api/, config.yml, profile/ (read-only)
  - Copy --from=stage1 frontend/dist/ → /app/frontend/dist/
  - EXPOSE 8000
  - CMD: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**docker-compose.yml:**

```yaml
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data            # SQLite DB persists here
      - ./outputs:/app/outputs  # Generated materials persist here
      - ./profile:/app/profile  # Profile doc + master resume (read-only in container)
    environment:
      - DB_PATH=/data/jobs.db
    env_file:
      - .env                    # ANTHROPIC_API_KEY, OPENAI_API_KEY etc.
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # GPU passthrough for sentence-transformers
```

---

## 5. Configuration file (config.yml)

```yaml
schedule:
  cron: "0 2 * * *"

embedding:
  model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.28

llm:
  provider: "claude"
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

## 6. profile.md format

The agent reads this file verbatim. Required sections (H2 headers):

```markdown
## Target roles
List of role families the candidate is targeting, with notes on priority.

## Hard constraints
Location, salary floor, remote requirements, roles explicitly excluded.

## Skills
Flat list of technical skills, tools, languages, frameworks.

## Experience anchors
2–4 bullet summaries of most relevant experience chunks for the LLM to reason against.

## Preferred stack
Technologies the candidate wants to work in (positive signal for stack scoring).

## Anti-targets
Role types, industries, or companies to exclude regardless of score.
```

Missing sections → WARNING log, empty string substituted. Agent does not crash.

---

## 7. career_pages.txt format

One entry per line. Format: `URL | Company Name`. Lines beginning with `#` are comments.

```
# Pittsburgh-area companies
https://careers.bosch.com | Bosch
https://duolingo.com/careers | Duolingo

# Remote-friendly targets
https://jobs.lever.co/anthropic | Anthropic
```

The agent reads this file at the start of each run. Adding or removing entries
takes effect on the next scheduled run — no restart required.

---

## 8. Testing requirements

- `tests/test_scoring.py`: mock LLM backend, verify score parsing, verify
  composite calculation, verify location gate logic
- `tests/test_ingest.py`: mock HTTP responses, verify RawPosting output,
  verify dedup skips seen IDs
- `tests/test_db.py`: in-memory SQLite, verify insert/update/query behavior
- `tests/test_generation.py`: mock LLM, verify master_resume.docx is never
  modified, verify output folder structure

All tests must pass with `pytest` in CI. No tests may make real HTTP or API calls.

---

## 9. Implementation phases

### Phase 1 — Core pipeline (no dashboard)
- DB schema + repository
- Config loader
- One scraper (Indeed RSS, simplest to implement)
- Embedding pre-filter (local model)
- LLM scorer (Claude backend)
- Location gate
- End-to-end: run script → scored jobs in SQLite
- Tests for scoring + DB

### Phase 2 — Ingest completeness
- LinkedIn scraper
- Company careers page scraper
- Scheduler (APScheduler)
- POST /run manual trigger
- Tests for all scrapers

### Phase 3 — Dashboard
- FastAPI app + all routes
- React frontend (job list, status filters, job detail expand)
- Status update flow
- Docker build (multi-stage)

### Phase 4 — Generation
- Resume tailor
- Cover letter generator
- notes.md writer
- POST /jobs/{id}/generate endpoint
- Tests for generation (master_resume immutability check)

### Phase 5 — Hardening
- Scraper retry logic + rate limiting
- Parse error handling for LLM scorer
- Threshold tuning workflow (CLI command to re-run scoring with new threshold)
- OpenAI + Ollama LLM backend impls
- OpenAI embedding backend impl

---

## 11. Text cleaning and keyword-anchored extraction

### 11.1 `clean_text` — ingest-layer utility

**Location:** `agent/ingest/base.py` (module-level free function, not a method)

**Signature:** `clean_text(s: str) -> str`

**Purpose:** Normalize raw scraper output before it is stored in `RawPosting.description`.
Cleaning happens once at ingest; downstream code (embedding, LLM scoring) receives clean text.

**Operations (applied in order):**

1. `html.unescape(s)` — decode HTML entities (`&amp;` → `&`, `&lt;` → `<`, etc.)
2. `re.sub(r'<[^>]+>', ' ', s)` — strip HTML tags, replacing with a space to avoid word-merging
3. `unicodedata.normalize('NFKC', s)` — normalize Unicode (compatibility decomposition + canonical
   composition); collapses ligatures, normalizes dashes, smart quotes, bullet variants, etc.
4. `re.sub(r'[ \t]+', ' ', s)` — collapse runs of spaces/tabs to a single space
5. `re.sub(r'\n{3,}', '\n\n', s)` — collapse 3+ consecutive newlines to exactly 2
6. `.strip()` — remove leading/trailing whitespace

**Dependencies:** `html`, `re`, `unicodedata` — all stdlib. No new packages.

**Integration:** Every scraper's `fetch()` must call `clean_text` on the `description` field
of each `RawPosting` before appending to the result list. New scrapers must do the same
(a comment in `base.py` documents this contract).

---

### 11.2 `extract_embedding_text` — scoring-layer extraction

**Location:** `agent/scoring/pipeline.py` (module-level free function)

**Signature:** `extract_embedding_text(title: str, description: str, max_chars: int = 2000) -> str`

**Purpose:** Choose the most signal-dense window of a job description for embedding, rather than
blindly head-truncating. Many postings lead with boilerplate (company mission, benefits, legal
disclaimers) before the actual responsibilities/requirements — exactly the text that matters
for fit scoring. Anchor-based extraction skips the boilerplate and finds the earliest structural
section header.

**Algorithm:**

1. Define `ANCHOR_HEADERS` as a module-level list of raw pattern strings — the section headers
   that signal the start of substantive content:
   ```python
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
   ```
2. Compile once at module load into `_ANCHOR_PATTERNS` (list of `re.Pattern`):
   ```python
   _ANCHOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ANCHOR_HEADERS]
   ```
3. Search for the **earliest** match position across all patterns in `description`.
4. If a match is found at position `anchor_pos`:
   - Extract `description[anchor_pos : anchor_pos + max_chars]`
5. If no match is found:
   - Log at DEBUG: `"No anchor found in description for: <title>"`
   - Fall back to `description[:max_chars]`
6. Return `title + "\n\n" + extracted_slice`

**Constraints:**
- Do **not** call `clean_text` inside this function. It receives already-cleaned text.
- Do **not** hardcode `2000`; always use the `max_chars` parameter, which is read from
  `config.embedding.extraction_chars`.
- The anchor search is a regex pass only — no LLM calls, no full section parsing.

**Config key:** `config.yml` → `embedding.extraction_chars: 2000`

---

## 10. Out of scope (v0.1)

- Automatic job application submission
- Email notifications
- Multi-user support
- Cloud deployment
- ATS optimization scoring
- Salary data scraping
- Interview tracking
