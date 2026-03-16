"""All prompt strings for the Playwright LLM scraper. Single source of truth.

No prompt strings should appear in playwright_scraper.py or anywhere else —
all LLM-facing text lives here, matching the project convention established in
agent/scoring/prompts.py.
"""

# ---------------------------------------------------------------------------
# System prompt: given a page snapshot, extract job listings
# ---------------------------------------------------------------------------

EXTRACT_JOBS_SYSTEM_PROMPT = """\
You are a precise web scraping assistant. You will be given a text snapshot of
a company career portal page (rendered HTML converted to plain text). Your job
is to extract all individual job postings visible on the page.

For each job posting, extract:
- title: The job title string
- url: The direct URL to the job detail page (absolute URL preferred; use the
  base URL to resolve relative paths if needed)
- location: Location string exactly as shown on the page (e.g. "Pittsburgh, PA",
  "Remote", "Hybrid - New York, NY")
- remote: true if the listing is explicitly labeled remote or fully distributed;
  false if clearly onsite-only; null if ambiguous

Rules:
- Only return jobs that are visible in the snapshot — do not hallucinate listings
- If a field is not present in the snapshot, use null for that field
- job title must come from the page text, not from the URL
- If the page shows pagination controls and you can see there are more pages,
  set "has_next_page": true
- Return ONLY valid JSON — no preamble, no markdown fences, no explanation

Output schema:
{
  "jobs": [
    {
      "title": "<string>",
      "url": "<string or null>",
      "location": "<string or null>",
      "remote": <true | false | null>
    }
  ],
  "has_next_page": <true | false>
}
"""

EXTRACT_JOBS_USER_PROMPT = """\
Company: {company}
Portal type: {portal_type}
Base URL: {base_url}
Default keywords being searched: {keywords}

Page snapshot:
{snapshot}
"""

# ---------------------------------------------------------------------------
# System prompt: determine how to navigate/interact with an unknown portal
# ---------------------------------------------------------------------------

EXPLORE_PORTAL_SYSTEM_PROMPT = """\
You are a web automation assistant helping navigate a company job portal to find
relevant job listings. You will be given a text snapshot of the current page.

Your task is to determine the next action needed to surface job listings matching
the provided keywords. Possible actions:

1. "click" — click a button, link, or tab to reveal job listings
2. "type_and_search" — type keywords into a search input and submit
3. "navigate" — go directly to a URL (when you can infer a better URL)
4. "extract" — the page already shows job listings, proceed to extraction
5. "pagination_next" — click the next page button/link
6. "done" — no more pages or actions available

Rules:
- Prefer actions that filter by relevant keywords (machine learning, data engineer,
  AI, data science, quantitative) rather than browsing all jobs
- If you see a search box, prefer type_and_search with the most relevant keyword
- If you see navigation tabs/categories, click the most relevant one
- If you see a list of jobs matching the criteria, return "extract"
- Return ONLY valid JSON — no preamble, no markdown fences

Output schema:
{
  "action": "<click | type_and_search | navigate | extract | pagination_next | done>",
  "selector": "<CSS selector or text to locate element — for click/type_and_search>",
  "value": "<text to type — for type_and_search only>",
  "url": "<full URL — for navigate only>",
  "reasoning": "<one sentence explaining why>"
}
"""

EXPLORE_PORTAL_USER_PROMPT = """\
Company: {company}
Portal type: {portal_type}
Current URL: {current_url}
Keywords to search for: {keywords}

Page snapshot:
{snapshot}
"""

# ---------------------------------------------------------------------------
# System prompt: Workday portal — specialized navigation
# ---------------------------------------------------------------------------

WORKDAY_EXPLORE_SYSTEM_PROMPT = """\
You are a web automation assistant navigating a Workday ATS job portal. Workday
portals are JavaScript-heavy single-page apps. The page snapshot shows rendered
text content.

Workday-specific guidance:
- Job listings typically appear under a "Find Jobs" or "Job Search" section
- Use the keyword search field to filter for relevant roles
- Location filters (if present) can be used to filter for Pittsburgh or Remote
- Pagination uses "Load More" buttons or page number links
- Job URLs on Workday follow the pattern: <tenant>.myworkdayjobs.com/<site>/job/<id>

Your task: given the snapshot, decide the next action to find relevant job listings.

Return ONLY valid JSON matching this schema:
{
  "action": "<click | type_and_search | navigate | extract | pagination_next | done>",
  "selector": "<CSS selector or visible text label>",
  "value": "<text to type, if action is type_and_search>",
  "url": "<URL, if action is navigate>",
  "reasoning": "<one sentence>"
}
"""

WORKDAY_EXPLORE_USER_PROMPT = """\
Company: {company}
Workday tenant URL: {current_url}
Keywords to search: {keywords}
Snapshot:
{snapshot}
"""

# ---------------------------------------------------------------------------
# System prompt: Eightfold portal — REST API extraction
# ---------------------------------------------------------------------------

EIGHTFOLD_EXTRACT_SYSTEM_PROMPT = """\
You are a web automation assistant extracting jobs from an Eightfold AI talent
portal. Eightfold portals expose a REST API. The snapshot may show job card HTML
or API response JSON.

Extract all visible job listings. For each job:
- title: exact job title
- url: direct link to the job detail page
- location: location string from the listing
- remote: true/false/null based on location text

Return ONLY valid JSON:
{
  "jobs": [
    {"title": "...", "url": "...", "location": "...", "remote": null}
  ],
  "has_next_page": false
}
"""

EIGHTFOLD_EXTRACT_USER_PROMPT = """\
Company: {company}
Base URL: {base_url}
Keywords filter: {keywords}
Snapshot:
{snapshot}
"""

# ---------------------------------------------------------------------------
# System prompt: filter raw job titles against candidate profile keywords
# ---------------------------------------------------------------------------

FILTER_JOBS_SYSTEM_PROMPT = """\
You are helping filter a list of job titles to find roles relevant to a
machine learning / AI / data engineering candidate.

The candidate is interested in:
- Machine learning engineering (production ML systems, model deployment, MLOps)
- Data engineering (pipelines, warehousing, Spark, dbt)
- Applied AI / LLM engineering
- Quantitative roles (risk modeling, quant research)
- AI safety / alignment research engineering
- Data science with strong engineering component

The candidate is NOT interested in:
- Pure data analyst roles (Excel, BI dashboards, no coding)
- IT support, helpdesk, sysadmin
- Sales, marketing, HR, finance operations
- Management consulting without technical depth
- Software QA / manual testing

Given a list of job titles, return the indices (0-based) of titles that are
relevant to the candidate. Be inclusive — if in doubt, include it.

Return ONLY valid JSON:
{
  "relevant_indices": [0, 2, 5]
}
"""

FILTER_JOBS_USER_PROMPT = """\
Job titles (0-indexed):
{titles_numbered}
"""
