"""Autonomous portal bug-fixer.

Reads the error log produced by the pipeline, inspects each failing portal's
live DOM with Playwright, calls an LLM to generate a targeted code patch,
applies it, retries the portal scrape, validates with pytest, and loops until
all failing portals are fixed or the iteration limit is reached.

Usage:
    python -m agent.tools.debug_portals
    python -m agent.tools.debug_portals --error-log logs/errors_latest.json
    python -m agent.tools.debug_portals --max-iterations 3 --company "Dollar Bank"
    python -m agent.tools.debug_portals --model claude-opus-4-6
"""
import argparse
import asyncio
import importlib
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root is three levels up: tools/ -> agent/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)

# Only these files may be patched by the debug loop
_PATCH_ALLOWLIST = {
    "agent/ingest/portal_prompts.py",
    "agent/ingest/playwright_scraper.py",
}


# ---------------------------------------------------------------------------
# Error log loading
# ---------------------------------------------------------------------------

def load_error_log(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Live DOM inspection
# ---------------------------------------------------------------------------

async def _inspect_dom_async(url: str, nav_timeout: int, wait_timeout: int) -> dict:
    from playwright.async_api import async_playwright
    from agent.ingest.playwright_scraper import _build_snapshot

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=nav_timeout)
            await page.wait_for_load_state("networkidle", timeout=wait_timeout)
        except Exception as exc:
            logger.warning("DOM inspection navigation failed for %s: %s", url, exc)

        inputs = await page.evaluate("""
            Array.from(document.querySelectorAll('input, textarea, select')).map(el => ({
                id: el.id || null,
                name: el.name || null,
                type: el.type || null,
                placeholder: el.placeholder || null,
                ariaLabel: el.getAttribute('aria-label'),
                automationId: el.getAttribute('data-automation-id'),
                visible: el.offsetWidth > 0 && el.offsetHeight > 0
            }))
        """)
        html = await page.content()
        snapshot = _build_snapshot(html, max_chars=12000)
        final_url = page.url
        await browser.close()

    return {"inputs": inputs, "snapshot": snapshot, "final_url": final_url}


def inspect_live_dom(url: str, pw_config: dict) -> dict:
    nav_timeout = pw_config.get("nav_timeout_ms", 30000)
    wait_timeout = pw_config.get("wait_timeout_ms", 10000)
    return asyncio.run(_inspect_dom_async(url, nav_timeout, wait_timeout))


# ---------------------------------------------------------------------------
# Code excerpt extraction
# ---------------------------------------------------------------------------

def _read_file(rel_path: str) -> str:
    return (_PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def _extract_executor(portal_type: str) -> str:
    """Return the _execute_<portal_type> function body from playwright_scraper.py."""
    source = _read_file("agent/ingest/playwright_scraper.py")
    pattern = rf"(async def _execute_{re.escape(portal_type)}\b.*?)(?=\nasync def |\nclass |\Z)"
    match = re.search(pattern, source, re.DOTALL)
    if match:
        return match.group(1)[:3000]
    return f"(could not find _execute_{portal_type})"


def _extract_prompt_constants(portal_type: str) -> str:
    """Return the relevant prompt constant(s) for this portal type from portal_prompts.py."""
    source = _read_file("agent/ingest/portal_prompts.py")
    prefix = portal_type.upper()
    # Match any UPPERCASE_NAME = """...""" blocks that start with the portal prefix
    pattern = rf'({prefix}[A-Z_]+ = """.*?""")'
    matches = re.findall(pattern, source, re.DOTALL)
    if matches:
        combined = "\n\n".join(matches)
        return combined[:4000]
    return f"(no prompt constants found for {portal_type})"


# ---------------------------------------------------------------------------
# LLM diagnosis + patch generation
# ---------------------------------------------------------------------------

_DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert at debugging web scraping failures in a Python Playwright + LLM job portal scraper.

You will be given:
- A portal failure description (company, portal type, failed selectors, error context)
- Live DOM inspection results (all input elements found on the actual live page)
- The live page text snapshot
- Any previous patches attempted this session and their outcomes
- The current executor function code
- The current prompt constants for this portal type

Your task: identify the root cause of the scraping failure and generate a single targeted
code patch to fix it.

PATCH RULES:
- The patch must target exactly one file from this allowlist:
    agent/ingest/portal_prompts.py
    agent/ingest/playwright_scraper.py
- "old_string" must be an exact substring of the current file content (it will be used
  for string replacement — it must be unique in the file)
- Keep the patch minimal — only change what is necessary to fix the specific failure
- Prefer fixing prompt constants over executor logic when possible
- If you cannot identify a fix, set "patch" to null

Return ONLY valid JSON with no preamble or markdown fences:
{
  "diagnosis": "<explanation of root cause>",
  "patch": {
    "file": "<relative file path>",
    "old_string": "<exact text to replace>",
    "new_string": "<replacement text>"
  },
  "rationale": "<why this patch will fix the problem>"
}

Or if no fix can be identified:
{
  "diagnosis": "<explanation>",
  "patch": null,
  "rationale": "<why no fix is possible>"
}
"""


def _build_diagnosis_prompt(
    portal: dict,
    dom_findings: dict,
    executor_code: str,
    prompt_constants: str,
    history: list[dict],
) -> str:
    lines = [
        f"Company: {portal['company']}",
        f"Portal type: {portal['portal_type']}",
        f"URL: {portal['url']}",
        f"Status: {portal['status']} ({portal['jobs_raw']} jobs found)",
        "",
        "## Failed selectors from last pipeline run",
    ]
    for sf in portal.get("selector_failures", []):
        lines.append(f"  - action={sf['action']} selector={sf['selector']!r}")
    for ex in portal.get("exceptions", []):
        lines.append(f"  - exception: {ex}")
    for lf in portal.get("llm_failures", []):
        lines.append(f"  - llm_failure call_type={lf['call_type']} raw={lf['raw_response']!r}")

    lines += [
        "",
        "## First page snapshot from pipeline run (first 2000 chars)",
        portal.get("first_snapshot", "(none)"),
        "",
        "## Live DOM inspection — inputs found on live page",
        json.dumps(dom_findings["inputs"], indent=2),
        "",
        f"## Live page URL (after redirects): {dom_findings['final_url']}",
        "",
        "## Live page snapshot (current)",
        dom_findings["snapshot"][:3000],
    ]

    if history:
        lines += ["", "## Previous patch attempts this session (all failed or caused test failures)"]
        for i, entry in enumerate(history, 1):
            lines.append(f"\nIteration {i}:")
            lines.append(f"  Diagnosis: {entry['diagnosis']}")
            if entry["patch"]:
                lines.append(f"  Patch: {entry['patch']['file']} — replaced {entry['patch']['old_string'][:80]!r}")
            lines.append(f"  Outcome: {entry['outcome']}")

    lines += [
        "",
        "## Current executor function",
        executor_code,
        "",
        "## Current prompt constants",
        prompt_constants,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patch application and revert
# ---------------------------------------------------------------------------

def apply_patch(patch: dict) -> bool:
    """Apply a patch dict. Returns True on success, False if old_string not found."""
    rel_path = patch["file"]
    if rel_path not in _PATCH_ALLOWLIST:
        logger.error("Patch target %r is not in allowlist — refusing", rel_path)
        return False

    full_path = _PROJECT_ROOT / rel_path
    content = full_path.read_text(encoding="utf-8")
    if patch["old_string"] not in content:
        logger.warning("Patch old_string not found in %s — skipping", rel_path)
        return False

    # Backup before first patch per file this session
    backup_path = str(full_path) + ".debugbak"
    if not Path(backup_path).exists():
        shutil.copy(str(full_path), backup_path)
        logger.info("Backed up %s → %s", rel_path, backup_path)

    new_content = content.replace(patch["old_string"], patch["new_string"], 1)
    full_path.write_text(new_content, encoding="utf-8")
    logger.info("Patch applied to %s", rel_path)
    return True


def revert_patch(rel_path: str) -> None:
    full_path = _PROJECT_ROOT / rel_path
    backup_path = str(full_path) + ".debugbak"
    if Path(backup_path).exists():
        shutil.copy(backup_path, str(full_path))
        Path(backup_path).unlink()
        logger.info("Reverted %s from backup", rel_path)


def _reload_scraper_modules() -> None:
    """Reload portal_prompts then playwright_scraper so patched constants take effect."""
    import agent.ingest.portal_prompts
    import agent.ingest.playwright_scraper
    importlib.reload(agent.ingest.portal_prompts)
    importlib.reload(agent.ingest.playwright_scraper)


# ---------------------------------------------------------------------------
# Portal retry
# ---------------------------------------------------------------------------

def retry_portal(portal: dict, llm_config: dict, pw_config: dict) -> int:
    """Re-run PlaywrightLLMScraper for a single portal. Returns jobs found count."""
    from agent.ingest.playwright_scraper import PlaywrightLLMScraper, configure_trace
    from agent.ingest.careers_page import PortalConfig

    # Write a fresh debug trace
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trace_path = str(log_dir / f"scrape_debug_{ts}.jsonl")
    configure_trace(trace_path)
    logger.info("Debug retry trace: %s", trace_path)

    portal_config = PortalConfig(portal_type=portal["portal_type"])
    scraper = PlaywrightLLMScraper(
        url=portal["url"],
        company=portal["company"],
        portal_config=portal_config,
        llm_config=llm_config,
    )
    try:
        postings = scraper.fetch()
        jobs_found = len(postings)
        logger.info("Retry result for %s: %d jobs", portal["company"], jobs_found)
        return jobs_found
    except Exception as exc:
        logger.warning("Retry failed for %s: %s", portal["company"], exc)
        return 0


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


# ---------------------------------------------------------------------------
# Diagnosis report
# ---------------------------------------------------------------------------

class DiagnosisReport:
    def __init__(self, path: str) -> None:
        self._path = path
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        Path(path).write_text(f"# Debug Session — {ts}\n\n", encoding="utf-8")

    def append(self, text: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# Main fix loop
# ---------------------------------------------------------------------------

def fix_portal(
    portal: dict,
    llm_config: dict,
    pw_config: dict,
    max_iterations: int,
    report: DiagnosisReport,
) -> bool:
    """Run the iterative fix loop for one portal. Returns True if fixed."""
    from agent.scoring.llm_scorer import build_llm_backend
    from agent.scoring.llm_scorer import ClaudeBackend

    llm = build_llm_backend(llm_config)
    company = portal["company"]
    portal_type = portal["portal_type"]

    report.append(f"## {company} ({portal_type}) — {portal['status'].upper()} ({portal['jobs_raw']} jobs)\n")
    report.append(f"**URL:** {portal['url']}")
    report.append(f"**Failed selectors:** " + ", ".join(
        f"`{sf['selector']}`" for sf in portal.get("selector_failures", [])
    ) or "(none)")
    report.append("")

    history: list[dict] = []
    patched_files: set[str] = set()

    for iteration in range(1, max_iterations + 1):
        logger.info("=== %s iteration %d/%d ===", company, iteration, max_iterations)
        report.append(f"**Iteration {iteration}:**")

        # Step 1: Inspect live DOM
        logger.info("Inspecting live DOM for %s at %s", company, portal["url"])
        try:
            dom = inspect_live_dom(portal["url"], pw_config)
        except Exception as exc:
            logger.warning("DOM inspection failed: %s", exc)
            report.append(f"- DOM inspection failed: {exc}")
            break

        visible_inputs = [i for i in dom["inputs"] if i.get("visible")]
        report.append(f"- Live DOM inputs ({len(visible_inputs)} visible): " +
                      ", ".join(
                          f"`#{i['id']}`" if i.get("id") else
                          f"`[name={i['name']}]`" if i.get("name") else
                          f"`[placeholder={i['placeholder']!r}]`" if i.get("placeholder") else
                          f"`[automation-id={i['automationId']!r}]`" if i.get("automationId") else
                          "(unnamed input)"
                          for i in visible_inputs[:10]
                      ))

        # Step 2: Build context and call LLM
        executor_code = _extract_executor(portal_type)
        prompt_constants = _extract_prompt_constants(portal_type)
        user_prompt = _build_diagnosis_prompt(portal, dom, executor_code, prompt_constants, history)

        logger.info("Calling LLM for diagnosis...")
        try:
            raw = llm.complete(system=_DIAGNOSIS_SYSTEM_PROMPT, user=user_prompt, prefill="{")
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())
            diagnosis_result = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM diagnosis call failed: %s", exc)
            report.append(f"- LLM call failed: {exc}")
            break

        diagnosis = diagnosis_result.get("diagnosis", "")
        patch = diagnosis_result.get("patch")
        rationale = diagnosis_result.get("rationale", "")
        report.append(f"- Diagnosis: {diagnosis}")

        if not patch:
            logger.info("LLM could not identify a fix for %s", company)
            report.append("- No patch identified — stopping")
            history.append({"diagnosis": diagnosis, "patch": None, "outcome": "no patch generated"})
            break

        report.append(f"- Patch: `{patch['file']}` — `{patch['old_string'][:60]}...` → `{patch['new_string'][:60]}...`")
        report.append(f"- Rationale: {rationale}")

        # Step 3: Apply patch
        if not apply_patch(patch):
            report.append("- Patch application failed (old_string not found)")
            history.append({"diagnosis": diagnosis, "patch": patch, "outcome": "patch application failed"})
            continue

        patched_files.add(patch["file"])
        _reload_scraper_modules()

        # Step 4: Retry portal
        jobs_found = retry_portal(portal, llm_config, pw_config)
        report.append(f"- Retry result: {jobs_found} jobs found")

        if jobs_found > 0:
            # Step 5: Run tests
            logger.info("Jobs found — running pytest...")
            tests_passed, test_output = run_tests()
            if tests_passed:
                report.append("- Tests: ✓ all passed")
                report.append(f"\n**Result: FIXED in {iteration} iteration(s)**\n")
                report.append("---\n")
                # Clean up backups for patched files
                for f in patched_files:
                    bak = str(_PROJECT_ROOT / f) + ".debugbak"
                    if Path(bak).exists():
                        Path(bak).unlink()
                return True
            else:
                # Count passed/failed from pytest output
                summary_match = re.search(r"(\d+) passed", test_output)
                fail_match = re.search(r"(\d+) failed", test_output)
                summary = f"{summary_match.group(0) if summary_match else '?'}, {fail_match.group(0) if fail_match else '0 failed'}"
                report.append(f"- Tests: ✗ failed ({summary}) — reverting patch")
                logger.warning("Tests failed after patch — reverting %s", patch["file"])
                revert_patch(patch["file"])
                patched_files.discard(patch["file"])
                _reload_scraper_modules()
                history.append({
                    "diagnosis": diagnosis,
                    "patch": patch,
                    "outcome": f"jobs found={jobs_found} but tests failed: {summary}",
                })
                continue
        else:
            report.append("- Still 0 jobs after patch")
            history.append({
                "diagnosis": diagnosis,
                "patch": patch,
                "outcome": "patch applied but still 0 jobs",
            })

    # Max iterations reached or early exit without fix
    logger.warning("Could not fix %s within %d iterations — reverting all patches", company, max_iterations)
    for f in patched_files:
        revert_patch(f)
    if patched_files:
        _reload_scraper_modules()
    report.append(f"\n**Result: UNRESOLVED after {max_iterations} iteration(s)**\n")
    report.append("---\n")
    return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Autonomous portal bug-fixer")
    parser.add_argument("--error-log", default="logs/errors_latest.json",
                        help="Path to errors JSON produced by the pipeline")
    parser.add_argument("--model", default=None,
                        help="Override LLM model (e.g. claude-opus-4-6)")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max fix iterations per portal (default: 5)")
    parser.add_argument("--company", default=None,
                        help="Fix only this company (exact match)")
    args = parser.parse_args()

    # Load config
    sys.path.insert(0, str(_PROJECT_ROOT))
    from agent.main import load_config
    config = load_config(str(_PROJECT_ROOT / "config.yml"))

    llm_config = dict(config["llm"])
    if args.model:
        llm_config["model"] = args.model

    pw_config = config.get("playwright", {})

    # Load error log
    error_log_path = str(_PROJECT_ROOT / args.error_log)
    logger.info("Loading error log: %s", error_log_path)
    error_log = load_error_log(error_log_path)

    portals = error_log.get("portals", [])
    if args.company:
        portals = [p for p in portals if p["company"] == args.company]

    if not portals:
        logger.info("No failing portals found in error log — nothing to fix")
        return

    # Set up report
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = str(log_dir / f"diagnosis_{ts}.md")
    report = DiagnosisReport(report_path)
    logger.info("Diagnosis report: %s", report_path)

    fixed = 0
    for portal in portals:
        logger.info("--- Processing: %s (%s) ---", portal["company"], portal["portal_type"])
        success = fix_portal(portal, llm_config, pw_config, args.max_iterations, report)
        if success:
            fixed += 1

    logger.info(
        "Debug session complete: %d/%d portals fixed. Report: %s",
        fixed, len(portals), report_path,
    )


if __name__ == "__main__":
    main()
