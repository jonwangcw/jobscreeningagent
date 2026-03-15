# Candidate Profile — Jonathan Zhu Wang

Last updated: 2026-03-15

---

## Target roles

Primary targets (in rough priority order):

1. **ML Engineer / Applied AI** — Production ML systems, model deployment, inference pipelines, LLMOps, agentic systems. Companies building AI products where the role involves owning models end-to-end from training through serving.
2. **AI Safety / Alignment** — Technical safety research, evaluation, interpretability, red-teaming, or safety-adjacent engineering at labs or safety-focused orgs (Anthropic, Redwood, ARC, Apollo, etc.).
3. **Data Engineering / MLOps** — Building the infrastructure that ML runs on: data pipelines, feature stores, distributed compute, experiment tracking, model registries.
4. **Quant Finance / Risk Modeling** — Quantitative research, systematic strategies, risk modeling, signal research at hedge funds, prop trading firms, or risk desks at banks. Strong preference for roles that are analytically heavy rather than pure software.

Actuarial science is a lower-priority fallback; include only if composite score is high and role involves genuine statistical modeling rather than exam-path actuarial work.

---

## Hard constraints

- **Location**: Pittsburgh, PA only. Will not relocate. Remote-first or hybrid (Pittsburgh office) acceptable. Fully onsite outside Pittsburgh is a hard no.
- **Salary floor**: $110,000 base. Pittsburgh ML/DS market; negotiate up from there.
- **Start timeline**: Available immediately.
- **Work authorization**: US citizen, no sponsorship needed.

---

## Skills

**Languages**: Python (expert — 7+ years production use), SQL (proficient), Bash, LaTeX

**ML / AI**:
- PyTorch, scikit-learn, HuggingFace Transformers
- QLoRA / PEFT fine-tuning, Unsloth, GGUF quantization
- Ollama (local LLM serving), LangChain, agentic pipeline design
- Synthetic data generation, SFT dataset curation
- Embedding models (sentence-transformers), vector similarity search

**Data & Pipelines**:
- NumPy, SciPy, pandas, Dask
- HDF5, large-scale time-series data
- ETL pipeline design and implementation
- Feature engineering, data cleaning at scale

**Statistics & Signal Processing**:
- Bayesian inference, matched filtering, hypothesis testing
- Fourier / spectral analysis, semicoherent detection methods
- Monte Carlo simulation, upper-limit setting, sensitivity estimation
- Time-series analysis, noise characterization

**Infrastructure**:
- HPC / SLURM (operated ~3M CPU-hour distributed workloads)
- Docker, Git, GitHub Actions (learning)
- FastAPI, SQLite, PostgreSQL (moderate)
- GPU compute (RTX 4070 Super, sentence-transformers, local training)

---

## Experience anchors

**LIGO gravitational wave pipeline (PhD, U Michigan, 2017–2024)**
Built and operated a production hierarchical search pipeline for continuous gravitational waves — a signal processing and statistical inference system spanning ~10¹² parameter combinations, ~3M CPU hours of distributed HPC compute, multi-detector data fusion, automated outlier triage, and sensitivity benchmarking. Led two first-author Physical Review D publications. This is the core experience anchor for ML Eng, Data Eng, and quant roles: large-scale pipelines, rigorous statistical methodology, safety-critical (false alarm rate) validation, and ownership of a complex system from design through peer-reviewed results.

**CGGM postdoc (U Pittsburgh, Jan–Oct 2025)**
Proved mathematically that a Conditional Gaussian Graphical Model decomposes into a product of smaller tractable distributions, reducing computational cost of gene network weight-learning from intractable to feasible. Built ETL pipelines on public genomic datasets (OneK1K). Relevant for: roles requiring mathematical modeling, probabilistic graphical models, high-dimensional statistics, or bioinformatics-adjacent work.

**braintrashtogold (personal project, 2024–present)**
End-to-end LLM fine-tuning and agentic pipeline project: QLoRA fine-tuning of Qwen2.5-3B-Instruct via Unsloth on local GPU, GGUF/Ollama deployment, LangChain-based agentic orchestration, synthetic dataset generation. Relevant for: LLMOps, MLOps, applied AI engineering roles. GitHub: https://github.com/jonwangcw/braintrashtogold

---

## Preferred stack

Strong positive signal if the role involves any of:
- Python-first ML/data stack
- LLM fine-tuning, RLHF, model evaluation, or inference optimization
- Distributed compute, HPC, or large-scale data pipelines
- Probabilistic / statistical modeling (Bayesian, time-series, signal processing)
- FastAPI / Python backend services
- PyTorch ecosystem

Moderate positive signal:
- Rust (interested, no production experience)
- Julia (familiar from physics)
- dbt, Airflow, Spark (adjacent to existing skills, willing to learn fast)

---

## Anti-targets

Exclude the following regardless of score:

- **Weapons systems work**: Roles where the primary work product is a weapons system, targeting system, or direct lethal-capability software — regardless of employer. Defense-adjacent analytics, logistics, intelligence analysis, and operational research roles are acceptable.
- **Pure data analyst roles**: Roles that are primarily dashboarding, Excel/Tableau work, or business reporting without meaningful modeling — even if titled "Data Scientist."
- **Exam-track actuarial roles**: Entry-level actuarial positions that expect exam progress as the primary development path, with no significant statistical modeling component.
- **Roles requiring relocation for any portion of work**: Any posting requiring relocation, extended travel, or regular presence outside Pittsburgh.
- **Pure frontend / product engineering**: Roles where >50% of the work is frontend development, mobile, or non-ML software engineering.
- **MLM / crypto / NFT companies**: Any company whose primary product is a multi-level marketing scheme, speculative crypto asset, or NFT platform.
