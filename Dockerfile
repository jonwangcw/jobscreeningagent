# Stage 1: build the React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# System packages needed by Playwright's Chromium and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium runtime dependencies
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    # Sentence-transformers / torch build deps
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (includes playwright and pydantic)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Install Playwright browser (Chromium only — smallest footprint)
RUN playwright install chromium

# Copy application source
COPY agent/ ./agent/
COPY api/ ./api/
COPY config.yml ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Profile dir is mounted at runtime (read-only)
# outputs/ and data/ are mounted at runtime

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
