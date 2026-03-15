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

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir playwright \
    && playwright install chromium --with-deps

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
