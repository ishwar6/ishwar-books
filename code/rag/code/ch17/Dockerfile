# Chapter 17 - ONE image for the API. Build from the repo root:
#   docker build -f code/ch17/Dockerfile -t lumora-rag-api .
FROM python:3.12-slim

# uv gives reproducible installs from uv.lock (same versions as your laptop)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first (cached layer), source second (changes often).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY ragbook ./ragbook
COPY code ./code
COPY data ./data
RUN uv sync --frozen --no-dev

# Never run as root inside the container.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "code.ch17.app:app", "--host", "0.0.0.0", "--port", "8000"]
