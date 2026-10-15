# One image, three roles (selected by CMD / Cloud Run command):
#   api      uvicorn supportagent.api.main:app        (default)
#   tools    python -m supportagent.tools.mcp_server --http
#   worker   python -m supportagent.temporal.worker
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TRACE_DIR=/tmp/traces PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-deps -e .

RUN useradd -m app
USER app

EXPOSE 8080
# Cloud Run sets $PORT; one worker per container, scale with instances.
CMD ["sh", "-c", "uvicorn supportagent.api.main:app --host 0.0.0.0 --port ${PORT}"]
