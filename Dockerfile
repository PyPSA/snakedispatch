# syntax=docker/dockerfile:1

# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
COPY .git/ .git/
COPY app/ app/

RUN uv sync --frozen --no-dev --all-extras

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends sudo curl ca-certificates && rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser -g 1000 && \
    useradd -r -u 1000 -g appuser -m -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: /usr/bin/apt-get" >> /etc/sudoers.d/appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /data

COPY --from=builder --chown=appuser:appuser /root/.local/bin/uv /usr/local/bin/uv
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/app /app/app
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
