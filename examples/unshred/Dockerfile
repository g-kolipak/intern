# Unshred website: upload a photo of torn pieces, Drex rebuilds the page.
# Needs NACE_API_KEY set as a service variable (never baked into the image).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY unshred ./unshred
COPY scripts ./scripts
COPY web ./web
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH"
CMD ["sh", "-c", "uvicorn unshred.webapp:app --host 0.0.0.0 --port ${PORT:-8000}"]
