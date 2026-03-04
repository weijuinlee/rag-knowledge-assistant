FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY ui ./ui

RUN pip install --no-cache-dir -e .
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
