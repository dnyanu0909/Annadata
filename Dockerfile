FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight inference dependencies
COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code and class mapping
COPY api/ api/
COPY artifacts/ artifacts/

# Create empty models directory for runtime download
RUN mkdir -p models

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
