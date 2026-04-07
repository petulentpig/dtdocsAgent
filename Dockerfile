FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Playwright browser deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies (including Playwright for remote scraping)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install --with-deps chromium

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY config.py .
COPY agent/ agent/
COPY indexer/ indexer/
COPY scraper/ scraper/
COPY web/ web/

# Data directory (mount a Railway volume here for persistence)
RUN mkdir -p /app/data/raw_pages /app/data/chromadb

ENV PYTHONUNBUFFERED=1

# Railway sets PORT dynamically
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}
