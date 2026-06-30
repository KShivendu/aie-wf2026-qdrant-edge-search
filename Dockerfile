# Generic container for the AIE WF2026 Qdrant Edge hybrid search server.
# Works on HF Spaces (port 7860), Cloud Run, Fly, Render, plain VM, etc.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py index.html sessions.json ./

# Writable locations (HF Spaces app dir is read-only at runtime; /tmp is writable)
ENV AIE_DATA_DIR=/tmp/aie \
    FASTEMBED_CACHE_PATH=/tmp/fastembed_cache \
    HF_HOME=/tmp/hf

# Pre-warm the embedding models into the image so the first query isn't slow.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')" || true

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
