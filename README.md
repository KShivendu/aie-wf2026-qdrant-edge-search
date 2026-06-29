# AIE WF2026 · Qdrant Edge hybrid search

Search the **AI Engineer World's Fair 2026** session catalog (560 talks) with a
**hybrid** retriever running entirely on **embedded Qdrant Edge** (`qdrant-edge-py`)
behind FastAPI. No cloud, no external vector DB — the engine runs in-process.

- **Dense** vector: `BAAI/bge-small-en-v1.5` (fastembed)
- **Sparse** vector: `Qdrant/bm25` (fastembed) with Qdrant's **IDF** modifier — real BM25, in-engine
- **Fusion**: Reciprocal Rank Fusion (RRF) over dense + BM25, computed by Qdrant Edge

Each result is badged by which retriever found it (**Dense / BM25 / Both**) and ranked
by the RRF hybrid score — a live demo of why hybrid beats either alone.

## Run

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

On startup the server embeds all 560 talks (dense + BM25) and builds a Qdrant Edge
shard on local disk. First run downloads the bge-small model (~once, then cached).

## API

| Endpoint | Description |
|---|---|
| `GET /` | Web UI (keyword-vs-vector union, RRF-ranked) |
| `GET /search?q=...` | JSON: dense + BM25 + RRF-hybrid results, badged by source |
| `GET /health` | Index status + model names |

```bash
curl 'http://localhost:8000/search?q=agent%20memory' | jq .summary
```

## How it works

`server.py` builds a Qdrant Edge shard with one dense and one sparse named vector:

```python
EdgeConfig(
    vectors={"bge": EdgeVectorParams(size=384, distance=Distance.Cosine)},
    sparse_vectors={"bm25": EdgeSparseVectorParams(modifier=Modifier.Idf)},
)
```

Hybrid query = two prefetches fused with RRF, all inside Qdrant Edge:

```python
QueryRequest(
    prefetches=[
        Prefetch(limit=30, query=Query.Nearest(dense_qv,  using="bge")),
        Prefetch(limit=30, query=Query.Nearest(sparse_qv, using="bm25")),
    ],
    query=qdrant_edge.Fusion.Rrf(60), limit=15, with_payload=False,
)
```

## Data

`sessions.json` — the WF2026 session catalog (title, description, track, speakers, time).

Built for the Qdrant booth at AI Engineer World's Fair 2026.
