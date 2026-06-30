"""AIE WF2026 search — FastAPI + embedded Qdrant Edge (qdrant-edge-py). No cloud.

Real Qdrant Edge runs IN-PROCESS with a HYBRID index:
  - dense vector "bge"  : bge-small-en-v1.5 (fastembed TextEmbedding)
  - sparse vector "bm25": Qdrant/bm25 (fastembed SparseTextEmbedding) + IDF modifier
Keyword search is REAL BM25 inside the engine (proper tokenization/stemming/IDF),
not a Python substring hack. /search returns dense, BM25, and an RRF-fused hybrid
ranking — all from Qdrant Edge.

Run:
  pip install -r requirements.txt
  uvicorn server:app --reload    # open http://localhost:8000
"""
import json
import os
import shutil
import tarfile
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding
from fastapi import FastAPI, Query as Q
from fastapi.responses import JSONResponse, HTMLResponse

import qdrant_edge as qe
from qdrant_edge import (
    Distance, EdgeConfig, EdgeVectorParams, EdgeSparseVectorParams, EdgeShard,
    Point, UpdateOperation, Query, QueryRequest, Prefetch, SparseVector, Modifier,
)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DIM = 384
DENSE, SPARSE = "bge", "bm25"
TOPK = 15
RRF_K = 60
HERE = Path(__file__).parent
# shard/tarball go in a writable dir (HF Spaces app dir is read-only at runtime → set AIE_DATA_DIR=/tmp/aie)
DATA_DIR = Path(os.environ.get("AIE_DATA_DIR", str(HERE)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "aie_edge_index"
TARBALL = DATA_DIR / "aie-edge-index.tar.gz"
RELEASE_URL = os.environ.get("AIE_INDEX_URL",
    "https://github.com/KShivendu/aie-wf2026-qdrant-edge-search/releases/download/v1/aie-edge-index.tar.gz")
STATE = {}
_emb = {}  # lazily-loaded query embedders


def doc_text(t):
    p = [t.get("title") or ""]
    if t.get("track"): p.append("Track: " + t["track"])
    if t.get("speakers"): p.append("Speakers: " + ", ".join(t["speakers"]))
    if t.get("description"): p.append(t["description"])
    return "\n".join(p)[:2000]


def to_sparse(emb):
    return SparseVector(indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values])


# query-term extraction is ONLY for visual highlighting (BM25 does the real scoring)
STOP = set("the a an and or of for to in on with is are be as at by from how what why your you we our this that it its can will into not no using use build building".split())
def hl_terms(q):
    out, seen = [], set()
    for tk in q.lower().replace("/", " ").split():
        tk = "".join(c for c in tk if c.isalnum())
        if len(tk) > 2 and tk not in STOP and tk not in seen:
            seen.add(tk); out.append(tk)
    return out


def get_dense():
    if "dense" not in _emb:
        print(f"[lazy] loading dense query model {DENSE_MODEL} …")
        _emb["dense"] = TextEmbedding(DENSE_MODEL)
    return _emb["dense"]

def get_sparse():
    if "sparse" not in _emb:
        _emb["sparse"] = SparseTextEmbedding(SPARSE_MODEL)
    return _emb["sparse"]


def _build_shard(talks):
    texts = [doc_text(t) for t in talks]
    print(f"[startup] no prebuilt index — embedding {len(texts)} talks (dense + BM25) …")
    dvecs = list(get_dense().embed(texts))
    svecs = list(get_sparse().embed(texts))
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    shard = EdgeShard.create(str(INDEX_DIR), EdgeConfig(
        vectors={DENSE: EdgeVectorParams(size=DIM, distance=Distance.Cosine)},
        sparse_vectors={SPARSE: EdgeSparseVectorParams(modifier=Modifier.Idf)},
    ))
    pts = [
        Point(id=i, vector={DENSE: dv.tolist(), SPARSE: to_sparse(sv)}, payload={
            "title": t.get("title"), "track": t.get("track"), "type": t.get("type"),
            "day": t.get("day"), "time": t.get("time"), "room": t.get("room"),
            "speakers": t.get("speakers") or [], "description": (t.get("description") or "")[:400],
        })
        for i, (t, dv, sv) in enumerate(zip(talks, dvecs, svecs))
    ]
    shard.update(UpdateOperation.upsert_points(pts))
    shard.flush()
    return shard


def ensure_shard(talks):
    """Load the prebuilt Qdrant Edge shard; download it from the GitHub release if
    missing; only re-embed from scratch as a last resort."""
    if INDEX_DIR.exists():
        print(f"[startup] loading prebuilt Qdrant Edge shard ← {INDEX_DIR}")
        return EdgeShard.load(str(INDEX_DIR))
    if not TARBALL.exists() and RELEASE_URL:
        try:
            print(f"[startup] downloading prebuilt index ← {RELEASE_URL}")
            urllib.request.urlretrieve(RELEASE_URL, TARBALL)
        except Exception as e:
            print(f"[startup] release download failed ({e}); will build locally")
    if TARBALL.exists():
        print(f"[startup] unpacking {TARBALL.name}")
        with tarfile.open(TARBALL) as tar:
            tar.extractall(HERE)
        if INDEX_DIR.exists():
            return EdgeShard.load(str(INDEX_DIR))
    return _build_shard(talks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    talks = json.load(open(HERE / "sessions.json"))["sessions"]
    shard = ensure_shard(talks)
    STATE.update(shard=shard, talks=talks)
    print(f"[startup] Qdrant Edge ready — {shard.info().points_count} points. "
          f"Query models load lazily on first /search.")
    yield
    STATE.clear()


app = FastAPI(title="AIE WF2026 · Qdrant Edge hybrid search", lifespan=lifespan)


def _dense_qv(q): return list(get_dense().query_embed(q))[0].tolist()
def _sparse_qv(q): return to_sparse(list(get_sparse().query_embed(q))[0])


def search_dense(q):
    r = STATE["shard"].query(QueryRequest(query=Query.Nearest(_dense_qv(q), using=DENSE), limit=TOPK, with_payload=False))
    return [{"i": int(p.id), "score": float(p.score)} for p in r]

def search_bm25(q):
    r = STATE["shard"].query(QueryRequest(query=Query.Nearest(_sparse_qv(q), using=SPARSE), limit=TOPK, with_payload=False))
    return [{"i": int(p.id), "score": float(p.score)} for p in r]

def search_hybrid(q):
    r = STATE["shard"].query(QueryRequest(
        prefetches=[
            Prefetch(limit=30, query=Query.Nearest(_dense_qv(q), using=DENSE)),
            Prefetch(limit=30, query=Query.Nearest(_sparse_qv(q), using=SPARSE)),
        ],
        query=qe.Fusion.Rrf(RRF_K), limit=TOPK, with_payload=False,
    ))
    return [{"i": int(p.id), "score": float(p.score)} for p in r]


@app.get("/search")
def search(q: str = Q(..., min_length=1)):
    talks = STATE["talks"]
    dense_r, bm25_r, hybrid_r = search_dense(q), search_bm25(q), search_hybrid(q)
    terms = hl_terms(q)

    info = {}
    for rank, r in enumerate(dense_r):
        info.setdefault(r["i"], {})["vector"] = {"rank": rank + 1, "score": round(r["score"], 4)}
    for rank, r in enumerate(bm25_r):
        info.setdefault(r["i"], {})["bm25"] = {"rank": rank + 1, "score": round(r["score"], 4)}
    for rank, r in enumerate(hybrid_r):
        info.setdefault(r["i"], {})["hybrid"] = {"rank": rank + 1, "score": round(r["score"], 4)}

    def cls(e):
        has_v, has_b = "vector" in e, "bm25" in e
        return "both" if (has_v and has_b) else ("vector" if has_v else "bm25")

    items = []
    for i, e in info.items():
        t = talks[i]
        items.append({
            "title": t.get("title"), "track": t.get("track"), "type": t.get("type"),
            "day": t.get("day"), "time": t.get("time"), "room": t.get("room"),
            "speakers": t.get("speakers") or [], "description": (t.get("description") or "")[:280],
            "source": cls(e), "terms": terms, **e,
        })
    # primary order = RRF hybrid rank (the "best of both"); items not in hybrid top-k fall after
    BIG = 10**6
    items.sort(key=lambda x: (x.get("hybrid", {}).get("rank", BIG),
                              -(x.get("vector", {}).get("score") or -1)))
    summary = {
        "total": len(items),
        "both": sum(1 for x in items if x["source"] == "both"),
        "vector_only": sum(1 for x in items if x["source"] == "vector"),
        "bm25_only": sum(1 for x in items if x["source"] == "bm25"),
        "engine": "qdrant-edge hybrid (dense bge + sparse BM25, RRF fused)",
        "ranked_by": "RRF hybrid",
    }
    return JSONResponse({"query": q, "summary": summary, "results": items})


@app.get("/health")
def health():
    return {"ok": "shard" in STATE, "talks": len(STATE.get("talks", [])),
            "engine": "qdrant-edge-py hybrid", "dense_model": DENSE_MODEL, "sparse_model": SPARSE_MODEL}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()
