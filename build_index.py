"""Precompute the Qdrant Edge hybrid index and pack it as a release artifact.

Embeds all 560 talks (dense bge-small + sparse BM25), persists a Qdrant Edge shard
to ./aie_edge_index, and tars it to ./aie-edge-index.tar.gz. Upload that tarball as
a GitHub Release asset (kept OUT of git history); the server downloads + loads it
instead of re-embedding on every boot.

  python build_index.py
  gh release create v1 aie-edge-index.tar.gz --title "Prebuilt Qdrant Edge index" --notes "560 talks, dense bge-small + sparse BM25"
"""
import json
import shutil
import tarfile
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_edge import (
    Distance, EdgeConfig, EdgeVectorParams, EdgeSparseVectorParams, EdgeShard,
    Point, UpdateOperation, SparseVector, Modifier,
)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DIM = 384
DENSE, SPARSE = "bge", "bm25"
HERE = Path(__file__).parent
INDEX_DIR = HERE / "aie_edge_index"
TARBALL = HERE / "aie-edge-index.tar.gz"


def doc_text(t):
    p = [t.get("title") or ""]
    if t.get("track"): p.append("Track: " + t["track"])
    if t.get("speakers"): p.append("Speakers: " + ", ".join(t["speakers"]))
    if t.get("description"): p.append(t["description"])
    return "\n".join(p)[:2000]


def to_sparse(emb):
    return SparseVector(indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values])


def main():
    talks = json.load(open(HERE / "sessions.json"))["sessions"]
    texts = [doc_text(t) for t in talks]
    print(f"dense embed ({DENSE_MODEL}) …")
    dvecs = list(TextEmbedding(DENSE_MODEL).embed(texts))
    print(f"sparse BM25 embed ({SPARSE_MODEL}) …")
    svecs = list(SparseTextEmbedding(SPARSE_MODEL).embed(texts))

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
    shard.close()
    print(f"persisted shard → {INDEX_DIR} ({len(pts)} points)")

    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(INDEX_DIR, arcname=INDEX_DIR.name)
    print(f"packed → {TARBALL} ({TARBALL.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
