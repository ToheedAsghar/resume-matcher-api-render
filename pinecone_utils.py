import os
from typing import List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from pinecone import Pinecone
except Exception:  # pragma: no cover
    Pinecone = None  # type: ignore


EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "jobs-index")


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or Pinecone is None:
        return None
    pc = Pinecone(api_key=api_key)
    try:
        return pc.Index(PINECONE_INDEX)
    except Exception:
        # Attempt to create if missing
        pc.create_index(name=PINECONE_INDEX, dimension=3072, metric="cosine")
        return pc.Index(PINECONE_INDEX)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def upsert_jobs(jobs: List[Dict[str, Any]]):
    index = get_pinecone_index()
    if index is None:
        raise RuntimeError("Pinecone index not available")
    texts = [f"{j.get('title','')} {j.get('company','')} {j.get('description','')} {','.join(j.get('must_have_skills', []))}" for j in jobs]
    vectors = embed_texts(texts)
    items = []
    for j, v in zip(jobs, vectors):
        items.append({
            "id": j["job_id"],
            "values": v,
            "metadata": j,
        })
    index.upsert(vectors=items)


def query_jobs(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    index = get_pinecone_index()
    if index is None:
        raise RuntimeError("Pinecone index not available")
    vector = embed_texts([query_text])[0]
    res = index.query(vector=vector, top_k=top_k, include_metadata=True)
    results = []
    for m in res.matches:
        meta = dict(m.metadata or {})
        meta["_score"] = float(m.score or 0.0)
        results.append(meta)
    return results


