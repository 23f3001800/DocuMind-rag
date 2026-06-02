import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Iterable
from config import settings
from embeddings import embed_texts

# Persistent ChromaDB client
_client = None


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def add_chunks(chunks: Iterable[Dict], collection_name: str = "default") -> int:
    """Embed and store chunks in ChromaDB collection."""
    client = get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    batch = []
    total_added = 0

    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            _process_batch(collection, batch)
            total_added += len(batch)
            batch = []

    if batch:
        _process_batch(collection, batch)
        total_added += len(batch)

    return total_added

def _process_batch(collection, batch: List[Dict]):
    texts = [c["text"] for c in batch]
    ids = [c["chunk_id"] for c in batch]
    metadatas = [
        {"source": c["source"], "page": c["page"]}
        for c in batch
    ]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_dense(
    query_embedding: List[float],
    collection_name: str = "default",
    top_k: int = 10,
) -> List[Dict]:
    """Dense retrieval from ChromaDB."""
    client = get_client()
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return []

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "score": 1 - dist,  # cosine similarity
        })
    return hits


def list_collections() -> List[str]:
    client = get_client()
    return [c.name for c in client.list_collections()]