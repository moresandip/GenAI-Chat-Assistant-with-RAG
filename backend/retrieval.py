import json
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def search_similarity(db, query_embedding: List[float], top_k: int = 3, threshold: float = 0.70) -> List[Dict[str, Any]]:
    """
    Computes cosine similarity between query embedding and database chunks.
    Filters by similarity threshold and returns the top_k matching chunks.
    """
    chunks = db.get_all_chunks()
    if not chunks:
        logger.warning("No chunks found in database during similarity search.")
        return []

    query_vector = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    
    if query_norm == 0:
        logger.error("Query embedding has magnitude 0.")
        return []

    results = []
    for chunk in chunks:
        chunk_vector = np.array(json.loads(chunk["embedding"]), dtype=np.float32)
        chunk_norm = np.linalg.norm(chunk_vector)
        
        if chunk_norm == 0:
            continue

        # Dot Product divided by magnitudes (Cosine Similarity)
        dot_product = np.dot(query_vector, chunk_vector)
        similarity = float(dot_product / (query_norm * chunk_norm))

        # Keep only chunks matching threshold
        if similarity >= threshold:
            results.append({
                "chunk_id": chunk["id"],
                "document_title": chunk["doc_title"],
                "text": chunk["text"],
                "similarity": similarity
            })

    # Sort descending by similarity score
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
