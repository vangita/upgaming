from collections import defaultdict

from app.storage.qdrant_store import SearchHit

# Constant from the original RRF paper (Cormack et al., 2009)
RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[SearchHit]], k: int = RRF_K, limit: int = 10) -> list[SearchHit]:
    scores = defaultdict(float)
    hits_by_id = {}

    for hit_list in result_lists:
        for rank, hit in enumerate(hit_list, start=1):
            scores[hit.id] += 1 / (rank + k)
            hits_by_id.setdefault(hit.id, hit)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)

    results = []
    for doc_id in sorted_ids[:limit]:
        original_hit = hits_by_id[doc_id]
        updated_hit = original_hit.model_copy(update={"score": scores[doc_id]})
        results.append(updated_hit)

    return results
