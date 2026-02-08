from core.cache import RAG_CACHE,MAX_PAPERS_IN_CACHE

def touch_cache(paper_id: str):
    """
    Mark paper as recently used.
    If cache size exceeds limit, evict LRU paper.
    """
    # move accessed paper to most-recent position
    RAG_CACHE.move_to_end(paper_id)

    # evict if overflow
    if len(RAG_CACHE) > MAX_PAPERS_IN_CACHE:
        evicted_paper_id, _ = RAG_CACHE.popitem(last=False)
        print(f"🧹 LRU Evicted: {evicted_paper_id}")
