from langgraph.store.memory import BaseStore
from memory.ltm.utils import cosine_similarity


def retrieve_ltm(
    store: BaseStore,
    user_id: str,
    query: str,
    embed_fn,
    top_k: int = 5,
    min_importance: float = 0.6
):
    """
    Normal MVP LTM retrieval for your current graph.
    - Uses InMemoryStore
    - User-scoped
    - Semantic similarity + importance
    """

    namespace = ("user", user_id, "details")

    try:
        items = store.search(namespace)
    except Exception:
        return []

    if not items:
        return []

    query_emb = embed_fn(query)

    scored = []

    for item in items:
        mem = item.value   # 🔥 THIS IS IMPORTANT

        importance = mem.get("importance", 0.0)
        if importance < min_importance:
            continue

        emb = mem.get("embedding")
        if not emb:
            continue

        sim = cosine_similarity(query_emb, emb)

        scored.append({
            "data": mem.get("data"),
            "importance": importance,
            "score": sim
        })

    scored.sort(
        key=lambda x: (x["score"], x["importance"]),
        reverse=True
    )

    return scored[:top_k]
