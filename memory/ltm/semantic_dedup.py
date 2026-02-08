from core.state import LTMState
from typing import Dict,Any
from memory.ltm.utils import get_embeddings,cosine_similarity
from langgraph.store.memory import BaseStore

def SemanticDedupNode(state: LTMState, store: BaseStore) -> Dict[str, Any]:
    user_id = state.get("user_id")
    extracted = state.get("extracted_memories")

    if not extracted:
        return {"current_memory": None, "duplicate_memory": None}

    current_mem = " ".join(extracted) if isinstance(extracted, list) else extracted
    current_mem = current_mem.strip().lower()
    current_emb = get_embeddings(current_mem)

    namespace = ("user", user_id, "details")

    # search may return objects or dicts depending on implementation
    try:
        existing = store.search(namespace)
    except Exception:
        existing = []

    for item in existing:
        # support multiple item types
        mem = None
        if isinstance(item, dict):
            mem = item.get("value") or item.get("data") or item
        else:
            # object-like
            mem = getattr(item, "value", None) or getattr(item, "data", None) or item

        if not mem:
            continue

        old_emb = None
        if isinstance(mem, dict):
            old_emb = mem.get("embedding")
        else:
            old_emb = getattr(mem, "embedding", None)

        if not old_emb:
            continue

        try:
            similarity = cosine_similarity(current_emb, old_emb)
        except Exception:
            similarity = 0.0

        if similarity >= 0.75:
             return {
        "current_memory": current_mem,
        "duplicate_memory": mem,
        "current_embedding": current_emb   # 🔥 ADD THIS
    }

    return {"current_memory": current_mem, "duplicate_memory": None,"current_embedding":current_emb}