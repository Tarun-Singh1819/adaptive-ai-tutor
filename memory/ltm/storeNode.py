from core.state import LTMState
from typing import Dict,Any
import uuid
import time
import math
from datetime import datetime, timezone

from langgraph.store.memory import BaseStore

def StoreNode(state: LTMState, store: BaseStore) -> Dict[str, Any]:
    user_id = state.get("user_id")
    final_mem = state.get("final_memory")
    importance = state.get("importance")
    embedding = state.get("current_embedding")

    if not final_mem or importance is None:
        return {}

    if importance < 0.6:
        return {}

    namespace = ("user", user_id, "details")
    mem_text = final_mem.strip().lower()
    mem_id = str(uuid.uuid5(uuid.NAMESPACE_OID, mem_text))

    now = datetime.now(timezone.utc).isoformat()

    try:
        existing_item = store.get(namespace, mem_id)
    except Exception:
        existing_item = None

    # Extract payload safely
    existing = (
        existing_item.value
        if existing_item is not None and hasattr(existing_item, "value")
        else None
    )

    # Skip identical overwrite
    if existing and existing.get("data") == mem_text:
        return {}

    created_at = existing.get("created_at") if existing else now

    payload = {
        "data": mem_text,
        "importance": importance,
        "embedding": embedding,
        "created_at": created_at,
    }

    print(f"[LTM StoreNode] 💾 Storing memory: '{mem_text[:50]}...' importance={importance}")
    store.put(namespace, mem_id, payload)
    return {"stored_memory": mem_text}
