"""
Fire-and-Forget Memory Write Node
---------------------------------
Fires async memory write at START of turn using your existing STM/LTM graphs.
Non-blocking: Returns immediately, write happens in background.

Uses YOUR existing graph implementations with SEPARATE states:
- memory.stm.build_graph.build_stm_workflow (uses STMState)
- memory.ltm.build_graph.build_ltm_graph (uses LTMState)
"""

import asyncio
import threading
from typing import Optional
from core.state import BaseState, STMState, LTMState
from langgraph.store.memory import InMemoryStore, BaseStore

# Import YOUR existing graph builders
from memory.stm.build_graph import build_stm_workflow
from memory.ltm.build_graph import build_ltm_graph


# Singleton store
_ltm_store: Optional[InMemoryStore] = None

def get_ltm_store() -> InMemoryStore:
    global _ltm_store
    if _ltm_store is None:
        _ltm_store = InMemoryStore()
    return _ltm_store


def _run_stm_sync(stm_state: STMState, store: InMemoryStore):
    """Synchronous STM write in background thread"""
    try:
        print("[STM] 🔄 Starting background write...")
        stm_workflow = build_stm_workflow(store)
        result = stm_workflow.invoke(stm_state)
        print(f"[STM] ✅ Write complete. Summary: {result.get('stm_summary', 'N/A')[:50]}...")
    except Exception as e:
        print(f"[STM] ❌ Background write failed: {e}")
        import traceback
        traceback.print_exc()


def _run_ltm_sync(ltm_state: LTMState, store: InMemoryStore):
    """Synchronous LTM write in background thread"""
    try:
        print("[LTM] 🔄 Starting background write...")
        ltm_workflow = build_ltm_graph(store)
        result = ltm_workflow.invoke(ltm_state)
        print(f"[LTM] ✅ Write complete. Extracted: {result.get('extracted_memories', 'N/A')}")
    except Exception as e:
        print(f"[LTM] ❌ Background write failed: {e}")
        import traceback
        traceback.print_exc()


def _run_memory_writes(stm_state: STMState, ltm_state: LTMState, store: InMemoryStore):
    """Run both STM and LTM writes"""
    _run_ltm_sync(ltm_state, store)
    _run_stm_sync(stm_state, store)
    


def memory_fire_node(state: BaseState, store: BaseStore) -> dict:
    """
    Fire-and-forget memory write at START of turn.
    
    Uses YOUR existing graphs with ISOLATED states:
    - STM graph (STMState): ingest → stm_raw → stm_sum → build_context → update_idx
    - LTM graph (LTMState): extract → dedup → importance → merge → store
    
    Temporary LTM fields (extracted_memories, importance, etc.) stay in LTMState
    and DO NOT pollute your main BaseState.
    
    Non-blocking: fires background thread and returns immediately.
    Returns sync update to stm_raw for instant read availability.
    """
    messages = state.get("messages", [])
    
    # Need at least 2 messages (previous turn's user + AI)
    if len(messages) < 2:
        print("[MEMORY] ⏭️ Skipping - less than 2 messages")
        return {}
    
    # At START of turn: messages[-1] is current human query, messages[-2] is previous AI
    # We want to store the PREVIOUS turn (human + AI pair)
    # So we need at least 3 messages: [old_human, old_ai, current_human]
    if len(messages) < 3:
        print("[MEMORY] ⏭️ Skipping - need at least 3 messages (prev turn + current)")
        return {}
    
    current_human = messages[-1]  # Current user query
    prev_ai = messages[-2]        # Previous AI response  
    prev_human = messages[-3]     # Previous user query
    
    # Verify message types
    if current_human.type != "human" or prev_ai.type != "ai" or prev_human.type != "human":
        print(f"[MEMORY] ⏭️ Skipping - unexpected msg types: {[m.type for m in messages[-3:]]}")
        return {}
    
    # store is now passed as parameter from the compiled graph
    user_msg = prev_human.content  # Previous user message
    ai_msg = prev_ai.content       # Previous AI response
    user_id = state.get("user_id", "default_user")
    
    print(f"[MEMORY] 🚀 Writing previous turn: '{user_msg[:50]}...'")
    
    # Build STMState - ISOLATED from BaseState
    stm_state: STMState = {
        "user_id": user_id,
        "messages": messages,
        "stm_messages": state.get("stm_messages", []),
        "stm_raw": state.get("stm_raw", []),
        "stm_summary": state.get("stm_summary", ""),
        "last_ingested_idx": state.get("last_ingested_idx", 0),
        "last_summarized_idx": state.get("last_summarized_idx", 0),
    }
    
    # Build LTMState - ISOLATED from BaseState  
    ltm_state: LTMState = {
        "user_id": user_id,
        "user_query": f"User: {user_msg}\nAI: {ai_msg}",
        "extracted_memories": None,
        "current_memory": None,
        "duplicate_memory": None,
        "final_memory": None,
        "importance": None
    }
    
    # Fire in background thread - simpler and more reliable than asyncio
    thread = threading.Thread(
        target=_run_memory_writes,
        args=(stm_state, ltm_state, store),
        daemon=True
    )
    thread.start()
    thread.join(timeout=3.0)
    # Sync update: Quick append to stm_raw for immediate read availability
    stm_raw = state.get("stm_raw", [])
    new_raw = stm_raw + [
        {"role": "human", "content": user_msg},
        {"role": "ai", "content": ai_msg}
    ]
    
    # Keep last ~10 messages for quick sync update
    if len(new_raw) > 10:
        new_raw = new_raw[-10:]
    
    return {"stm_raw": new_raw}
