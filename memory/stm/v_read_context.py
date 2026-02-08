"""
Read-Only STM Context Node
--------------------------
Provides stm_context for router and agents without any LLM calls.
Reads stm_summary from store (written by async STM) + stm_raw from state.
"""

from core.state import BaseState
from typing import List, Dict
from langgraph.store.memory import BaseStore


def read_stm_context(state: BaseState, store: BaseStore) -> dict:
    """
    Read-only STM context builder - NO LLM calls.
    
    Builds stm_context from:
    - stm_summary from STORE (written by async STM background task)
    - stm_raw from state (quick sync updates)
    
    This is extremely fast since it only reads, no LLM calls.
    """
    stm_raw = state.get("stm_raw", [])
    user_id = state.get("user_id", "default_user")
    
    # Read summary from store (written by async STM)
    stm_summary = ""
    if store:
        try:
            existing = store.get(("stm", user_id), "summary")
            if existing and hasattr(existing, "value"):
                stm_summary = existing.value.get("text", "")
        except Exception:
            pass
    
    # Fallback to state if store is empty
    if not stm_summary:
        stm_summary = state.get("stm_summary", "")
    
    # Build context: summary + last 6 messages
    context: List[Dict[str, str]] = []
    
    # Add summary as system context if available
    if stm_summary:
        context.append({
            "role": "system",
            "content": f"Conversation summary: {stm_summary}"
        })
    
    # Add last 6 raw messages for recent context
    recent_msgs = stm_raw[-6:] if len(stm_raw) > 6 else stm_raw
    context.extend(recent_msgs)
    
    return {"stm_context": context}
