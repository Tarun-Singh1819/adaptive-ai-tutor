from core.state import STMState
from memory.stm.get_message_to_sum import get_message_to_summarize
from adaptive_tutor.models import LLM
from langgraph.store.memory import BaseStore


def update_stm_summary_token_aware(
    state: STMState,
    store: BaseStore,
    summary_token_budget: int = 300
) -> STMState:
    """
    Updates STM summary and writes to store for cross-state access.
    
    Writes summary to store namespace: ("stm", user_id, "summary")
    This allows v_read_context to fetch summary even though STMState is temp.
    """
    llm = LLM
    to_summarize = get_message_to_summarize(state)
    if not to_summarize:
        return {}

    if len(to_summarize) < 2:
        return {}

    messages = state.get("stm_messages", [])
    raw = state.get("stm_raw", [])

    raw_start_idx = len(messages) - len(raw)

    old_summary = state.get("stm_summary", "")
    
    # Also try to fetch from store if state is empty
    user_id = state.get("user_id", "default_user")
    if not old_summary and store:
        try:
            existing = store.get(("stm", user_id), "summary")
            if existing and hasattr(existing, "value"):
                old_summary = existing.value.get("text", "")
        except Exception:
            pass
    
    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

    old_summary = old_summary[-1000:]

    prompt = f"""
You are maintaining short-term memory for an adaptive tutor.

Existing summary:
{old_summary}

Evicted conversation messages:
{convo_text}

Update the summary concisely.
Focus on:
- Topics learned
- User understanding & weak areas
- Learning preferences
- Current goals

Do NOT store emotional or transient states unless recurring.
Do NOT include code.
Keep it under {summary_token_budget} tokens.
"""

    try:
        result = llm.invoke(prompt)
        new_summary = result.content.strip()
    except Exception:
        return {}

    # Write summary to store for cross-state access
    if store:
        try:
            store.put(
                ("stm", user_id),  # namespace
                "summary",         # key
                {"text": new_summary}  # only text needed for reading
            )
        except Exception as e:
            print(f"[STM] Failed to write summary to store: {e}")

    return {
        "stm_summary": new_summary,
        "summary_updated": True,
        "summary_upto_idx": raw_start_idx
    }
