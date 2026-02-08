from core.state import STMState

def update_last_summarized_index(state: STMState) -> STMState:
    messages = state.get("stm_messages", [])
    raw = state.get("stm_raw", [])
    if not raw:
        return {"last_summarized_idx":len(messages)}
    return {"last_summarized_idx":len(messages) - len(raw)}