from core.state import Message, STMState
from typing import TypedDict, List

def build_stm_context_token_aware(state: STMState) -> dict:
    context: List[Message] = []
    summary = state.get("stm_summary", "")
    if summary:
        context.append({"role": "system", "content": f"Conversation summary:\n{summary}"})
    context.extend(state.get("stm_raw", []))
    #state["stm_context"] = context
    return {"stm_context":context}
