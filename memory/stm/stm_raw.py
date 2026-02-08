from typing import TypedDict,List
from core.state import STMState,Message
from memory.stm.utils import message_tokens
def update_stm_raw_token_aware(state: STMState, raw_token_budget: int = 50) -> STMState:
    messages = state.get("stm_messages", [])
    raw: List[Message] = []
    total_token = 0

    for msg in reversed(messages):
        msg_tokens = message_tokens(msg)
        if msg_tokens + total_token > raw_token_budget:
            break
        raw.append(msg)
        total_token += msg_tokens

    stm_raw = list(reversed(raw))
    return {"stm_raw":stm_raw}