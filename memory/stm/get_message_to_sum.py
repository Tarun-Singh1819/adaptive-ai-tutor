from core.state import STMState,Message
from typing import List

def get_message_to_summarize(state: STMState) -> List[Message]:
    messages = state.get("stm_messages", [])
    raw = state.get("stm_raw", [])
    last_idx = state.get("last_summarized_idx", 0)

    if not raw:
        if last_idx < len(messages):
            return messages[last_idx:]
        return []

    raw_start_idx = len(messages) - len(raw)
    if raw_start_idx <= last_idx:
        return []

    return messages[last_idx:raw_start_idx]
