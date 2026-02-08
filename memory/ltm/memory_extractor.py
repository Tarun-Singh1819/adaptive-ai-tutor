from core.state import LTMState,MemoryDecision
from typing import Dict,Any
from langchain_core.messages import SystemMessage
from memory.ltm.utils import memory_extractor


def MemoryExtractorNode(state: LTMState) -> Dict[str, Any]:
    user_id = state.get("user_id")
    last_msg = state.get("user_query", "")

    prompt_system = SystemMessage(
        content=(
            "Extract LONG-TERM memories from the user's message.\n"
            "Only store stable, user-specific info (identity, preferences, ongoing projects).\n"
            "Do NOT store transient info.\n"
            "Return should_write=false if nothing is worth storing.\n"
            "Each memory should be a short atomic sentence."
            "Return each memory as ONE atomic, self-contained fact."
            "Do NOT combine multiple facts in one sentence."

        )
    )

    try:
        decision: MemoryDecision = memory_extractor.invoke(
            [prompt_system, {"role": "user", "content": last_msg}]
        )
    except Exception as e:
        # If the extractor fails, return no extracted memories but don't raise
        return {"extracted_memories": None, "_error": str(e)}

    if decision and decision.should_write:
        return {"extracted_memories": decision.memories}

    return {"extracted_memories": None}