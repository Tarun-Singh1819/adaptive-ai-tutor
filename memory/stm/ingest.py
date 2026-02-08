from core.state import STMState


def ingest_to_stm_messages(state: STMState):
    messages = state.get("messages", [])
    last_idx = state.get("last_ingested_idx", 0)

    if last_idx >= len(messages):
        return {}

    new = messages[last_idx:]

    stm_messages = state.get("stm_messages", [])
    converted = []

    for m in new:
        if m.type not in ("human", "ai"):
            continue   # skip system, tool, etc.

        converted.append({
            "role": m.type,
            "content": m.content
        })

    return {
        "stm_messages": stm_messages + converted,
        "last_ingested_idx": len(messages)
    }
