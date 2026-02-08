from typing import TypedDict,Dict
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
def message_tokens(msg: Dict[str, str]) -> int:
    return estimate_tokens(msg.get("content", ""))