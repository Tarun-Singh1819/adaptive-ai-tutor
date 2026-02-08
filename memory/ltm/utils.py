import math
from typing import List
from adaptive_tutor.models import EMBED_MODEL,LLM
from core.state import LTMState,BaseState,MemoryDecision

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_embeddings(text: str) -> List[float]:
    if not text:
        return []
    return EMBED_MODEL.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def is_contradiction(new: str, old: str) -> bool:
    negations = [
        "not", "no longer", "stopped", "quit",
        "nahi", "ab nahi", "band"
    ]

    for n in negations:
        if n in new and n.replace(" ", "") not in old:
            return True

    return False

def should_continue(state: LTMState):
    return "write" if state.get("extracted_memories") else "end"


memory_extractor = LLM.with_structured_output(MemoryDecision)