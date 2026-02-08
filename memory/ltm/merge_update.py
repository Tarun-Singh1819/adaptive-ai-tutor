from core.state import LTMState
from typing import Dict,Any
from memory.ltm.utils import is_contradiction

def MergeUpdateNode(state: LTMState) -> Dict[str, Any]:
    current_mem = state.get("current_memory")
    duplicate_mem = state.get("duplicate_memory")
    importance = state.get("importance")   # 🔥 CRITICAL

    if not current_mem:
        return {
            "final_memory": None,
            
        }

    if not duplicate_mem:
        return {
            "final_memory": current_mem,
            
        }

    old_text = duplicate_mem.get("data")

    if is_contradiction(current_mem, old_text):
        return {
            "final_memory": current_mem,
           
        }

    return {
        "final_memory": old_text,
       
    }
