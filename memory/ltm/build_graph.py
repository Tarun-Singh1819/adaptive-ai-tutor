from memory.ltm.memory_extractor import MemoryExtractorNode
from memory.ltm.semantic_dedup import SemanticDedupNode
from memory.ltm.importance_score import ImportanceScorerNode
from memory.ltm.merge_update import MergeUpdateNode
from memory.ltm.storeNode import StoreNode

from langgraph.graph import StateGraph, START, END
from core.state import LTMState

def build_ltm_graph(store):
    ltm_graph = StateGraph(LTMState)

    # ========= Nodes =========
    ltm_graph.add_node("Memory_Ex", MemoryExtractorNode)
    ltm_graph.add_node("Semantic_Node", SemanticDedupNode)
    ltm_graph.add_node("Imp_sc", ImportanceScorerNode)
    ltm_graph.add_node("MergeUpdate", MergeUpdateNode)
    ltm_graph.add_node("StoreNode", StoreNode)

    # ========= Edges =========
    ltm_graph.add_edge(START, "Memory_Ex")

    # ---- conditional routing ----
    def should_continue(state: LTMState):
        return "write" if state.get("extracted_memories") else "end"

    ltm_graph.add_conditional_edges(
        "Memory_Ex",
        should_continue,
        {
            "write": "Semantic_Node",
            "end": END,
        }
    )

    # ---- sequential flow: importance MUST be calculated before merge ----
    ltm_graph.add_edge("Semantic_Node", "Imp_sc")
    ltm_graph.add_edge("Imp_sc", "MergeUpdate")

    # ---- final write ----
    ltm_graph.add_edge("MergeUpdate", "StoreNode")
    ltm_graph.add_edge("StoreNode", END)

    # ========= Compile with store =========
    return ltm_graph.compile(store=store)


