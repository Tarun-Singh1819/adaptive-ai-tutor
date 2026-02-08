from langgraph.graph import START,END,StateGraph
from core.state import STMState
from memory.stm.ingest import ingest_to_stm_messages
from memory.stm.stm_raw import update_stm_raw_token_aware
from memory.stm.stm_sum import update_stm_summary_token_aware
from memory.stm.build_context import build_stm_context_token_aware
from memory.stm.upd_lst_idx import update_last_summarized_index
from langgraph.store.memory import InMemoryStore

def build_stm_workflow(store=None):
    stm_graph = StateGraph(STMState)
    stm_graph.add_node("ingest",ingest_to_stm_messages)
    stm_graph.add_node("update_stm_raw_token_aware", update_stm_raw_token_aware)
    stm_graph.add_node("update_stm_summary_token_aware", update_stm_summary_token_aware)
    stm_graph.add_node("build_stm_context_token_aware", build_stm_context_token_aware)
    stm_graph.add_node("update_last_summarized_index", update_last_summarized_index)
    stm_graph.add_edge(START, "ingest")
    stm_graph.add_edge("ingest","update_stm_raw_token_aware")
    stm_graph.add_edge("update_stm_raw_token_aware", "update_stm_summary_token_aware")
    stm_graph.add_edge("update_stm_summary_token_aware", "build_stm_context_token_aware")
    stm_graph.add_edge("build_stm_context_token_aware", "update_last_summarized_index")
    stm_graph.add_edge("update_last_summarized_index", END)

    return stm_graph.compile(store=store)





