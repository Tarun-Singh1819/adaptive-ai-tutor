"""
v_build_graph.py - New version with Start-of-Turn Memory Write pattern

Flow: START → memory_fire (async) → read_stm → router → agent → END

Memory writes happen at START using only 2 messages (last user + last AI).
This is non-blocking - router runs immediately while memory writes in background.
"""
from langgraph.graph import StateGraph, START, END
from core.state import BaseState
from agents.router import QueryRouter
from agents.overview_agent import OverView
from agents.research_agent import research_agent
from agents.doubt_agent import doubt_handling
from agents.router_node import make_router_node
from memory.v_fire_memory import memory_fire_node, get_ltm_store
from memory.stm.v_read_context import read_stm_context
from adaptive_tutor.models import LLM


def build_graph():
    """
    Main graph with Start-of-Turn Memory Write pattern.
    
    Flow: START → memory_fire (async) → read_stm → router → agent → END
    
    Memory writes happen at START using only 2 messages (last user + last AI).
    This is non-blocking - router runs immediately while memory writes in background.
    read_stm provides stm_context for router/agents without LLM calls.
    """
    llm = LLM
    router = QueryRouter(llm=llm)

    graph = StateGraph(BaseState)
    
    # Nodes
    graph.add_node("memory_fire", memory_fire_node)  # Fire-and-forget async write
    graph.add_node("read_stm", read_stm_context)     # Read-only STM context (no LLM)
    graph.add_node("router", make_router_node(router))
    graph.add_node("overview", OverView)
    graph.add_node("doubt", doubt_handling)
    graph.add_node("research", research_agent)

    # Edges: START → memory_fire → read_stm → router → agent → END
    graph.add_edge(START, "memory_fire")
    graph.add_edge("memory_fire", "read_stm")
    graph.add_edge("read_stm", "router")
   
    graph.add_conditional_edges(
        "router",
        lambda state: str(state.get("routed_to", "research")),
        {
            "overview": "overview",
            "doubt": "doubt",
            "research": "research",
        }
    )

    graph.add_edge("overview", END)
    graph.add_edge("doubt", END)
    graph.add_edge("research", END)

    store = get_ltm_store()
    return graph.compile(store=store)
