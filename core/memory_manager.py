
import os
import uuid
import time
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, TypedDict
from dotenv import load_dotenv

from pydantic import BaseModel, Field


# Langchain / langgraph imports (keep these as in your environment)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


load_dotenv()

# %%
class MemoryDecision(BaseModel):
    should_write: bool=Field(description="Wheter to stay this in memory or not")
    memories:List[str]=Field(default_factory=list,description="Atomic user memories to store")

# %%

class MemoryState(TypedDict, total=False):
    user_id: str
    user_message: str


    extracted_memories: Optional[List[str]]


    current_memory: Optional[str]
    duplicate_memory: Optional[Dict[str, Any]]


    final_memory: Optional[str]
    importance: Optional[float]

# %%
# HUGGINGFACEHUB_API_TOKEN is loaded from .env file via load_dotenv()

# %%
_embeddings_client = HuggingFaceEndpointEmbeddings(
model="sentence-transformers/all-MiniLM-L6-v2"
)

# %%
def get_embeddings(text: str) -> List[float]:
    """Return an embedding vector for the text."""
    if not text:
        return []
    return _embeddings_client.embed_query(text)

# %%
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# %%
store = InMemoryStore()

# %%
extract_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
memory_extractor = extract_llm.with_structured_output(MemoryDecision)

# %%
def MemoryExtractorNode(state: MemoryState) -> Dict[str, Any]:
    user_id = state.get("user_id")
    last_msg = state.get("user_message", "")

    prompt_system = SystemMessage(
        content=(
            "Extract LONG-TERM memories from the user's message.\n"
            "Only store stable, user-specific info (identity, preferences, ongoing projects).\n"
            "Do NOT store transient info.\n"
            "Return should_write=false if nothing is worth storing.\n"
            "Each memory should be a short atomic sentence."
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

def should_continue(state: MemoryState):
    return "write" if state.get("extracted_memories") else "end"


# %%
def SemanticDedupNode(state: MemoryState, store: BaseStore) -> Dict[str, Any]:
    user_id = state.get("user_id")
    extracted = state.get("extracted_memories")

    if not extracted:
        return {"current_memory": None, "duplicate_memory": None}

    current_mem = " ".join(extracted) if isinstance(extracted, list) else extracted
    current_mem = current_mem.strip().lower()
    current_emb = get_embeddings(current_mem)

    namespace = ("user", user_id, "details")

    # search may return objects or dicts depending on implementation
    try:
        existing = store.search(namespace)
    except Exception:
        existing = []

    for item in existing:
        # support multiple item types
        mem = None
        if isinstance(item, dict):
            mem = item.get("value") or item.get("data") or item
        else:
            # object-like
            mem = getattr(item, "value", None) or getattr(item, "data", None) or item

        if not mem:
            continue

        old_emb = None
        if isinstance(mem, dict):
            old_emb = mem.get("embedding")
        else:
            old_emb = getattr(mem, "embedding", None)

        if not old_emb:
            continue

        try:
            similarity = cosine_similarity(current_emb, old_emb)
        except Exception:
            similarity = 0.0

        if similarity >= 0.75:
            return {"current_memory": current_mem, "duplicate_memory": mem}

    return {"current_memory": current_mem, "duplicate_memory": None}

# %%
def MergeUpdateNode(state: MemoryState) -> Dict[str, Any]:
    current_mem = state.get("current_memory")
    duplicate_mem = state.get("duplicate_memory")

    if not current_mem:
        return {"final_memory": None}

    # Placeholder: if duplicate exists you might append or refine
    # For now keep the new memory as final (you can change this logic later)
    return {"final_memory": current_mem}

# %%
def ImportanceScorerNode(state: MemoryState) -> Dict[str, Any]:
    final_mem = state.get("final_memory")
    if not final_mem:
        return {"importance": None}

    text = final_mem.lower()
    score = 0.5

    identity_keywords = ["name", "background", "student", "professional", "experience", "identity"]
    if any(k in text for k in identity_keywords):
        score = max(score, 0.95)

    skill_keywords = ["learning", "studying", "beginner", "intermediate", "advanced", "progressing", "understands", "struggles"]
    if any(k in text for k in skill_keywords):
        score = max(score, 0.9)

    goal_keywords = ["goal", "aim", "wants to", "planning to", "building", "working on", "project"]
    if any(k in text for k in goal_keywords):
        score = max(score, 0.85)

    preference_keywords = ["likes", "prefers", "enjoys", "comfortable with", "doesn't like"]
    if any(k in text for k in preference_keywords):
        score = max(score, 0.75)

    transient_keywords = ["today", "now", "currently", "right now", "feels", "tired", "hungry"]
    if any(k in text for k in transient_keywords):
        score = min(score, 0.4)

    score = max(0.0, min(score, 1.0))
    return {"importance": score}


# %%
def StoreNode(state: MemoryState, store: BaseStore) -> Dict[str, Any]:
    user_id = state.get("user_id")
    final_mem = state.get("final_memory")
    importance = state.get("importance")

    if not final_mem or importance is None:
        return {}

    namespace = ("user", user_id, "details")
    mem_text = final_mem.strip().lower()
    mem_id = uuid.uuid5(uuid.NAMESPACE_OID, mem_text)

    now = datetime.now(timezone.utc).isoformat()

    # Use string key for store
    try:
        existing = store.get(namespace, str(mem_id))
    except Exception:
        existing = None

    if existing:
        created_at = existing.get("created_at") if isinstance(existing, dict) else None
    else:
        created_at = now

    payload = {
        "data": mem_text,
        "importance": importance,
        "embedding": get_embeddings(mem_text),
        "created_at": created_at,
        "last_accessed": now,
    }

    store.put(namespace, str(mem_id), payload)
    return {}

# %%
ltm_graph = StateGraph(MemoryState)
ltm_graph.add_node("Memory_Ex", MemoryExtractorNode)
ltm_graph.add_node("Semantic_Node", SemanticDedupNode)
ltm_graph.add_node("MergeUpdate", MergeUpdateNode)
ltm_graph.add_node("Imp_sc", ImportanceScorerNode)
ltm_graph.add_node("StoreNode", StoreNode)

ltm_graph.add_edge(START, "Memory_Ex")
ltm_graph.add_conditional_edges(
    "Memory_Ex",
    should_continue,
    {
        "write": "Semantic_Node",
        "end": END,
    }
)

ltm_graph.add_edge("Semantic_Node", "MergeUpdate")
ltm_graph.add_edge("MergeUpdate", "Imp_sc")
ltm_graph.add_edge("Imp_sc", "StoreNode")
ltm_graph.add_edge("StoreNode", END)

ltm_workflow = ltm_graph.compile(store=store)

# %%
ltm_workflow

# %%
class Message(TypedDict):
    role: str
    content: str


# %%
class STMState(TypedDict, total=False):
    messages: List[Message]
    stm_raw: List[Message]
    stm_summary: str
    last_summarized_idx: int
    stm_context: List[Message]

# %%
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_tokens(msg: Dict[str, str]) -> int:
    return estimate_tokens(msg.get("content", ""))

# %%
def update_stm_raw_token_aware(state: STMState, raw_token_budget: int = 10) -> STMState:
    messages = state.get("messages", [])
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

# %%
def get_message_to_summarize(state: STMState) -> List[Message]:
    messages = state.get("messages", [])
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

# %%
def update_stm_summary_token_aware(state: STMState, summary_token_budget: int = 300) -> STMState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    to_summarize = get_message_to_summarize(state)
    if not to_summarize:
        return {}

    old_summary = state.get("stm_summary", "")
    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

    prompt = f"""
You are maintaining short-term memory for an adaptive tutor.

Existing summary:
{old_summary}

Evicted conversation messages:
{convo_text}

Update the summary concisely.
Focus on:
- Topics learned
- User understanding & weak areas
- Learning preferences
- Current goals

Do NOT include code.
Keep it under {summary_token_budget} tokens.
"""

    try:
        result = llm.invoke(prompt)
        new_summary = result.content.strip()
    except Exception:
        new_summary = old_summary

  
    return {"stm_summary":new_summary}

# %%
def build_stm_context_token_aware(state: STMState) -> STMState:
    context: List[Message] = []
    summary = state.get("stm_summary", "")
    if summary:
        context.append({"role": "system", "content": f"Conversation summary:\n{summary}"})
    context.extend(state.get("stm_raw", []))
    #state["stm_context"] = context
    return {"stm_context":context}


def update_last_summarized_index(state: STMState) -> STMState:
    messages = state.get("messages", [])
    raw = state.get("stm_raw", [])
    if not raw:
        state["last_summarized_idx"] = len(messages)
        return {}
    return {"last_summarized_idx":len(messages) - len(raw)}

# %%
stm_graph = StateGraph(STMState)
stm_graph.add_node("update_stm_raw_token_aware", update_stm_raw_token_aware)
stm_graph.add_node("update_stm_summary_token_aware", update_stm_summary_token_aware)
stm_graph.add_node("build_stm_context_token_aware", build_stm_context_token_aware)
stm_graph.add_node("update_last_summarized_index", update_last_summarized_index)

stm_graph.add_edge(START, "update_stm_raw_token_aware")
stm_graph.add_edge("update_stm_raw_token_aware", "update_stm_summary_token_aware")
stm_graph.add_edge("update_stm_summary_token_aware", "build_stm_context_token_aware")
stm_graph.add_edge("build_stm_context_token_aware", "update_last_summarized_index")
stm_graph.add_edge("update_last_summarized_index", END)

stm_workflow = stm_graph.compile(store=store)

# %%
stm_workflow

# %%
from functools import partial

class AgentState(TypedDict, total=False):
    user_id: str
    messages: List[Dict[str, Any]]
    query:str
    stm: Dict[str, Any]
    ltm: Dict[str, Any]
    assistant_answer: str
    stm_context:List[Dict[str,str]]
    ltm_memories:List[Dict[str,Any]]
    final_prompt_messages:List[Dict[str,str]]


def V1memory_manager_node(state: AgentState, stm_workflow, ltm_workflow) -> AgentState:
    # Run STM subgraph (should not block main reply)
    try:
        stm_state = state.get("stm", {"messages": []})
        updated_stm = stm_workflow.invoke(stm_state)
        state["stm"] = updated_stm
    except Exception as e:
        state.setdefault("memory_errors", []).append({"source": "STM", "error": str(e)})

    # Run LTM subgraph
    try:
        ltm_state = state.get("ltm", {})
        # ensure required fields exist
        if "user_id" not in ltm_state and "user_id" in state:
            ltm_state["user_id"] = state["user_id"]
        if "user_message" not in ltm_state and state.get("messages"):
            # the latest user message
            last_user = next((m for m in reversed(state["messages"]) if m.get("role") == "user"), None)
            ltm_state["user_message"] = last_user.get("content") if last_user else ltm_state.get("user_message")

        updated_ltm = ltm_workflow.invoke(ltm_state)
        state["ltm"] = updated_ltm
    except Exception as e:
        state.setdefault("memory_errors", []).append({"source": "LTM", "error": str(e)})

    return state
def memory_manager_node(state: AgentState, stm_workflow, ltm_workflow) -> AgentState:
    # -------- STM --------
    try:
        stm_state = state.get("stm", {
            "messages": [],
            "stm_raw": [],
            "stm_summary": "",
            "last_summarized_idx": 0
        })

        # 🔥 MOST IMPORTANT LINE
        stm_state["messages"] = state.get("messages", [])

        updated_stm = stm_workflow.invoke(stm_state)
        state["stm"] = updated_stm

    except Exception as e:
        state.setdefault("memory_errors", []).append(
            {"source": "STM", "error": str(e)}
        )

    # -------- LTM --------
    try:
        ltm_state = {}

        ltm_state["user_id"] = state.get("user_id")

        last_user = next(
            (m for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
            None
        )
        if last_user:
            ltm_state["user_message"] = last_user["content"]

        updated_ltm = ltm_workflow.invoke(ltm_state)
        state["ltm"] = updated_ltm

    except Exception as e:
        state.setdefault("memory_errors", []).append(
            {"source": "LTM", "error": str(e)}
        )

    return state


# %%
memory_manager = partial(memory_manager_node, stm_workflow=stm_workflow, ltm_workflow=ltm_workflow)

# Expose top-level workflows / helper functions
__all__ = [
    "ltm_workflow",
    "stm_workflow",
    "memory_manager",
    "store",
]

# %%
def run_test():
    state = {
        "user_id": "user_1",
        "messages": [],
        "stm": {
            "messages": [],
            "stm_raw": [],
            "stm_summary": "",
            "last_summarized_idx": 0
        },
        "ltm": {}
    }

    user_inputs = [
        "I am a student learning machine learning.",
        "I am currently working on an AI agent project.",
        "I prefer learning by building projects.",
        "Today I am feeling tired."  # should NOT be stored
        ""
    ]

    for msg in user_inputs:
        state["messages"].append({"role": "user", "content": msg})
        state = memory_manager(state)
        print("----")

    return state


# %%
final_state = run_test()


# %%
final_state["messages"]

# %%
print("\nSTM SUMMARY:")
print(final_state["stm"].get("stm_summary"))

print("\nSTM RAW:")
for m in final_state["stm"].get("stm_raw", []):
    print(m)


# %%
print("\nLTM STORED MEMORIES:")
items = store.search(("user", "user_1", "details"))
for item in items:
    print(item.value)


# %%
def STMContextNode(state: AgentState) -> AgentState:
    stm = state.get("stm", {})
    state["stm_context"] = stm.get("stm_context", [])
    return state


# %%
def mark_memories_accessed(
    store: BaseStore,
    namespace: tuple,
    items: List[Any]
):
    now = datetime.now(timezone.utc).isoformat()

    for item in items:
        key = item.key
        value = dict(item.value)   # 🔥 copy to avoid mutation issues
        value["last_accessed"] = now

        store.put(namespace, key, value)


# %%
def retrieve_memories(
    store: BaseStore,
    user_id: str,
    query: str,
    top_k: int = 3,
    alpha: float = 0.7,   # similarity weight
    beta: float = 0.3    # importance weight
):
    """
    Retrieve top-k LTM memories using:
    final_score = alpha * similarity + beta * importance
    """

    namespace = ("user", user_id, "details")

    try:
        items = store.search(namespace)
    except Exception:
        items = []

    if not items:
        return []

    query_emb = get_embeddings(query.lower().strip())
    scored = []

    for item in items:
        mem = item.value
        if not isinstance(mem, dict):
            continue

        mem_emb = mem.get("embedding")
        if not mem_emb:
            continue

        similarity = cosine_similarity(query_emb, mem_emb)
        importance = mem.get("importance", 0.5)

        final_score = alpha * similarity + beta * importance

        scored.append({
            "item": item,              # keep reference
            "data": mem.get("data"),
            "similarity": similarity,
            "importance": importance,
            "score": final_score
        })

    if not scored:
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_items = scored[:top_k]

    # 🔥 mark memories as accessed (for decay / freshness)
    mark_memories_accessed(
        store,
        namespace,
        [x["item"] for x in top_items]
    )

    return [
        {
            "data": x["data"],
            "similarity": round(x["similarity"], 3),
            "importance": round(x["importance"], 2),
            "score": round(x["score"], 3)
        }
        for x in top_items
    ]


# %%
def LTMContextNode(state: AgentState, store) -> AgentState:
    user_id = state.get("user_id")
    query = state.get("query")

    if not user_id or not query:
        state["ltm_memories"] = []
        return state

    memories = retrieve_memories(
        store=store,
        user_id=user_id,
        query=query,
        top_k=3
    )

   #  state["ltm_memories"] = memories
    return {"ltm_memories":memories}




# %%
def AnswerNode(state: AgentState) -> AgentState:
    stm_context = state.get("stm_context", [])
    ltm_memories = state.get("ltm_memories", [])

    messages = []

    # System guardrails
    messages.append({
        "role": "system",
        "content": (
            "You are an adaptive AI assistant.\n"
            "Prioritize recent conversation context.\n"
            "User memory is background information only.\n"
            "Do NOT mention memory explicitly.\n"
            "If memory conflicts with conversation, prefer conversation."
        )
    })

    # LTM background (low priority)
    if ltm_memories:
        memory_block = "\n".join(f"- {m['data']}" for m in ltm_memories)
        messages.append({
            "role": "system",
            "content": f"User background:\n{memory_block}"
        })

    # STM context (high priority)
    messages.extend(stm_context)

    # Current user query
    messages.append({
        "role": "user",
        "content": state.get("query", "")
    })

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    response = llm.invoke(messages)
   # state["assistant_answer"] = response.content
    #state["final_prompt_messages"] = messages
    return {
        "assistant_answer":response.content,
        "final_prompt_messages":messages
    }


# %%
graph = StateGraph(AgentState)

graph.add_node("memory_manager", memory_manager)
graph.add_node("stm_context", STMContextNode)
graph.add_node("ltm_context", partial(LTMContextNode, store=store))
graph.add_node("answer", AnswerNode)

# fan-out
graph.add_edge(START, "memory_manager")
graph.add_edge(START, "stm_context")
graph.add_edge(START, "ltm_context")

# fan-in
graph.add_edge("stm_context", "answer")
graph.add_edge("ltm_context", "answer")

# 🔥 THIS WAS MISSING
graph.add_edge("memory_manager", END)

graph.add_edge("answer", END)

agent_workflow = graph.compile(store=store)


# %%
agent_workflow

# %%
initial_stm_state = {
    "messages": [],
    "stm_raw": [],
    "stm_summary": "",
    "last_summarized_idx": 0
}

    

# %%
agent_state = {
    "user_id": "user_1",
    "messages": [],          # full conversation
    "query": "",

    "stm": initial_stm_state,
    "ltm": {},

    # optional (will be filled later)
    "stm_context": [],
    "ltm_memories": [],
    "final_prompt_messages": [],
    "assistant_answer": ""
}


# %%
user_input = "How should I improve my machine learning skills?"

agent_state["messages"].append({
    "role": "user",
    "content": user_input
})
agent_state["query"] = user_input


# %%
result = agent_workflow.invoke(agent_state)


# %%
answer_graph = StateGraph(AgentState)

answer_graph.add_node("stm_context", STMContextNode)
answer_graph.add_node("ltm_context", partial(LTMContextNode, store=store))
answer_graph.add_node("answer", AnswerNode)

answer_graph.add_edge(START, "stm_context")
answer_graph.add_edge(START, "ltm_context")

answer_graph.add_edge("stm_context", "answer")
answer_graph.add_edge("ltm_context", "answer")

answer_graph.add_edge("answer", END)

answer_workflow = answer_graph.compile(store=store)


# %%
memory_graph = StateGraph(AgentState)

memory_graph.add_node("memory_manager", memory_manager)
memory_graph.add_edge(START, "memory_manager")
memory_graph.add_edge("memory_manager", END)

memory_workflow = memory_graph.compile(store=store)


# %%
import threading

def run_agent(agent_state: AgentState):
    # background memory write
    threading.Thread(
        target=memory_workflow.invoke,
        args=(agent_state,),
        daemon=True
    ).start()

    # fast answer path
    return answer_workflow.invoke(agent_state)


# %%
result = run_agent(agent_state)
print(result["assistant_answer"])


# %%
print("ANSWER:\n", result["assistant_answer"])


# %%
agent_state = {
    "user_id": "user_1",
    "messages": [
        {"role": "user", "content": "I want to improve my ML skills"}
    ],
    "query": "How should I improve my machine learning skills?",
    "stm": initial_stm_state,
    "ltm": {}
}

result = agent_workflow.invoke(agent_state)



