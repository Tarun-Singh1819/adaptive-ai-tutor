from db.sqlite_db import get_paper
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from collections import OrderedDict
from core.state import BaseState
from create_pipeline.initialize_rag import initialize_rag
from create_pipeline.normalize_name import normalize_name
from create_pipeline.load_and_save import load_docs,load_faiss_index,load_parents
from create_pipeline.build_bm25 import build_bm25
from create_pipeline.hybrid_retrieve import hybrid_retrieve
from create_pipeline.build_context import build_context
from create_pipeline.fetch_unique_parents import fetch_unique_parents
from create_pipeline.touch_cache import touch_cache
from adaptive_tutor.models import EMBED_MODEL,LLM
from core.cache import MAX_PAPERS_IN_CACHE,RAG_CACHE
from memory.ltm.retrieve_ltm import retrieve_ltm
from memory.ltm.utils import get_embeddings
from langgraph.store.memory import BaseStore


def format_stm_context(stm_context):
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in stm_context
    )


def format_ltm_context(ltm_memories):
    """Format LTM memories for prompt"""
    if not ltm_memories:
        return "No long-term memories available."
    return "\n".join(
        f"- {mem['data']} (importance: {mem['importance']:.2f})"
        for mem in ltm_memories
    )


def research_agent(state: BaseState, store: BaseStore):
    """
    Research Agent (CACHED):
    - SQLite decides if paper exists
    - Disk load only once per paper
    - RAM cache reused across queries
    - Uses LTM for user personalization
    """

    # 1️⃣ Extract query & paper name
    query = state.get("user_query", "")
    paper_name = state["paper_name"]
    paper_id = normalize_name(paper_name)
    user_id = state.get("user_id", "default_user")

    # 2️⃣ LLM (global singleton)
    llm = LLM

    
    # 4️⃣ ---- RAG CACHE LAYER ----
    if paper_id in RAG_CACHE:
        cached = RAG_CACHE[paper_id]

        index = cached["index"]
        docs = cached["docs"]
        parents = cached["parents"]
        bm25 = cached["bm25"]

    else:
        # 3️⃣ SQLite check (SINGLE SOURCE OF TRUTH)
        paper = get_paper(paper_id)

        if paper is None:
            store_path = initialize_rag(paper_name)
        else:
            store_path = Path(paper["rag_path"])
         

        # Cold load (disk)
        index = load_faiss_index(store_path)
        docs = load_docs(store_path)
        parents = load_parents(store_path)

        bm25, _ = build_bm25(docs=docs)

        # Save to cache
        RAG_CACHE[paper_id] = {
            "index": index,
            "docs": docs,
            "parents": parents,
            "bm25": bm25
        }
        touch_cache(paper_id)

    print("📦 Current Cache:", list(RAG_CACHE.keys()))

    # 5️⃣ Retrieval
    hybrid_hits = hybrid_retrieve(
        query=query,
        model=EMBED_MODEL,
        index=index,
        docs=docs,
        bm25=bm25
    )

    parent_map = {p.id: p for p in parents}

    selected_parents = fetch_unique_parents(
        child_hits=hybrid_hits,
        parent_map=parent_map
    )

    context = build_context(parents=selected_parents)
    stm_context = state.get("stm_context", [])
    stm_history_text = format_stm_context(stm_context)
    
    # 5.5️⃣ LTM Retrieval - Get user's long-term memories
    ltm_memories = retrieve_ltm(
        store=store,
        user_id=user_id,
        query=query,
        embed_fn=get_embeddings,  # Use wrapper function, not model directly
        top_k=3
    )
    ltm_context_text = format_ltm_context(ltm_memories)

    # 6️⃣ Prompt with STM + LTM context
    prompt = PromptTemplate(
        template="""
You are an AI research tutor.

You MUST answer the question using ONLY the provided research context.
If the context does not contain enough information,
clearly say that the paper does not cover this.

Explain in a clear and structured way:
- Start with intuition
- Then give technical explanation if required

Research Context:
{context}

Learner's Short-Term History (recent conversation):
{stm_history}

Learner's Long-Term Memory (user preferences & facts):
{ltm_context}

Question:
{question}
""",
        input_variables=["context", "question", "stm_history", "ltm_context"],
    )

    answer = (prompt | llm).invoke(
        {
            "context": context,
            "question": query,
            "stm_history": stm_history_text,
            "ltm_context": ltm_context_text
        }
    )

    # 7️⃣ Return updated state
    return {
        "messages": [
            AIMessage(content=answer.content)
        ]
    }
