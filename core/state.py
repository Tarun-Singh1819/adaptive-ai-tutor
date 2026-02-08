from typing import TypedDict, List, Optional, Annotated, Dict, Any,Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field

class MemoryDecision(BaseModel):
    should_write: bool=Field(description="Wheter to stay this in memory or not")
    memories:List[str]=Field(default_factory=list,description="Atomic user memories to store")

class Message(TypedDict):
    role: str
    content: str


class BaseState(TypedDict, total=False):
    # ========== Core Conversation ==========
    messages: Annotated[List[BaseMessage], add_messages]  # main chat messages
    stm_messages:List[Message]
    # ========== User Input ==========
    user_query: str
    paper_name: Optional[str]
    
    # ========== Routing ==========
    routed_to: Literal["overview", "doubt", "research"]  # in state.py
    agent_history: List[str]
    
    # ========== RAG ==========
    retrieved_docs: List[str]
    active_papers: List[str]  # max 3 paper ids
    current_papers: List[str]  # paper ids for current answer
    
    # ========== Short-Term Memory ==========
    stm_raw: List[Message]  # raw conversation messages
    stm_summary: str  # summary of recent conversation
    last_summarized_idx: int  # index of last summarized message
    last_ingested_idx: int
    stm_context: List[Message]  # context window for current interaction
    summary_updated: bool
    summary_upto_idx: int
    
    # ========== Long-Term Memory ==========
    user_id: str  # user identifier
    
   

class STMState(TypedDict, total=False):
    """Isolated state for STM graph - temp fields don't pollute BaseState"""
    user_id: str
    messages: List[BaseMessage]
    stm_messages: List[Message]
    stm_raw: List[Message]
    stm_summary: str
    last_ingested_idx: int
    last_summarized_idx: int
    stm_context: List[Message]
    summary_updated: bool
    summary_upto_idx: int


class LTMState(TypedDict, total=False):
    """Isolated state for LTM graph - temp fields don't pollute BaseState"""
    user_id: str
    user_query: str
    extracted_memories: Optional[List[str]]
    current_memory: Optional[str]
    duplicate_memory: Optional[Dict[str, Any]]
    current_embedding: Optional[List[float]]
    final_memory: Optional[str]
    importance: Optional[float]