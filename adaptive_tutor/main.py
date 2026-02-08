from agents.router import QueryRouter, router_node
from utils.llm_client import get_llm

llm = get_llm()
router = QueryRouter(llm)
