from typing import Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.router_base import BaseRouter


class QueryRouter(BaseRouter):

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
You are an expert Intent Router for an Adaptive AI Tutor.
Your goal is to categorize the User Query into exactly ONE category based on the intent.

### AGENT CAPABILITIES:
1. **overview**: 
   - Use this if the user wants a summary, a high-level explanation, or "what is this paper about?".
   - Keywords: "explain", "summary", "overview", "tl;dr", "intro".

2. **doubt**: 
   - Use this for specific, localized questions about a term, a formula, or a specific sentence and very simple query.
   - Use this if the user is stuck on a concept and needs a simple "teacher-like" explanation.
   - Keywords: "what does X mean?", "explain this line", "confused about X".

3. **research**: 
   - Use this for deep, cross-sectional questions that require searching through the whole paper.
   - Use this for methodology, results, comparisons, or data-heavy queries.
   - Keywords: "how was X measured?", "results of experiment Y", "what methodology", "compare A and B".

### CONTEXT:
Chat History: {history}
User Query: {query}

### INSTRUCTIONS:
- Analyze the User Query against the Chat History to resolve pronouns (like "it", "this", "they").
- If the query is broad, pick 'overview'.
- If it's a "how" or "why" about the paper's core, pick 'research'.
- If it's a "what is" about a specific term, pick 'doubt'.

Return ONLY the lowercase word: overview, doubt, or research.
""",
            input_variables=["query", "history"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, state: Dict) -> str:
        # ✅ READ only
        query = state.get("user_query") 

        stm = state.get("stm_context", [])
        history = "\n".join(
   f"{m['role']}: {m['content']}"
    for m in stm
)


        decision = self.chain.invoke(
            {"query": query, "history": history}
        ).strip().lower()

        if decision not in {"overview", "doubt", "research"}:
            return "research"

        return decision


