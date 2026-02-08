from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import BaseState
from langchain_community.tools import DuckDuckGoSearchRun
from adaptive_tutor.models import LLM

search_tool=DuckDuckGoSearchRun()

#agent - 2 for doubts handling
def doubt_handling(state:BaseState):
    llm=LLM

    SYSTEM_PROMPT = f"""
You are an AI Doubt-Handling Tutor Agent.
The current paper is: {state.get('paper_name')}


Your role:
- Answer the user's doubt clearly and correctly.
- FIRST try to solve using your own knowledge and reasoning.
- Use the web search tool ONLY if:
  - the doubt depends on recent info, OR
  - the concept is unclear / missing context, OR
  - factual verification is needed.

Important rules:
1. Keep explanations concise and clear.
2. Match the user's depth:
   - If doubt is shallow → give a short intuitive explanation.
   - If doubt seems deep (math, derivation, internals):
        - Briefly explain at high level
        - Then ASK: "Do you want to go deeper (math / internals)?"
3. Do NOT hallucinate facts.
4. Do NOT jump into long RAG-style explanations automatically.
5. If the question is ambiguous, ask a clarifying question.
6. Prefer examples over definitions when possible.

Tone:
- Friendly, teacher-like
- Simple language (Hinglish-friendly if user uses it)

You have access to a web search tool if needed.
Learner history (for personalization only) try to give answers peronaly by taking his or her name
also you can take data from histoy to give personal answer:
{state.get("stm_context")}
"""

    agent=create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=SYSTEM_PROMPT
    )

    result=agent.invoke({
        "messages":state["messages"]
    })

    return {
        "messages":result["messages"]
    }