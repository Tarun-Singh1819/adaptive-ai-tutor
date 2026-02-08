from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.state import BaseState
from langchain_community.tools import DuckDuckGoSearchRun
from adaptive_tutor.models import LLM
search_tool = DuckDuckGoSearchRun()


#agent -1 for overview generation
def OverView(state: BaseState):

    llm = LLM

    SYSTEM_PROMPT = f"""
You are a Research Paper Understanding Assistant.
The current paper is: {state.get('paper_name')}
Your task:
- Read the provided research paper content or title.
- Generate a clear, structured, beginner-friendly overview.

If the provided content is insufficient or unclear,
you MAY use DuckDuckGoSearch to gather reliable information
about the paper (title, authors, blogs, conference pages).

Rules:
- Do NOT hallucinate unknown facts.
- Clearly separate:
  • facts stated in the paper
  • inferred explanations
Tone:
- Friendly, teacher-like
- Simple language (Hinglish-friendly if user uses it)
  Learner history (for personalization only) try to give answers peronaly by taking his or her name
also you can take data from histoy to give personal answer:
{state.get("stm_context")}
"""

    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=SYSTEM_PROMPT
    )

    result = agent.invoke({
        "messages": [HumanMessage(content=state["user_query"])]
    })

    return {
        "messages": result["messages"]
    }