from adaptive_tutor.build_graph import build_graph
from langchain_core.messages import HumanMessage
from core.state import BaseState
from db.sqlite_db import init_db
def main():
    init_db()
    app = build_graph()

    state: BaseState = {
    "messages": [
        HumanMessage(
            content="What future directions dows the paper propose for agent research?"
        )
    ],
    "paper_name": "AI Agents: Evolution, Architecture, and Real-World Applications",
    "rag_registry": {}
}


    result = app.invoke(state)

    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
