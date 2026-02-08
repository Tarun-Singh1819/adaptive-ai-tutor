from langchain_core.messages import HumanMessage
from adaptive_tutor.v_build_graph import build_graph
from db.sqlite_db import init_db
from utils.debug_store import inspect_store, inspect_state

def main():
    init_db()

    # build graph ONCE
    app = build_graph()

    # persistent session state
    state = {
        "messages": [],
        "user_id": "default_user",  # Added user_id for memory
        "active_papers": [
            "AI Agents: Evolution, Architecture, and Real-World Applications"
        ],
        "paper_name": "AI Agents: Evolution, Architecture, and Real-World Applications",
    }

    print("🤖 Adaptive Tutor ready. Type 'exit' to quit.")
    print("   Type 'debug' to inspect store and state.\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break
        
        # Debug command
        if user_input.lower() == "debug":
            inspect_store(state.get("user_id", "default_user"))
            inspect_state(state)
            continue

        state["messages"].append(
            HumanMessage(content=user_input)
        )
        state["user_query"] = user_input
        
        # invoke graph
        state = app.invoke(state)

        print("Tutor:", state["messages"][-1].content)
        print("-" * 50)

if __name__ == "__main__":
    main()
