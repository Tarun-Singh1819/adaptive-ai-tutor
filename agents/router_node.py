from typing import Dict
from agents.router import QueryRouter

# router object yahan inject hota hai
def make_router_node(router: QueryRouter):
    def router_node(state: dict) -> dict:
        decision = router.route(state)

        state["routed_to"] = decision
        state["agent_history"] = state.get("agent_history", []) + [decision]

        return state
    return router_node

