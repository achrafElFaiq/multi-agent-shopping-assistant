"""The shopping agent: the four chapters wired together.

memory -> search -> approval (pauses for the user) -> payment
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.shopping.nodes.approval import ask_approval
from backend.agents.shopping.nodes.memory import load_preferences
from backend.agents.shopping.nodes.payment import pay
from backend.agents.shopping.nodes.search import search_products
from backend.agents.shopping.schemas import (
    ApprovalDecision,
    PaymentResult,
    ProductOffer,
    Recommendation,
    ShoppingRequest,
    UserPreferences,
)
from backend.agents.shopping.state import ShoppingState

# Our types the checkpointer is allowed to load back from a saved state.
CHECKPOINT_TYPES = [
    ShoppingRequest,
    UserPreferences,
    ProductOffer,
    Recommendation,
    ApprovalDecision,
    PaymentResult,
]

type ShoppingGraph = CompiledStateGraph[ShoppingState, None, ShoppingState, ShoppingState]


def build_graph() -> ShoppingGraph:
    graph = StateGraph(ShoppingState)

    graph.add_node("memory", load_preferences)
    graph.add_node("search", search_products)
    graph.add_node("approval", ask_approval)
    graph.add_node("payment", pay)

    graph.add_edge(START, "memory")
    graph.add_edge("memory", "search")
    graph.add_edge("search", "approval")
    graph.add_edge("approval", "payment")
    graph.add_edge("payment", END)

    # The checkpointer saves the state, so the graph can pause at approval and resume later.
    checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    return graph.compile(checkpointer=checkpointer)
