"""The shopping agent: the four chapters wired together.

user_preferences -> product_search -> approval (pauses for the user) -> payment
"""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.shopping.nodes.approval import ask_approval
from backend.agents.shopping.nodes.payment import pay
from backend.agents.shopping.nodes.product_search import search_products
from backend.agents.shopping.nodes.user_preferences import make_user_preferences_node
from backend.agents.shopping.state import ShoppingState
from backend.core.ports.preferences import PreferencesRepository

type ShoppingGraph = CompiledStateGraph[ShoppingState, None, ShoppingState, ShoppingState]


def build_graph(model: BaseChatModel, repository: PreferencesRepository) -> ShoppingGraph:
    """Build the agent. The caller chooses the model and the data source (real ones in the app, fakes in tests)."""
    graph = StateGraph(ShoppingState, input_schema=ShoppingState, output_schema=ShoppingState)

    graph.add_node("user_preferences", make_user_preferences_node(model, repository))
    graph.add_node("product_search", search_products)
    graph.add_node("approval", ask_approval)
    graph.add_node("payment", pay)

    graph.add_edge(START, "user_preferences")
    graph.add_edge("user_preferences", "product_search")
    graph.add_edge("product_search", "approval")
    graph.add_edge("approval", "payment")
    graph.add_edge("payment", END)

    # The checkpointer saves the state, so the graph can pause at approval and resume later.
    return graph.compile(checkpointer=InMemorySaver())
