"""The shopping agent: the four chapters wired together.

user_preferences -> recommendation (works with the search agent over A2A, pauses for the user) -> payment
"""

from typing import get_type_hints

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.shopping.nodes.payment import pay
from backend.agents.shopping.nodes.recommendation import make_recommendation_node
from backend.agents.shopping.nodes.user_preferences import make_user_preferences_node
from backend.agents.shopping.state import ShoppingState
from backend.core.ports.preferences import PreferencesRepository
from backend.core.ports.recommendations import RecommendationRepository
from backend.core.ports.searcher import ProductSearcher

type ShoppingGraph = CompiledStateGraph[ShoppingState, None, ShoppingState, ShoppingState]


def build_graph(
    model: BaseChatModel,
    repository: PreferencesRepository,
    searcher: ProductSearcher,
    recommendations: RecommendationRepository,
) -> ShoppingGraph:
    """Build the agent. The caller chooses the model, the preferences store, the search agent and where answers
    to recommendations are kept (real ones in the app, fakes in tests)."""
    graph = StateGraph(ShoppingState, input_schema=ShoppingState, output_schema=ShoppingState)

    graph.add_node("user_preferences", make_user_preferences_node(model, repository))
    graph.add_node("recommendation", make_recommendation_node(model, searcher, recommendations))
    graph.add_node("payment", pay)

    graph.add_edge(START, "user_preferences")
    graph.add_edge("user_preferences", "recommendation")
    graph.add_edge("recommendation", "payment")
    graph.add_edge("payment", END)

    # Register the state models so checkpoints can restore them without warnings.
    return graph.compile(
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=list(get_type_hints(ShoppingState).values()))
        )
    )
