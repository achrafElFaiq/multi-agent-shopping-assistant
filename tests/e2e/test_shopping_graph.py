from decimal import Decimal
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langgraph.types import Command

from backend.agents.shopping.graph import ShoppingGraph
from backend.agents.shopping.graph import build_graph as build_real_graph
from backend.agents.shopping.schemas import ShoppingRequest, UserPreferences
from backend.core.adapters.mock_preferences import MockPreferencesRepository
from tests.fakes import FakeChatModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def build_graph() -> ShoppingGraph:
    """The real graph, with a fake model that answers without calling tools."""
    model = FakeChatModel(
        messages=iter([AIMessage("I have what I need.")]),
        structured_answer=UserPreferences(user_id="user-1", category="shoes", summary="Likes Asics."),
    )
    return build_real_graph(model, MockPreferencesRepository())


def make_request(spending_limit: str = "200") -> ShoppingRequest:
    return ShoppingRequest(user_id="user-1", query="running shoes", spending_limit=Decimal(spending_limit))


def test_graph_compiles() -> None:
    graph = build_graph()

    nodes = graph.get_graph().nodes
    for name in ["user_preferences", "product_search", "approval", "payment"]:
        assert name in nodes


def test_graph_pauses_at_approval() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "pause"}}

    graph.invoke({"request": make_request()}, config)

    assert graph.get_state(config).next == ("approval",)


def test_approve_pays() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "approve"}}

    graph.invoke({"request": make_request()}, config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)

    assert result["payment"].status == "paid"


def test_cancel_does_not_pay() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "cancel"}}

    graph.invoke({"request": make_request()}, config)
    result = graph.invoke(Command(resume={"action": "cancel"}), config)

    assert result["payment"].status == "cancelled"


def test_over_limit_is_blocked() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "limit"}}

    graph.invoke({"request": make_request(spending_limit="50")}, config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)

    assert result["payment"].status == "blocked"


def test_invalid_answer_asks_again() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "invalid"}}
    graph.invoke({"request": make_request()}, config)

    graph.invoke(Command(resume={"action": "yes"}), config)

    [question] = graph.get_state(config).interrupts  # paused again: still waiting for an answer
    assert question.value["error"] == "Please answer approve or cancel."
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["payment"].status == "paid"
