"""Shopping flow with real PostgreSQL storage and scripted model responses."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langgraph.types import Command

from backend.agents.shopping.graph import ShoppingGraph
from backend.agents.shopping.graph import build_graph as build_real_graph
from backend.agents.shopping.schemas import ShoppingRequest
from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository
from backend.core.ports.preferences import Preferences
from tests.fakes import FakeChatModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def build_graph(repository: PostgresPreferencesRepository) -> ShoppingGraph:
    repository.update_preferences("shoes", Preferences(likes=["Asics"]))
    model = FakeChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[ToolCall(name="get_preferences", args={"category": "shoes"}, id="read")]),
                AIMessage(
                    "",
                    tool_calls=[
                        ToolCall(
                            name="submit_preferences",
                            args={"category": "shoes", "summary": "Likes Asics.", "likes": ["Asics"]},
                            id="submit",
                        )
                    ],
                ),
            ]
        ),
    )
    return build_real_graph(model, repository)


def make_request(spending_limit: str = "200") -> ShoppingRequest:
    return ShoppingRequest(
        query="running shoes",
        spending_limit=Decimal(spending_limit),
        currency="EUR",
        deliver_by=date(2026, 10, 20),
    )


@pytest.mark.parametrize(
    ("action", "spending_limit", "expected_status"),
    [("approve", "200", "paid"), ("cancel", "200", "cancelled"), ("approve", "50", "blocked")],
)
def test_pauses_before_payment_and_respects_decision_and_limit(
    repository: PostgresPreferencesRepository, action: str, spending_limit: str, expected_status: str
) -> None:
    graph = build_graph(repository)
    config: RunnableConfig = {"configurable": {"thread_id": "approval"}}

    graph.invoke({"request": make_request(spending_limit)}, config)

    paused = graph.get_state(config)
    assert paused.next == ("approval",)
    assert "payment" not in paused.values
    assert paused.values["preferences"].likes == ["Asics"]
    assert paused.interrupts[0].value == paused.values["recommendation"].model_dump(mode="json")

    result = graph.invoke(Command(resume={"action": action}), config)

    assert result["decision"].action == action
    assert result["payment"].status == expected_status
    assert graph.get_state(config).next == ()


def test_invalid_answer_asks_again(repository: PostgresPreferencesRepository) -> None:
    graph = build_graph(repository)
    config: RunnableConfig = {"configurable": {"thread_id": "invalid"}}
    graph.invoke({"request": make_request()}, config)

    graph.invoke(Command(resume={"action": "yes"}), config)

    paused = graph.get_state(config)
    assert "payment" not in paused.values
    [question] = paused.interrupts
    assert question.value["error"] == "Please answer approve or cancel."
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["payment"].status == "paid"


def test_new_category_asks_saves_answer_and_continues_to_approval(
    repository: PostgresPreferencesRepository,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_preferences", {"category": "coffee"}),
        ("ask_user", {"question": "Quelle torréfaction préfères-tu ?"}),
        ("update_preferences", {"category": "coffee", "changes": {"likes": ["light roast"]}}),
        ("submit_preferences", {"category": "coffee", "summary": "Prefers light roast.", "likes": ["light roast"]}),
    ]
    model = FakeChatModel(
        messages=iter(AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=name)]) for name, args in calls)
    )
    graph = build_real_graph(model, repository)
    config: RunnableConfig = {"configurable": {"thread_id": "new-category"}}
    request = make_request().model_copy(update={"query": "Je cherche du café de spécialité"})

    graph.invoke({"request": request}, config)

    paused = graph.get_state(config)
    assert paused.next == ("user_preferences",)
    assert paused.interrupts[0].value == {"type": "preferences", "question": "Quelle torréfaction préfères-tu ?"}
    assert model.tool_choices == ["get_preferences", "ask_user"]
    assert repository.get_preferences("coffee") is None

    result = graph.invoke(Command(resume="Je préfère une torréfaction légère"), config)

    assert model.tool_choices == ["get_preferences", "ask_user", "required", "required"]
    assert model.seen_inputs[2][-1].content == "Je préfère une torréfaction légère"
    stored = repository.get_preferences("coffee")
    assert stored is not None
    assert stored.preferences.likes == ["light roast"]
    assert result["preferences"].likes == ["light roast"]
    assert "recommendation" in result
    assert "payment" not in result
    assert graph.get_state(config).next == ("approval",)

    result = graph.invoke(Command(resume={"action": "cancel"}), config)
    assert result["payment"].status == "cancelled"
    assert graph.get_state(config).next == ()
