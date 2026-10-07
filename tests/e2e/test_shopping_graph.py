"""Shopping flow with real PostgreSQL storage and scripted model responses."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

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
