"""Preferences tools and agent behavior against isolated PostgreSQL data."""

import json
from datetime import date
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from backend.agents.shopping.nodes.user_preferences import make_user_preferences_node
from backend.agents.shopping.schemas import ShoppingRequest, UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.agents.shopping.tools.user_preferences import make_preference_tools
from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository
from backend.core.ports.preferences import Budget, Preferences
from tests.fakes import FakeChatModel

ANSWER = UserPreferences(
    category="shoes",
    summary="Prefers black Asics shoes, size 43 EU; usual budget exceeds this purchase limit.",
    attributes={"size": "43 EU"},
    likes=["black", "Asics"],
    dislikes=["suede"],
    usual_budget=Budget(amount=Decimal("130"), currency="EUR"),
)


def make_state() -> ShoppingState:
    return {
        "request": ShoppingRequest(
            query="running shoes",
            spending_limit=Decimal("120"),
            currency="EUR",
            deliver_by=date(2026, 10, 20),
        )
    }


def tool_reply(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=call_id)])


def test_reads_context_and_submits_without_saving(repository: PostgresPreferencesRepository) -> None:
    repository.update_preferences("general", Preferences(likes=["black"], dislikes=["suede"]))
    stored = repository.update_preferences(
        "shoes", Preferences(attributes={"size": "43 EU"}, likes=["Asics"], usual_budget=ANSWER.usual_budget)
    )
    model = FakeChatModel(
        messages=iter(
            [
                tool_reply("get_preferences", "read", category="shoes"),
                tool_reply("submit_preferences", "submit", **ANSWER.model_dump(mode="json")),
            ]
        )
    )

    result = make_user_preferences_node(model, repository)(make_state())

    context = json.loads(model.seen_inputs[0][-1].content)
    assert context["categories"] == ["general", "shoes"]
    assert context["general"]["likes"] == ["black"]
    assert context["request"]["query"] == "running shoes"
    assert context["request"]["spending_limit"] == "120"
    assert context["request"]["currency"] == "EUR"
    assert context["request"]["deliver_by"] == "2026-10-20"
    observation = model.seen_inputs[1][-1]
    assert isinstance(observation, ToolMessage)
    assert json.loads(str(observation.content))["preferences"]["attributes"] == {"size": "43 EU"}
    assert result["preferences"] == ANSWER
    assert len(model.seen_inputs) == 2
    assert repository.get_preferences("shoes") == stored


def test_partial_update_preserves_unrelated_preferences(repository: PostgresPreferencesRepository) -> None:
    before = repository.update_preferences(
        "shoes",
        Preferences(
            attributes={"size": "43 EU", "width": "wide"},
            likes=["Asics"],
            dislikes=["suede"],
            requirements=["wide toe box"],
            usual_budget=ANSWER.usual_budget,
        ),
    )
    tools = {tool.name: tool for tool in make_preference_tools(repository)}

    tools["update_preferences"].invoke(
        {
            "category": "shoes",
            "changes": {"attributes": {"size": "44 EU"}, "likes": []},
        }
    )

    saved = repository.get_preferences("shoes")
    assert saved is not None
    assert saved.preferences.attributes == {"size": "44 EU", "width": "wide"}
    assert saved.preferences.likes == []
    assert saved.preferences.dislikes == before.preferences.dislikes
    assert saved.preferences.requirements == before.preferences.requirements
    assert saved.preferences.usual_budget == before.preferences.usual_budget
    assert saved.updated_at >= before.updated_at


def test_invalid_submission_is_returned_to_model_for_correction(repository: PostgresPreferencesRepository) -> None:
    model = FakeChatModel(
        messages=iter(
            [
                tool_reply("get_preferences", "read", category="shoes"),
                tool_reply(
                    "submit_preferences",
                    "invalid",
                    category="shoes",
                    summary="Unknown preferences",
                    usual_budget={"amount": "100 EUR", "currency": "EUR"},
                ),
                tool_reply("submit_preferences", "corrected", category="shoes", summary="No stored preferences"),
            ]
        )
    )

    result = make_user_preferences_node(model, repository)(make_state())

    observation = model.seen_inputs[1][-1]
    assert isinstance(observation, ToolMessage)
    assert json.loads(str(observation.content))["preferences"] is None
    error = model.seen_inputs[2][-1]
    assert isinstance(error, ToolMessage)
    assert error.status == "error"
    assert "Invalid preferences" in str(error.content)
    assert result["preferences"] == UserPreferences(category="shoes", summary="No stored preferences")
    assert repository.list_categories() == []
