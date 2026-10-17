"""The user preferences agent with a scripted model and in-memory stores: no network, no database."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.errors import GraphRecursionError

from backend.agents.user_preferences_agent.agent import make_user_preferences_agent
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.models.recommendations import RecommendationRecord
from backend.core.models.shopping import ShoppingRequest
from backend.core.ports.preferences import Preferences
from tests.fakes import FakeChatModel, make_offer

REQUEST = ShoppingRequest(query="running shoes", spending_limit=Decimal("120"), currency="EUR")
ANSWER = {"category": "shoes", "summary": "Size 43, no suede.", "attributes": {"size": "43"}, "dislikes": ["suede"]}


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies), seen_inputs=[])


def test_reads_preferences_and_history_updates_and_submits() -> None:
    preferences = InMemoryPreferencesRepository({"shoes": Preferences(attributes={"size": "43"})})
    recommendations = InMemoryRecommendations()
    recommendations.save(
        RecommendationRecord(
            query="trail shoes",
            category="shoes",
            offer=make_offer("Lovell", "110"),
            pitch="Soft suede.",
            accepted=False,
            reason="I never wear suede",
            decided_at=datetime.now(UTC),
        )
    )
    model = script(
        ("get_preferences", {"category": "shoes"}),
        ("search_history", {"query": "running shoes"}),
        ("save_preferences", {"category": "shoes", "changes": {"dislikes": ["suede"]}}),
        ("submit_preferences", ANSWER),
    )

    answer = make_user_preferences_agent(model, preferences, recommendations)(REQUEST)

    assert answer.category == "shoes" and answer.dislikes == ["suede"]
    stored = preferences.get_preferences("shoes")
    assert stored is not None and stored.preferences.dislikes == ["suede"]
    assert stored.preferences.attributes == {"size": "43"}  # an update keeps what it does not change
    history = next(m for m in model.seen_inputs[-1] if isinstance(m, ToolMessage) and m.name == "search_history")
    assert "I never wear suede" in str(history.content)  # the agent saw why the user said no before


def test_stops_when_nothing_is_submitted() -> None:
    model = script(*[("get_preferences", {"category": "shoes"})] * 20)
    with pytest.raises(GraphRecursionError):
        make_user_preferences_agent(model, InMemoryPreferencesRepository(), InMemoryRecommendations())(REQUEST)
