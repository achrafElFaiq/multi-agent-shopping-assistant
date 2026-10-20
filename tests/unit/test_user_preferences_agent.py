"""The user preferences agent with a scripted model and in-memory stores: no network, no database."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from backend.agents.user_preferences_agent.agent import make_user_preferences_agent
from backend.agents.user_preferences_agent.server import pausable
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.models.recommendations import RecommendationRecord
from backend.core.models.shopping import ShoppingRequest
from backend.core.ports.preferences import Preferences
from tests.fakes import FakeChatModel, make_offer

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

REQUEST = ShoppingRequest(query="running shoes", spending_limit=Decimal("120"), currency="EUR")
ANSWER = {"category": "shoes", "summary": "Size 43, no suede.", "attributes": {"size": "43"}, "dislikes": ["suede"]}


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies), seen_inputs=[])


def test_uses_history_for_the_request_but_saves_nothing_from_it() -> None:
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
        ("done", {"category": "shoes"}),
        ("search_history", {"query": "running shoes"}),
        ("save_preferences", {"category": "shoes", "changes": {"dislikes": ["suede"]}}),
        ("submit_preferences", ANSWER),
    )

    answer = make_user_preferences_agent(model, preferences, recommendations)(REQUEST)

    assert answer.category == "shoes" and answer.dislikes == ["suede"]  # used for this request
    stored = preferences.get_preferences("shoes")
    assert stored is not None and stored.preferences.dislikes == []  # but not saved
    history = next(m for m in model.seen_inputs[-1] if isinstance(m, ToolMessage) and m.name == "search_history")
    assert "I never wear suede" in str(history.content)  # the agent saw why the user said no before
    tried = next(m for m in model.seen_inputs[-1] if isinstance(m, ToolMessage) and m.name == "save_preferences")
    assert tried.status == "error"  # the preparing step has no such tool


def test_saves_in_the_first_step_and_reads_past_offers_only_in_the_second() -> None:
    preferences = InMemoryPreferencesRepository({"shoes": Preferences(dislikes=["narrow fit"])})
    model = script(
        ("get_preferences", {"category": "shoes"}),
        ("save_preferences", {"category": "shoes", "changes": {"attributes": {"gender": "men's"}}}),
        ("done", {"category": "shoes"}),
        ("search_history", {"query": "trail running shoes"}),
        ("submit_preferences", ANSWER),
    )

    answer = make_user_preferences_agent(model, preferences, InMemoryRecommendations())(REQUEST)

    assert answer.category == "shoes"
    stored = preferences.get_preferences("shoes")
    assert stored is not None and stored.preferences.attributes == {"gender": "men's"}
    assert stored.preferences.dislikes == ["narrow fit"]
    saving, preparing = model.offered[:3], model.offered[3:]
    assert all(not {"search_history", "submit_preferences"} & set(names) for names in saving)
    assert all(not {"save_preferences", "remove_preferences", "ask_user"} & set(names) for names in preparing)
    assert model.tool_choices[3] == "search_history"  # what the preparing step starts with


def test_the_preparing_step_sees_what_the_saving_step_saved() -> None:
    preferences = InMemoryPreferencesRepository({"general": Preferences(), "shoes": Preferences()})
    model = script(
        ("get_preferences", {"category": "shoes"}),
        ("save_preferences", {"category": "general", "changes": {"dislikes": ["leather"]}}),
        ("save_preferences", {"category": "trail shoes", "changes": {"likes": ["Salomon"]}}),
        ("done", {"category": "shoes"}),
        ("search_history", {"query": "running shoes"}),
        ("submit_preferences", ANSWER),
    )

    make_user_preferences_agent(model, preferences, InMemoryRecommendations())(REQUEST)

    opening = next(inputs for inputs in model.seen_inputs if str(inputs[0].content).startswith("Prepare preferences"))
    context = json.loads(str(opening[1].content))
    assert context["general"]["dislikes"] == ["leather"]
    assert "trail shoes" in context["categories"]


def test_the_preparing_step_gets_the_questions_and_answers() -> None:
    model = script(
        ("get_preferences", {"category": "coffee"}),
        ("ask_user", {"question": "Which roast do you like?"}),
        ("done", {"category": "coffee"}),
        ("search_history", {"query": "coffee beans"}),
        ("submit_preferences", {"category": "coffee", "summary": "Light roast this time."}),
    )
    graph = pausable(make_user_preferences_agent(model, InMemoryPreferencesRepository(), InMemoryRecommendations()))
    config: RunnableConfig = {"configurable": {"thread_id": "coffee"}}

    graph.invoke({"request": REQUEST.model_copy(update={"query": "coffee beans"})}, config)
    graph.invoke(Command(resume="light roast, just this once"), config)

    opening = next(inputs for inputs in model.seen_inputs if str(inputs[0].content).startswith("Prepare preferences"))
    context = json.loads(str(opening[1].content))
    assert context["category"] == "coffee"
    assert context["answers"] == [{"question": "Which roast do you like?", "answer": "light roast, just this once"}]
    assert graph.get_state(config).values["preferences"].summary == "Light roast this time."


def test_stops_when_nothing_is_submitted() -> None:
    model = script(*[("get_preferences", {"category": "shoes"})] * 20)
    with pytest.raises(GraphRecursionError):
        make_user_preferences_agent(model, InMemoryPreferencesRepository(), InMemoryRecommendations())(REQUEST)
