"""The real A2A exchange: the preferences agent's A2A server and the master's A2A client, connected in memory."""

import asyncio
from decimal import Decimal
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from backend.agents.user_preferences_agent.server import create_app
from backend.core.adapters.a2a_preferences import A2APreferencesAgent
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.models.shopping import ShoppingRequest
from tests.fakes import FakeChatModel

URL = "http://preferences.test/"
REQUEST = ShoppingRequest(query="running shoes", spending_limit=Decimal("120"), currency="EUR")
ANSWER = {"category": "shoes", "summary": "Size 43.", "attributes": {"size": "43"}}


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies), seen_inputs=[])


def preferences_server(model: FakeChatModel) -> httpx.AsyncClient:
    app = create_app(model, InMemoryPreferencesRepository(), InMemoryRecommendations(), URL)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL)


def test_agent_card() -> None:
    async def get_card() -> dict[str, object]:
        response = await preferences_server(script()).get("/.well-known/agent-card.json")
        card: dict[str, object] = response.json()
        return card

    card = asyncio.run(get_card())
    assert card["name"] == "User preferences"
    assert card["skills"][0]["id"] == "get_preferences"  # type: ignore[index]


def test_a_question_and_its_answer_travel_over_a2a() -> None:
    model = script(
        ("get_preferences", {"category": "shoes"}),
        ("ask_user", {"question": "What size?"}),
        ("done", {"category": "shoes"}),
        ("search_history", {"query": "running shoes"}),
        ("submit_preferences", ANSWER),
    )
    agent = A2APreferencesAgent(URL, http_client=preferences_server(model))
    asked: list[dict[str, Any]] = []

    def ask_user(question: dict[str, Any]) -> str:
        asked.append(question)
        return "43"

    preferences = agent.get_preferences(REQUEST, "conversation-1", ask_user)

    assert asked == [{"type": "preferences", "question": "What size?"}]
    assert (preferences.category, preferences.attributes) == ("shoes", {"size": "43"})
    calls = len(model.seen_inputs)
    # The master sends the same steps again after a pause: answered from memory, the agent does not run again.
    assert agent.get_preferences(REQUEST, "conversation-1", ask_user) == preferences
    assert len(model.seen_inputs) == calls
