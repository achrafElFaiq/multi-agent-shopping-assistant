"""The master agent with a scripted model, a fake preferences agent, a fake product finder and the real pauses:
no network, no API key."""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from backend.agents.master_agent.agent import MasterGraph, build_master, first_messages
from backend.agents.master_agent.tools import payment_of
from backend.agents.user_preferences_agent.server import create_app
from backend.core.adapters.a2a_preferences import A2APreferencesAgent
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.models.preferences import UserPreferences
from backend.core.models.shopping import ShoppingRequest
from tests.fakes import FakeChatModel, FakeSearcher, make_offer

CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}
PREFERENCES_URL = "http://preferences.test/"
REQUEST = ShoppingRequest(query="running shoes", spending_limit=Decimal("120"), currency="EUR")
FLASHY, PLAIN = make_offer("Lovell", "110"), make_offer("Pro:Direct", "95")


class FakePreferencesAgent:
    def __init__(self) -> None:
        self.requests: list[ShoppingRequest] = []

    def get_preferences(
        self, request: ShoppingRequest, conversation_id: str, ask_user: Callable[[dict[str, Any]], Any]
    ) -> UserPreferences:
        self.requests.append(request)
        return UserPreferences(category="shoes", summary="Size 43", attributes={"country": "France"})


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies))


def question(master: MasterGraph) -> dict[str, Any]:
    value: dict[str, Any] = master.get_state(CONFIG).interrupts[0].value
    return value


def test_preferences_then_no_then_find_better_then_yes_then_pay() -> None:
    model = script(
        ("ask_preferences", {}),
        ("find_products", {"brief": "Running shoes, size 43"}),
        ("present_offer", {"offer_number": 1, "pitch": "Light and fast."}),
        ("find_products", {"brief": "Find better: the user refused offer 1 because: too flashy"}),
        ("present_offer", {"offer_number": 2, "pitch": "Plain and black, as you like."}),
        ("pay", {"offer_number": 2}),
    )
    preferences, searcher, saved = FakePreferencesAgent(), FakeSearcher([FLASHY], [PLAIN]), InMemoryRecommendations()
    master = build_master(model, preferences, searcher, saved, REQUEST)
    master.invoke(first_messages(REQUEST), CONFIG)

    shown = question(master)
    assert shown["offer"]["store"] == "Lovell" and shown["offer"]["price"] == "110"  # from the offer, not the model
    master.invoke(Command(resume="no"), CONFIG)
    assert question(master)["type"] == "reason"
    master.invoke(Command(resume="too flashy"), CONFIG)
    assert question(master)["offer"]["store"] == "Pro:Direct"
    result = master.invoke(Command(resume="yes"), CONFIG)

    payment = payment_of(result["messages"])
    assert payment is not None and payment.status == "paid" and payment.amount == Decimal("95")
    # The product finder got the reason, the limits from code, and what was already shown.
    second = searcher.requests[1]
    assert "too flashy" in second.brief
    assert (second.country, second.currency, second.max_price, second.seen) == (
        "FR",
        "EUR",
        Decimal("120"),
        [FLASHY.product_id],
    )
    assert [(r.offer.seller.name, r.accepted, r.reason) for r in saved.records] == [
        ("Lovell", False, "too flashy"),
        ("Pro:Direct", True, ""),
    ]
    assert len(preferences.requests) == 1 and len(searcher.requests) == 2  # the pauses did not call them again


def test_no_payment_without_a_yes() -> None:
    model = script(
        ("ask_preferences", {}),
        ("find_products", {"brief": "Running shoes"}),
        ("pay", {"offer_number": 1}),
        ("give_up", {"reason": "nothing fits"}),
    )
    master = build_master(model, FakePreferencesAgent(), FakeSearcher([PLAIN]), InMemoryRecommendations(), REQUEST)
    result = master.invoke(first_messages(REQUEST), CONFIG)

    assert payment_of(result["messages"]) is None
    refused = [m for m in result["messages"] if getattr(m, "name", None) == "pay"][0]
    assert "payment refused" in refused.content


def test_a_question_from_the_preferences_agent_over_a2a_pauses_the_master() -> None:
    preferences_model = script(
        ("get_preferences", {"category": "shoes"}),
        ("search_history", {"query": "running shoes"}),
        ("ask_user", {"question": "What size?"}),
        ("submit_preferences", {"category": "shoes", "summary": "Size 43", "attributes": {"size": "43"}}),
    )
    app = create_app(preferences_model, InMemoryPreferencesRepository(), InMemoryRecommendations(), PREFERENCES_URL)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PREFERENCES_URL)
    preferences = A2APreferencesAgent(PREFERENCES_URL, http_client=http)
    model = script(("ask_preferences", {}), ("give_up", {"reason": "test"}))
    master = build_master(model, preferences, FakeSearcher(), InMemoryRecommendations(), REQUEST)
    master.invoke(first_messages(REQUEST), CONFIG)

    assert question(master) == {"type": "preferences", "question": "What size?"}
    result = master.invoke(Command(resume="43"), CONFIG)

    got = [m for m in result["messages"] if getattr(m, "name", None) == "ask_preferences"][0]
    assert got.artifact["attributes"] == {"size": "43"}  # the answer reached the preferences agent
