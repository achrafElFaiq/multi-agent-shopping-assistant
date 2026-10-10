"""The recommendation agent working with the search agent, with a scripted model, a fake search agent and the
real pauses: no network, no API key."""

from datetime import date
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.agents.shopping.nodes.recommendation import make_recommendation_node
from backend.agents.shopping.schemas import ShoppingRequest, UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from tests.fakes import FakeChatModel, FakeSearcher, make_offer

CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}
FLASHY, PLAIN = make_offer("Lovell", "110"), make_offer("Pro:Direct", "95")


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies))


def start(model: FakeChatModel, searcher: FakeSearcher, saved: InMemoryRecommendations) -> Any:
    """A graph with only the recommendation node, paused at its first question."""
    graph = StateGraph(ShoppingState)
    graph.add_node("recommendation", make_recommendation_node(model, searcher, saved))
    graph.add_edge(START, "recommendation")
    graph.add_edge("recommendation", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    state: ShoppingState = {
        "request": ShoppingRequest(
            query="running shoes", spending_limit=Decimal("120"), currency="EUR", deliver_by=date(2026, 10, 20)
        ),
        "preferences": UserPreferences(category="shoes", summary="Size 43", attributes={"country": "France"}),
    }
    compiled.invoke(state, CONFIG)
    return compiled


def question(graph: Any) -> dict[str, Any]:
    value: dict[str, Any] = graph.get_state(CONFIG).interrupts[0].value
    return value


def test_no_then_why_then_find_better_then_yes() -> None:
    model = script(
        ("ask_searcher", {"brief": "Running shoes, size 43"}),
        ("present_offer", {"offer_number": 1, "pitch": "Light and fast."}),
        ("ask_searcher", {"brief": "Find better: the shopper refused offer 1 because: too flashy"}),
        ("present_offer", {"offer_number": 2, "pitch": "Plain and black, as you like."}),
    )
    searcher, saved = FakeSearcher([FLASHY], [PLAIN]), InMemoryRecommendations()
    graph = start(model, searcher, saved)

    shown = question(graph)
    assert shown["type"] == "recommendation" and shown["number"] == 1
    assert shown["offer"]["store"] == "Lovell" and shown["offer"]["price"] == "110"  # from the offer, not the model
    assert shown["pitch"] == "Light and fast."

    graph.invoke(Command(resume="no"), CONFIG)
    assert question(graph) == {"type": "reason", "question": "Why not this one?"}

    graph.invoke(Command(resume="too flashy"), CONFIG)
    assert question(graph)["offer"]["store"] == "Pro:Direct"
    # The search agent got the reason, the limits from code, and what was already shown.
    second = searcher.requests[1]
    assert "too flashy" in second.brief
    assert (second.country, second.currency, second.max_price) == ("FR", "EUR", Decimal("120"))
    assert second.seen == [FLASHY.product_id]

    result = graph.invoke(Command(resume="yes"), CONFIG)
    assert result["decision"].action == "approve"
    assert result["recommendation"].offer.store == "Pro:Direct"
    assert result["recommendation"].reason == "Plain and black, as you like."
    assert [(r.offer.seller.name, r.accepted, r.reason) for r in saved.records] == [
        ("Lovell", False, "too flashy"),
        ("Pro:Direct", True, ""),
    ]
    assert len(searcher.requests) == 2  # the pauses did not send the searches again


def test_an_invalid_answer_is_asked_again() -> None:
    model = script(("ask_searcher", {"brief": "Running shoes"}), ("present_offer", {"offer_number": 1, "pitch": "."}))
    graph = start(model, FakeSearcher([PLAIN]), InMemoryRecommendations())

    graph.invoke(Command(resume="maybe"), CONFIG)

    assert question(graph)["error"] == "Please answer yes or no."
    assert graph.invoke(Command(resume="yes"), CONFIG)["decision"].action == "approve"


def test_offers_from_the_search_agent_that_break_the_limits_are_never_shown() -> None:
    model = script(
        ("ask_searcher", {"brief": "Running shoes"}),
        ("present_offer", {"offer_number": 1, "pitch": "."}),
        ("no_more_offers", {}),
    )
    too_expensive = make_offer("Lovell", "300")
    saved = InMemoryRecommendations()

    graph = start(model, FakeSearcher([too_expensive]), saved)

    final = graph.get_state(CONFIG)
    assert not final.interrupts  # nothing was shown
    assert final.values["decision"].action == "cancel"
    assert "There is no offer 1" in str(model.seen_inputs[2][-1].content)
    assert saved.records == []


def test_nothing_accepted_means_no_payment() -> None:
    model = script(
        ("ask_searcher", {"brief": "Running shoes"}),
        ("present_offer", {"offer_number": 1, "pitch": "."}),
        ("no_more_offers", {}),
    )
    graph = start(model, FakeSearcher([PLAIN]), InMemoryRecommendations())

    graph.invoke(Command(resume="no"), CONFIG)
    result = graph.invoke(Command(resume=""), CONFIG)  # no reason given

    assert result["decision"].action == "cancel"
    assert "recommendation" not in result
