"""Shopping flow with real PostgreSQL storage, scripted model responses and a fake search agent."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langgraph.types import Command

from backend.agents.shopping.graph import ShoppingGraph
from backend.agents.shopping.graph import build_graph as build_real_graph
from backend.agents.shopping.schemas import ShoppingRequest
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository
from backend.core.ports.preferences import Preferences
from tests.fakes import FakeChatModel, FakeSearcher, make_offer

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

PREFERENCES: list[tuple[str, dict[str, Any]]] = [
    ("get_preferences", {"category": "shoes"}),
    ("submit_preferences", {"category": "shoes", "summary": "Likes Asics.", "likes": ["Asics"]}),
]
RECOMMEND_FIRST: list[tuple[str, dict[str, Any]]] = [
    ("ask_searcher", {"brief": "Asics running shoes"}),
    ("present_offer", {"offer_number": 1, "pitch": "Asics, as you like."}),
]


def scripted(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies))


def build_graph(repository: PostgresPreferencesRepository, *calls: tuple[str, dict[str, Any]]) -> ShoppingGraph:
    repository.update_preferences("shoes", Preferences(likes=["Asics"]))
    searcher = FakeSearcher([make_offer("Pro:Direct", "99.90")])
    return build_real_graph(
        scripted(*PREFERENCES, *RECOMMEND_FIRST, *calls), repository, searcher, InMemoryRecommendations()
    )


def make_request(spending_limit: str = "200") -> ShoppingRequest:
    return ShoppingRequest(
        query="running shoes",
        spending_limit=Decimal(spending_limit),
        currency="EUR",
        deliver_by=date(2026, 10, 20),
    )


def test_pauses_before_payment_and_pays_when_the_shopper_says_yes(repository: PostgresPreferencesRepository) -> None:
    graph = build_graph(repository)
    config: RunnableConfig = {"configurable": {"thread_id": "yes"}}

    graph.invoke({"request": make_request()}, config)

    paused = graph.get_state(config)
    assert paused.next == ("recommendation",)
    assert "payment" not in paused.values
    assert paused.values["preferences"].likes == ["Asics"]
    shown = paused.interrupts[0].value
    assert (shown["type"], shown["offer"]["store"], shown["offer"]["price"]) == (
        "recommendation",
        "Pro:Direct",
        "99.90",
    )

    result = graph.invoke(Command(resume="yes"), config)

    assert result["decision"].action == "approve"
    assert result["payment"].status == "paid"
    assert graph.get_state(config).next == ()


def test_no_payment_when_the_shopper_says_no_to_everything(repository: PostgresPreferencesRepository) -> None:
    graph = build_graph(repository, ("no_more_offers", {}))
    config: RunnableConfig = {"configurable": {"thread_id": "no"}}
    graph.invoke({"request": make_request()}, config)

    graph.invoke(Command(resume="no"), config)
    assert graph.get_state(config).interrupts[0].value["type"] == "reason"
    result = graph.invoke(Command(resume="too expensive"), config)

    assert result["decision"].action == "cancel"
    assert result["payment"].status == "cancelled"


def test_offers_over_the_limit_are_never_shown(repository: PostgresPreferencesRepository) -> None:
    graph = build_graph(repository, ("no_more_offers", {}))  # the only offer costs 99.90
    config: RunnableConfig = {"configurable": {"thread_id": "over-limit"}}

    result = graph.invoke({"request": make_request("50")}, config)

    assert not graph.get_state(config).interrupts
    assert result["payment"].status == "cancelled"


def test_invalid_answer_asks_again(repository: PostgresPreferencesRepository) -> None:
    graph = build_graph(repository)
    config: RunnableConfig = {"configurable": {"thread_id": "invalid"}}
    graph.invoke({"request": make_request()}, config)

    graph.invoke(Command(resume="maybe"), config)

    paused = graph.get_state(config)
    assert "payment" not in paused.values
    [question] = paused.interrupts
    assert question.value["error"] == "Please answer yes or no."
    result = graph.invoke(Command(resume="yes"), config)
    assert result["payment"].status == "paid"


def test_new_category_asks_saves_answer_and_continues_to_recommendation(
    repository: PostgresPreferencesRepository,
) -> None:
    model = scripted(
        ("get_preferences", {"category": "coffee"}),
        ("ask_user", {"question": "Quelle torréfaction préfères-tu ?"}),
        ("update_preferences", {"category": "coffee", "changes": {"likes": ["light roast"]}}),
        ("submit_preferences", {"category": "coffee", "summary": "Prefers light roast.", "likes": ["light roast"]}),
        ("ask_searcher", {"brief": "Light roast specialty coffee beans"}),
        ("present_offer", {"offer_number": 1, "pitch": "Une torréfaction légère."}),
    )
    searcher = FakeSearcher([make_offer("Brûlerie", "18.50")])
    graph = build_real_graph(model, repository, searcher, InMemoryRecommendations())
    config: RunnableConfig = {"configurable": {"thread_id": "new-category"}}
    request = make_request().model_copy(update={"query": "Je cherche du café de spécialité"})

    graph.invoke({"request": request}, config)

    paused = graph.get_state(config)
    assert paused.next == ("user_preferences",)
    assert paused.interrupts[0].value == {"type": "preferences", "question": "Quelle torréfaction préfères-tu ?"}
    assert model.tool_choices == ["get_preferences", "ask_user"]
    assert repository.get_preferences("coffee") is None

    result = graph.invoke(Command(resume="Je préfère une torréfaction légère"), config)

    assert model.tool_choices[:4] == ["get_preferences", "ask_user", "required", "required"]  # then the recommender
    assert model.seen_inputs[2][-1].content == "Je préfère une torréfaction légère"
    stored = repository.get_preferences("coffee")
    assert stored is not None
    assert stored.preferences.likes == ["light roast"]
    assert result["preferences"].likes == ["light roast"]
    assert graph.get_state(config).next == ("recommendation",)  # waiting for the shopper's answer


def test_the_search_agent_gets_the_shoppers_limits_from_code(repository: PostgresPreferencesRepository) -> None:
    searcher = FakeSearcher([make_offer("Pro:Direct", "99.90")])
    repository.update_preferences("shoes", Preferences(likes=["Asics"]))
    graph = build_real_graph(scripted(*PREFERENCES, *RECOMMEND_FIRST), repository, searcher, InMemoryRecommendations())
    config: RunnableConfig = {"configurable": {"thread_id": "limits"}}

    graph.invoke({"request": make_request("150")}, config)

    [sent] = searcher.requests
    assert (sent.brief, sent.currency, sent.max_price) == ("Asics running shoes", "EUR", Decimal("150"))
