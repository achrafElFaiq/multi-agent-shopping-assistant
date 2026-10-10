"""The search agent, with a scripted model and a fake catalog: no network, no API key."""

from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from backend.agents.searcher.agent import find_offers
from backend.agents.searcher.tools import MAX_SEARCHES
from backend.core.models.products import CatalogPage
from backend.core.models.search import SearchRequest
from tests.fakes import FakeCatalog, FakeChatModel, catalog_with, make_offer


def script(*calls: tuple[str, dict[str, Any]]) -> FakeChatModel:
    replies = [
        AIMessage("", tool_calls=[ToolCall(name=name, args=args, id=f"{name}-{n}")])
        for n, (name, args) in enumerate(calls)
    ]
    return FakeChatModel(messages=iter(replies))


def request(brief: str = "Asics running shoes, size 43, men", seen: list[str] | None = None) -> SearchRequest:
    return SearchRequest(brief=brief, country="FR", currency="EUR", max_price=Decimal("120"), seen=seen or [])


def test_searches_with_the_agents_words_and_filters_within_the_limits() -> None:
    model = script(
        ("search_catalog", {"query": "Asics cushioned running shoes", "size": ["43"], "target_gender": ["Male"]}),
        ("submit_search", {}),
    )
    catalog = catalog_with(make_offer("Pro:Direct", "95"), make_offer("Lovell", "200"))  # 200 is over the limit

    results = find_offers(model, catalog, request())

    assert catalog.searches == [("Asics cushioned running shoes", {"Size": ["43"], "Target gender": ["Male"]}, None)]
    assert results.queries == ["Asics cushioned running shoes"]
    assert [offer.seller.name for offer in results.offers] == ["Pro:Direct"]
    assert model.seen_inputs[0][-1].content == "Asics running shoes, size 43, men"  # the brief
    summary = model.seen_inputs[1][-1]
    assert isinstance(summary, ToolMessage)
    assert "Pool: 1 offers" in str(summary.content) and "Pro:Direct | 95" in str(summary.content)


def test_offers_already_shown_are_not_found_again() -> None:
    model = script(("search_catalog", {"query": "running shoes"}), ("submit_search", {}))
    shown, new = make_offer("A"), make_offer("B")

    results = find_offers(model, catalog_with(shown, new), request(seen=[shown.product_id]))

    assert [offer.seller.name for offer in results.offers] == ["B"]


def test_searches_again_from_another_angle_up_to_the_limit() -> None:
    calls = [("search_catalog", {"query": f"query {n}"}) for n in range(MAX_SEARCHES + 1)]
    model = script(*calls, ("submit_search", {}))
    catalog = FakeCatalog(*(CatalogPage(offers=[make_offer(f"S{n}")], messages=[], next_page=None) for n in range(5)))

    results = find_offers(model, catalog, request())

    assert len(catalog.searches) == MAX_SEARCHES
    assert "Search limit" in str(model.seen_inputs[MAX_SEARCHES + 1][-1].content)
    assert len(results.offers) == MAX_SEARCHES


def test_a_failed_catalog_search_is_reported_to_the_agent() -> None:
    class BrokenCatalog(FakeCatalog):
        def search(self, *args: Any, **kwargs: Any) -> CatalogPage:
            raise ConnectionError("no network")

    model = script(("search_catalog", {"query": "running shoes"}), ("submit_search", {}))

    results = find_offers(model, BrokenCatalog(), request())

    assert results.offers == []
    assert "The search failed (no network)" in str(model.seen_inputs[1][-1].content)
