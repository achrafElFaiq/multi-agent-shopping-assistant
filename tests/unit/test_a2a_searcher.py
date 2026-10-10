"""The real A2A exchange: the searcher's A2A server and the recommender's A2A client, connected in memory."""

from decimal import Decimal

import httpx
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from backend.agents.searcher.server import create_app
from backend.core.adapters.a2a_searcher import A2ASearcher
from backend.core.models.search import SearchRequest
from tests.fakes import FakeChatModel, catalog_with, make_offer

URL = "http://searcher.test/"


def searcher_server() -> httpx.AsyncClient:
    model = FakeChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[ToolCall(name="search_catalog", args={"query": "running shoes"}, id="s")]),
                AIMessage("", tool_calls=[ToolCall(name="submit_search", args={}, id="done")]),
            ]
        )
    )
    app = create_app(model, catalog_with(make_offer("Pro:Direct", "95")), URL)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL)


def test_agent_card() -> None:
    import asyncio

    async def get_card() -> dict[str, object]:
        response = await searcher_server().get("/.well-known/agent-card.json")
        card: dict[str, object] = response.json()
        return card

    card = asyncio.run(get_card())
    assert card["name"] == "Product searcher"
    assert card["skills"][0]["id"] == "search_products"  # type: ignore[index]


def test_search_request_and_results_travel_over_a2a() -> None:
    searcher = A2ASearcher(URL, http_client=searcher_server())

    results = searcher.search(
        SearchRequest(brief="Asics running shoes, size 43", country="FR", currency="EUR", max_price=Decimal("120"))
    )

    assert results.queries == ["running shoes"]
    assert [(offer.seller.name, offer.price) for offer in results.offers] == [("Pro:Direct", Decimal("95"))]
