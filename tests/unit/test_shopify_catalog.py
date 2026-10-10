"""Shopify Global Catalog adapter with simulated HTTP responses: no network."""

import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
import pytest

from backend.core.adapters.shopify_catalog import ShopifyGlobalCatalog

# The shape of a real search_catalog answer, shortened.
PRODUCT: dict[str, Any] = {
    "id": "gid://shopify/p/c1NfStVFc7VngwKWmScqo",
    "title": "Nike Pegasus Premium White Metallic Silver",
    "rating": {"value": 5.0, "count": 5},
    "variants": [
        {
            "seller": {
                "id": "gid://shopify/Shop/90214891816",
                "name": "RunningUpscale.com",
                "url": "https://runningupscale.com",
            },
            "price": {"amount": 11500, "currency": "EUR"},
            "condition": ["new"],
            "availability": {"available": True},
            "options": [{"name": "Shoe size", "label": "43 EU"}],
            "url": "https://runningupscale.com/products/nike-pegasus-premium",
            "checkout_url": "https://runningupscale.com/cart/51585777598760:1",
        },
        {"seller": {"id": "broken"}},  # missing fields: skipped
    ],
}


def catalog(answer: Callable[[httpx.Request], httpx.Response]) -> ShopifyGlobalCatalog:
    return ShopifyGlobalCatalog(
        profile_url="https://agent.example/profile.json", client=httpx.Client(transport=httpx.MockTransport(answer))
    )


def answer(*products: dict[str, Any], **content: Any) -> httpx.Response:
    return httpx.Response(200, json={"result": {"structuredContent": {"products": list(products), **content}}})


def test_search_sends_the_filters_and_reads_the_offers() -> None:
    sent: list[dict[str, Any]] = []

    def shopify(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return answer(PRODUCT, pagination={"has_next_page": True, "cursor": "next"})

    page = catalog(shopify).search("nike running shoes", "FR", "EUR", Decimal("120"), {"Color": ["Black"]})

    arguments = sent[0]["params"]["arguments"]
    assert arguments["meta"] == {"ucp-agent": {"profile": "https://agent.example/profile.json"}}
    assert arguments["catalog"]["query"] == "nike running shoes"
    filters = arguments["catalog"]["filters"]
    assert filters["ships_to"] == {"country": "FR"}
    assert filters["price"] == {"max": 12000}  # in cents
    assert filters["attributes"] == [{"name": "Color", "values": ["Black"]}]
    assert page.next_page == "next"
    assert len(page.offers) == 1  # the unreadable variant is skipped
    offer = page.offers[0]
    assert (offer.seller.name, offer.price, offer.currency) == ("RunningUpscale.com", Decimal("115"), "EUR")
    assert offer.options == {"Shoe size": "43 EU"}
    assert offer.available and offer.condition == ["new"]
    assert (offer.rating, offer.rating_count) == (5.0, 5)
    assert offer.checkout_url == "https://runningupscale.com/cart/51585777598760:1"


def test_next_page_no_filters_and_catalog_messages() -> None:
    sent: list[dict[str, Any]] = []

    def shopify(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return answer(messages=[{"content": 'Attribute "Material" is not supported and was ignored.'}])

    page = catalog(shopify).search("nike", "FR", "EUR", None, {}, page="cursor-2")

    request = sent[0]["params"]["arguments"]["catalog"]
    assert "price" not in request["filters"] and "attributes" not in request["filters"]
    assert request["pagination"]["cursor"] == "cursor-2"
    assert page.offers == [] and page.next_page is None
    assert page.messages == ['Attribute "Material" is not supported and was ignored.']


def test_currencies_without_decimals() -> None:
    sent: list[dict[str, Any]] = []
    yen = {**PRODUCT, "variants": [{**PRODUCT["variants"][0], "price": {"amount": 5000, "currency": "JPY"}}]}

    def shopify(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return answer(yen)

    page = catalog(shopify).search("tea", "JP", "JPY", Decimal("6000"), {})

    assert sent[0]["params"]["arguments"]["catalog"]["filters"]["price"] == {"max": 6000}
    assert page.offers[0].price == Decimal("5000")


def test_refused_search_raises() -> None:
    def shopify(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": -32001, "message": "UCP discovery failed"}})

    with pytest.raises(RuntimeError, match="UCP discovery failed"):
        catalog(shopify).search("nike", "FR", "EUR", None, {})


def test_http_error_raises() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        catalog(lambda request: httpx.Response(503)).search("nike", "FR", "EUR", None, {})
