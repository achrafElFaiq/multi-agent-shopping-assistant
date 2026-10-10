"""ProductCatalog with Shopify's Global Catalog: one search covers every participating Shopify store.

https://shopify.dev/docs/agents/catalog/global-catalog. No API key: each request points to our agent profile
(ucp/agent-profile.json), a public file that Shopify fetches to see who is asking.
The request is an MCP tool call (`search_catalog`) whose arguments and answer follow UCP.
"""

import logging
import re
from decimal import Decimal
from typing import Any

import httpx
from pydantic import ValidationError

from backend.core.models.products import CatalogPage, Offer, Seller

CATALOG_URL = "https://catalog.shopify.com/api/ucp/mcp"

# ucp/agent-profile.json, served by jsDelivr from this public repository, pinned to the commit that added it.
# Shopify needs it served as application/json, which GitHub's raw files are not.
AGENT_PROFILE_URL = (
    "https://cdn.jsdelivr.net/gh/achrafElFaiq/multi-agent-shopping-assistant"
    "@2a9e68bc50670ef6af58cd072cae08fa0d56779f/ucp/agent-profile.json"
)

DESCRIPTION_LENGTH = 400  # characters kept from each description: enough to check details

# Prices are integers in the currency's smallest unit: cents for EUR (11500 = 115.00), but yen for JPY.
ZERO_DECIMAL_CURRENCIES = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "XAF", "XOF"}

logger = logging.getLogger(__name__)


class ShopifyGlobalCatalog:
    def __init__(self, profile_url: str = AGENT_PROFILE_URL, limit: int = 50, client: httpx.Client | None = None):
        self._profile_url = profile_url
        self._limit = limit  # results per page, 50 at most
        self._client = client or httpx.Client(timeout=30)

    def search(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> CatalogPage:
        return page_of(self.answer(query, country, currency, max_price, attributes, page))

    def answer(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> dict[str, Any]:
        """The catalog's raw answer for one page of results."""
        filters: dict[str, Any] = {"ships_to": {"country": country}, "available": True, "condition": ["new"]}
        if max_price is not None:
            filters["price"] = {"max": to_minor_units(max_price, currency)}
        if attributes:
            filters["attributes"] = [{"name": name, "values": values} for name, values in attributes.items()]
        pagination: dict[str, Any] = {"limit": self._limit}
        if page:
            pagination["cursor"] = page
        arguments = {
            "meta": {"ucp-agent": {"profile": self._profile_url}},
            "catalog": {
                "query": query,
                "context": {"address_country": country, "currency": currency},
                "filters": filters,
                "pagination": pagination,
            },
        }
        logger.info("Shopify catalog: %r, filters %s%s", query, attributes or "none", ", next page" if page else "")

        response = self._client.post(
            CATALOG_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_catalog", "arguments": arguments},
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        response.raise_for_status()
        answer = response.json()
        if "error" in answer:
            raise RuntimeError(f"Shopify catalog refused the search: {answer['error'].get('message')}")
        if answer["result"].get("isError"):
            raise RuntimeError(f"Shopify catalog search failed: {answer['result'].get('content')}")
        content: dict[str, Any] = answer["result"]["structuredContent"]
        return content


def page_of(content: dict[str, Any]) -> CatalogPage:
    """Read one page of the catalog's answer."""
    pagination = content.get("pagination") or {}
    return CatalogPage(
        offers=offers_of(content.get("products") or []),
        messages=[message.get("content", "") for message in content.get("messages") or []],
        next_page=pagination.get("cursor") if pagination.get("has_next_page") else None,
    )


def offers_of(products: list[dict[str, Any]]) -> list[Offer]:
    """One offer per product variant, in the catalog's order. Variants that can't be read are skipped."""
    offers = []
    for product in products:
        rating = product.get("rating") or {}
        for variant in product.get("variants", []):
            try:
                offers.append(
                    Offer(
                        product_id=product["id"],
                        title=product["title"],
                        description=plain_text((product.get("description") or {}).get("plain", "")),
                        seller=Seller(
                            shopify_id=variant["seller"]["id"],
                            name=variant["seller"]["name"],
                            url=variant["seller"].get("url", ""),
                        ),
                        price=from_minor_units(variant["price"]["amount"], variant["price"]["currency"]),
                        currency=variant["price"]["currency"],
                        condition=variant.get("condition") or [],
                        available=bool((variant.get("availability") or {}).get("available")),
                        options={option["name"]: option["label"] for option in variant.get("options") or []},
                        url=variant.get("url", ""),
                        checkout_url=variant.get("checkout_url") or "",
                        rating=rating.get("value"),
                        rating_count=rating.get("count", 0),
                    )
                )
            except (KeyError, TypeError, ValidationError) as error:
                logger.info("skipped a variant of %r that could not be read: %s", product.get("title"), error)
    return offers


def plain_text(text: str) -> str:
    """The catalog's "plain" descriptions can still contain HTML: keep the words only, shortened."""
    words = re.sub(r"<[^>]+>", " ", text).split()
    return " ".join(words)[:DESCRIPTION_LENGTH]


def to_minor_units(amount: Decimal, currency: str) -> int:
    return int(amount if currency in ZERO_DECIMAL_CURRENCIES else amount * 100)


def from_minor_units(amount: int, currency: str) -> Decimal:
    return Decimal(amount) if currency in ZERO_DECIMAL_CURRENCIES else Decimal(amount) / 100
