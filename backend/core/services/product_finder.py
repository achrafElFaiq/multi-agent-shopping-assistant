"""Collect the offers a shopper can buy from catalog searches.

Rules that hold for every product, whatever its category. What fits this particular shopper (size, colour,
requirements...) is decided by the agents, not here.
"""

import logging
from decimal import Decimal
from urllib.parse import urlparse

from backend.core.models.products import Offer
from backend.core.ports.catalog import ProductCatalog

logger = logging.getLogger(__name__)

MIN_OFFERS = 20  # enough for a real choice: more pages are fetched until the pool has this many
MAX_PAGES = 3  # per search
MAX_PER_STORE = 3  # so that a few big stores don't fill the pool


class OfferPool:
    """The offers collected over several searches: buyable, no duplicates, at most MAX_PER_STORE per store."""

    def __init__(self, country: str, currency: str, max_price: Decimal | None, seen: list[str] | None = None) -> None:
        self.country = country
        self.currency = currency
        self.max_price = max_price
        self.seen = set(seen or [])  # product ids the shopper was already shown
        self.offers: list[Offer] = []

    def add(self, offers: list[Offer]) -> int:
        """Add the offers that pass the rules, and return how many were added."""
        added = 0
        for offer in offers:
            reason = self.rejection(offer)
            if reason:
                logger.info("SKIP  %s at %s: %s", offer.title, offer.seller.name, reason)
                continue
            self.offers.append(offer)
            added += 1
        return added

    def rejection(self, offer: Offer) -> str | None:
        """Why this offer can't go in the pool, or None if it can."""
        if not offer.available:
            return "out of stock"
        if "new" not in offer.condition:
            return "not new"
        if offer.currency != self.currency:
            return f"priced in {offer.currency}"
        if self.max_price is not None and offer.price > self.max_price:
            return f"{offer.price} is over the limit of {self.max_price}"
        if urlparse(offer.seller.url).netloc.endswith(".myshopify.com"):
            return "the store has no domain of its own"
        if offer.product_id in self.seen:
            return "already shown to the shopper"
        if any(kept.product_id == offer.product_id and kept.seller == offer.seller for kept in self.offers):
            return "already in the pool"
        if sum(kept.seller == offer.seller for kept in self.offers) >= MAX_PER_STORE:
            return f"already {MAX_PER_STORE} offers from this store"
        return None


def search_into(pool: OfferPool, catalog: ProductCatalog, query: str, attributes: dict[str, list[str]]) -> list[str]:
    """Search the catalog and add what passes to the pool, fetching more pages until it has MIN_OFFERS.

    Returns what the catalog said about the search (e.g. a filter it ignored). Raises if the catalog fails.
    """
    messages: list[str] = []
    page = None
    for _ in range(MAX_PAGES):
        result = catalog.search(query, pool.country, pool.currency, pool.max_price, attributes, page)
        added = pool.add(result.offers)
        logger.info("%d results, %d added, pool %d", len(result.offers), added, len(pool.offers))
        messages += [message for message in result.messages if message not in messages]
        if len(pool.offers) >= MIN_OFFERS or not result.next_page:
            break
        page = result.next_page
    return messages
