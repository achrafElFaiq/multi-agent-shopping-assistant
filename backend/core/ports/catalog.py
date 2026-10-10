"""Searching products across stores, implemented by the Shopify Global Catalog adapter."""

from decimal import Decimal
from typing import Protocol

from backend.core.models.products import CatalogPage


class ProductCatalog(Protocol):
    def search(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> CatalogPage:
        """New, in-stock offers that ship to `country`, priced in `currency` and under `max_price`, best first.

        `query` is natural language. `attributes` are catalog filters such as {"Color": ["Black"]}.
        `page` is a previous page's `next_page`. The catalog treats the filters as hints: it may still return
        offers that break them. Raises an error if the search fails.
        """
        ...
