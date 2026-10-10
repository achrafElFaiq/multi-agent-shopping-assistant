"""Tools for the search agent: search the Shopify catalog, then submit what was found.

The offers stay in the pool, in code: the model only sees a summary, so it can't change a price or a link.
"""

from langchain_core.tools import BaseTool, tool

from backend.core.ports.catalog import ProductCatalog
from backend.core.services.product_finder import MIN_OFFERS, OfferPool, search_into

MAX_SEARCHES = 3


def make_search_tools(catalog: ProductCatalog, pool: OfferPool, queries: list[str]) -> list[BaseTool]:
    """`queries` records each search made, for the node's output."""

    @tool
    def search_catalog(
        query: str,
        size: list[str] | None = None,
        color: list[str] | None = None,
        target_gender: list[str] | None = None,
    ) -> str:
        """Search new, in-stock products across all Shopify stores. Price, delivery country and currency are
        applied by the system.

        query: what the shopper wants, in natural language: the product, plus brand, use, likes and details
            that are not filters (e.g. "16 GB RAM", "water resistant").
        size, color, target_gender: catalog filters, only when the request or preferences give them clearly,
            written as stores do, e.g. ["43"], ["M"], ["Black"], ["Male"]. Several values mean "any of them".
        """
        if len(queries) >= MAX_SEARCHES:
            return f"Search limit of {MAX_SEARCHES} reached: call submit_search."
        queries.append(query)
        attributes = {
            name: values
            for name, values in [("Size", size), ("Color", color), ("Target gender", target_gender)]
            if values
        }
        before = len(pool.offers)
        try:
            messages = search_into(pool, catalog, query, attributes)
        except Exception as error:
            return f"The search failed ({error}). Try again, or call submit_search."
        return summary(pool, len(pool.offers) - before, messages)

    @tool(return_direct=True)
    def submit_search() -> str:
        """Finish: the offers collected so far go to the recommendation step. Call this tool alone."""
        return "Search submitted."

    return [search_catalog, submit_search]


def summary(pool: OfferPool, added: int, messages: list[str]) -> str:
    """What the model needs to judge the search: how many offers, and what they are."""
    stores = {offer.seller.name for offer in pool.offers}
    lines = [f"{added} new offers. Pool: {len(pool.offers)} offers from {len(stores)} stores (aim: {MIN_OFFERS})."]
    lines += [f"Catalog note: {message}" for message in messages]
    for offer in pool.offers:
        options = ", ".join(f"{name} {value}" for name, value in offer.options.items())
        lines.append(f"- {offer.title} | {offer.seller.name} | {offer.price} {offer.currency} | {options or '-'}")
    return "\n".join(lines)
