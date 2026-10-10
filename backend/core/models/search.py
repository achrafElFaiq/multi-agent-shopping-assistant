"""What the recommender asks the searcher, and what the searcher answers: the content of their A2A messages."""

from decimal import Decimal

from pydantic import BaseModel

from backend.core.models.products import Offer


class SearchRequest(BaseModel):
    brief: str  # written by the recommender: what to look for, and why earlier offers were refused
    country: str  # the rest is set by code, never by a model: the shopper's limits
    currency: str
    max_price: Decimal | None
    seen: list[str] = []  # product ids already shown, not to be found again


class SearchResults(BaseModel):
    queries: list[str]  # the catalog searches the searcher made
    offers: list[Offer]  # buyable, best first
