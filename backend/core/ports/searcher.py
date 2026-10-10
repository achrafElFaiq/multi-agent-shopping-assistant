"""Asking the search agent for offers, implemented over A2A."""

from typing import Protocol

from backend.core.models.search import SearchRequest, SearchResults


class ProductSearcher(Protocol):
    def search(self, request: SearchRequest) -> SearchResults:
        """Offers for the request's brief, within its limits. Raises an error if the searcher fails."""
        ...
