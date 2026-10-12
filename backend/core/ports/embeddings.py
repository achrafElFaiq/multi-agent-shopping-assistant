"""Turning a text into a vector, so texts can be compared by meaning."""

from typing import Protocol


class Embedder(Protocol):
    dimensions: int  # length of every vector

    def embed(self, text: str) -> list[float]:
        """Return the vector of a text. Texts with close meanings get close vectors, whatever their language."""
        ...
