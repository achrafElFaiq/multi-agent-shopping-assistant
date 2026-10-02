"""Data the user preferences tools read about a user.

Each model matches a table in the memory database (`profiles`, `recommendations`).
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """What the user told us about themselves. One per user."""

    sizes: dict[str, str] = Field(default_factory=dict)  # per category, e.g. {"shoes": "43"}
    budgets: dict[str, Decimal] = Field(default_factory=dict)  # per category, e.g. {"shoes": 130}
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)


class PastRecommendation(BaseModel):
    """A product we recommended before, and what the user answered."""

    category: str
    product_name: str
    brand: str
    price: Decimal = Field(gt=0)
    approved: bool
    reason: str | None = None  # why the user approved or rejected it, e.g. "too narrow"
    recommended_on: date
