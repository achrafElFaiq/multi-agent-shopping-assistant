"""Data passed between the four chapters of the shopping agent.

Real implementations of a chapter may change how they work, but must keep these shapes.
"""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class ShoppingRequest(BaseModel):
    """What the user asked for. Input to the whole graph. Every field is required."""

    user_id: str
    query: str = Field(min_length=1)
    spending_limit: Decimal = Field(gt=0)  # hard maximum for this purchase
    currency: str  # e.g. "EUR"
    deliver_by: date


class UserPreferences(BaseModel):
    """Chapter 1 (User preferences) output: the user's tastes for the requested category.

    Every field is required, so whoever fills it in must decide on each one. None means unknown.
    """

    user_id: str  # always set by the code from the request, never trusted from the model
    category: str  # inferred from the query, e.g. "shoes"
    summary: str  # the user's taste and why, in plain language
    size: str | None  # size for this category, e.g. "43"
    budget: Decimal | None  # what the user usually spends in this category
    likes: list[str]  # e.g. ["Asics", "cushioned", "black"]
    avoid: list[str]  # e.g. ["narrow fit", "suede"]


class ProductOffer(BaseModel):
    """One product sold by one store."""

    product_id: str
    name: str
    store: str
    price: Decimal = Field(gt=0)
    currency: str = "EUR"
    in_stock: bool
    delivery_date: date


class Recommendation(BaseModel):
    """Chapter 2 (Search) output: the best offer and why it was picked."""

    offer: ProductOffer
    reason: str


class ApprovalDecision(BaseModel):
    """Chapter 3 (Approval) output: what the user answered."""

    action: Literal["approve", "cancel"]


class PaymentResult(BaseModel):
    """Chapter 4 (Payment) output."""

    status: Literal["paid", "cancelled", "blocked"]
    amount: Decimal | None = None
    transaction_id: str | None = None
