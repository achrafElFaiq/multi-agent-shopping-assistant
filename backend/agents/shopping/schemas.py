"""Data passed between the four chapters of the shopping agent."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from backend.core.ports.preferences import Preferences


class ShoppingRequest(BaseModel):
    """What the user asked for. Input to the whole graph. Every field is required."""

    query: str = Field(min_length=1)
    spending_limit: Decimal = Field(gt=0)
    currency: str
    deliver_by: date


class UserPreferences(Preferences):
    """Preferences for the shopping request, with its category and a summary."""

    category: str = Field(min_length=1)
    summary: str


class ProductOffer(BaseModel):
    """One product sold by one store."""

    product_id: str
    name: str
    store: str
    price: Decimal = Field(gt=0)
    currency: str = "EUR"
    in_stock: bool
    delivery_date: date | None = None  # None until the store gives one


class Recommendation(BaseModel):
    """Chapter 2 output: the offer the shopper accepted and why it was recommended."""

    offer: ProductOffer
    reason: str


class ApprovalDecision(BaseModel):
    """Chapter 2 output: whether the shopper accepted an offer."""

    action: Literal["approve", "cancel"]


class PaymentResult(BaseModel):
    """Chapter 3 (Payment) output."""

    status: Literal["paid", "cancelled", "blocked"]
    amount: Decimal | None = None
    transaction_id: str | None = None
