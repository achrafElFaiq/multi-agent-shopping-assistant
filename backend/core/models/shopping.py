"""What the user asks the master agent for."""

from decimal import Decimal

from pydantic import BaseModel, Field


class ShoppingRequest(BaseModel):
    """The free text, plus the limits the code enforces: they never come from a model."""

    query: str = Field(min_length=1)
    spending_limit: Decimal = Field(gt=0)
    currency: str
