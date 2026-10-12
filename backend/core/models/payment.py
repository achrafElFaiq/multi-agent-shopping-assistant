"""The result of paying for an offer."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class PaymentResult(BaseModel):
    status: Literal["paid", "cancelled", "blocked"]
    amount: Decimal | None = None
    transaction_id: str | None = None
