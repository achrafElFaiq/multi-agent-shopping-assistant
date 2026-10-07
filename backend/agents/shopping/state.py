"""State shared by all nodes of the shopping graph."""

from typing import NotRequired, TypedDict

from backend.agents.shopping.schemas import (
    ApprovalDecision,
    PaymentResult,
    Recommendation,
    ShoppingRequest,
    UserPreferences,
)


class ShoppingState(TypedDict):
    request: ShoppingRequest
    preferences: NotRequired[UserPreferences]
    recommendation: NotRequired[Recommendation]
    decision: NotRequired[ApprovalDecision]
    payment: NotRequired[PaymentResult]
