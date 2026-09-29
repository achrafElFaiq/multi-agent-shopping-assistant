"""State shared by all nodes of the shopping graph.

The graph starts with only `request`. Each node adds its own field.
"""

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
    preferences: NotRequired[UserPreferences]  # set by memory
    recommendation: NotRequired[Recommendation]  # set by search
    decision: NotRequired[ApprovalDecision]  # set by approval
    payment: NotRequired[PaymentResult]  # set by payment
