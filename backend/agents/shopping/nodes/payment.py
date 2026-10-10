"""Chapter 4: Payment.

Stub: fakes the checkout. Will be replaced by real payment (AP2, ACP, Stripe test mode).
The two safety rules below must stay in the real version.
"""

from uuid import uuid4

from backend.agents.shopping.schemas import PaymentResult
from backend.agents.shopping.state import ShoppingState


def pay(state: ShoppingState) -> dict[str, PaymentResult]:
    # Rule 1: never pay without the user's approval.
    if state["decision"].action != "approve":
        return {"payment": PaymentResult(status="cancelled")}

    offer = state["recommendation"].offer

    # Rule 2: never go over the spending limit.
    if offer.price > state["request"].spending_limit:
        return {"payment": PaymentResult(status="blocked", amount=offer.price)}

    return {
        "payment": PaymentResult(
            status="paid",
            amount=offer.price,
            transaction_id=f"test_{uuid4().hex[:12]}",
        )
    }
