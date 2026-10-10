"""Payment's two rules, checked directly: they must hold even if an earlier step lets something through."""

from datetime import date
from decimal import Decimal

from backend.agents.shopping.nodes.payment import pay
from backend.agents.shopping.schemas import ApprovalDecision, ProductOffer, Recommendation, ShoppingRequest
from backend.agents.shopping.state import ShoppingState


def state(action: str, price: str, limit: str = "100") -> ShoppingState:
    offer = ProductOffer(product_id="p", name="Pegasus 42", store="Pro:Direct", price=Decimal(price), in_stock=True)
    return {
        "request": ShoppingRequest(
            query="running shoes", spending_limit=Decimal(limit), currency="EUR", deliver_by=date(2026, 10, 20)
        ),
        "recommendation": Recommendation(offer=offer, reason="test"),
        "decision": ApprovalDecision.model_validate({"action": action}),
    }


def test_never_pays_without_approval() -> None:
    assert pay(state("cancel", "50"))["payment"].status == "cancelled"


def test_never_pays_over_the_limit() -> None:
    assert pay(state("approve", "150"))["payment"].status == "blocked"


def test_pays_when_approved_and_under_the_limit() -> None:
    assert pay(state("approve", "50"))["payment"].status == "paid"
