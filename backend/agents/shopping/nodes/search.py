"""Chapter 2: Product Search.

Stub: returns one fixed offer. Will be replaced by real store search (A2A, MCP, hybrid RAG).
"""

from datetime import date, timedelta
from decimal import Decimal

from backend.agents.shopping.schemas import ProductOffer, Recommendation
from backend.agents.shopping.state import ShoppingState


def search_products(state: ShoppingState) -> dict[str, Recommendation]:
    preferences = state["preferences"]
    offer = ProductOffer(
        product_id="asics-gel-nimbus-26",
        name="Asics Gel-Nimbus 26",
        store="Demo Store",
        price=Decimal("99.90"),
        in_stock=True,
        delivery_date=date.today() + timedelta(days=3),
    )
    recommendation = Recommendation(
        offer=offer,
        reason=f"Matches your favourite brands ({', '.join(preferences.brands)}) and is in stock.",
    )
    return {"recommendation": recommendation}
