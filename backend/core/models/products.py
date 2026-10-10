"""Products as the catalog returns them: one offer is one product variant sold by one store."""

from decimal import Decimal

from pydantic import BaseModel


class Seller(BaseModel):
    shopify_id: str  # "gid://shopify/Shop/90214891816": the same for every domain of one store
    name: str
    url: str


class Offer(BaseModel):
    product_id: str
    title: str
    description: str = ""  # plain text, shortened
    seller: Seller
    price: Decimal  # in currency units: 115.00
    currency: str
    condition: list[str]  # "new", "secondhand"
    available: bool
    options: dict[str, str]  # the variant offered: {"Shoe size": "43 EU", "Color": "Black"}
    url: str  # the product page
    checkout_url: str  # a cart with this variant already in it
    rating: float | None = None  # out of 5
    rating_count: int = 0


class CatalogPage(BaseModel):
    """One page of catalog results."""

    offers: list[Offer]  # best first
    messages: list[str]  # what the catalog says about the search, e.g. a filter it ignored
    next_page: str | None  # where the next page starts, None when there are no more results
