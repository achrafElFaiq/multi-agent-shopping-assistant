"""The offer pool's rules and page fetching, with a fake catalog: no network."""

from decimal import Decimal

from backend.core.models.products import CatalogPage
from backend.core.services.product_finder import MAX_PAGES, MAX_PER_STORE, MIN_OFFERS, OfferPool, search_into
from tests.fakes import FakeCatalog, make_offer


def pool() -> OfferPool:
    return OfferPool("FR", "EUR", Decimal("120"))


def test_only_offers_the_shopper_can_buy_are_kept() -> None:
    offers = [
        make_offer("sold-out", available=False),
        make_offer("used", condition=["secondhand"]),
        make_offer("dollars", currency="USD"),
        make_offer("too-expensive", "130"),
        make_offer("no-domain", seller=make_offer("x").seller.model_copy(update={"url": "https://b2e2.myshopify.com"})),
        make_offer("fine"),
    ]
    collected = pool()

    assert collected.add(offers) == 1
    assert [offer.seller.name for offer in collected.offers] == ["fine"]


def test_no_duplicates_and_at_most_a_few_offers_per_store() -> None:
    collected = pool()
    collected.add([make_offer("big", "90"), make_offer("big", "90")])  # the same offer twice
    collected.add([make_offer("big", str(price)) for price in range(50, 60)])

    assert len(collected.offers) == MAX_PER_STORE
    assert collected.offers[0].price == Decimal("90")


def test_fetches_more_pages_until_the_pool_is_big_enough() -> None:
    def page(start: int, next_page: str | None) -> CatalogPage:
        return CatalogPage(
            offers=[make_offer(f"store{n}") for n in range(start, start + 8)], messages=[], next_page=next_page
        )

    catalog = FakeCatalog(page(0, "2"), page(8, "3"), page(16, "4"), page(24, None))
    collected = pool()

    search_into(collected, catalog, "running shoes", {"Size": ["43"]})

    assert len(collected.offers) >= MIN_OFFERS
    assert [page for _, _, page in catalog.searches] == [None, "2", "3"]  # stopped once there were enough
    assert len(catalog.searches) <= MAX_PAGES


def test_stops_when_there_are_no_more_pages_and_returns_the_catalog_notes() -> None:
    catalog = FakeCatalog(CatalogPage(offers=[make_offer()], messages=["Color was ignored"], next_page=None))

    messages = search_into(pool(), catalog, "running shoes", {})

    assert messages == ["Color was ignored"]
    assert len(catalog.searches) == 1
