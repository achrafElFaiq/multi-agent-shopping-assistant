"""Benchmark for the product search: does it find offers from the good stores?

    uv run python -m tests.benchmark.run record       # search the live Shopify catalog and save its answers (once)
    uv run python -m tests.benchmark.run              # replay the recordings: no network, same answers every time
    uv run python -m tests.benchmark.run candidates   # list every store seen, to label in cases.json

Add a case id to run only that case, and -v to see the search's logs. No API key needed.
It measures the search without the agent: each case's words are sent as they are, with no filters.

The catalog changes every day, so the benchmark replays a recording of its answers. Recordings are not in
git (they are large): run `record` first, it is free and takes about 15 seconds.
A change in the result then comes from the code, not from the catalog.
Labels in cases.json are decided by a person: `good_stores` should be returned, `bad_stores` should not.
Stores are labelled by their website, e.g. "https://www.prodirectsport.fr".
"""

import json
import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.core.adapters.shopify_catalog import ShopifyGlobalCatalog, offers_of, page_of
from backend.core.models.products import CatalogPage, Offer
from backend.core.services.product_finder import OfferPool, search_into

HERE = Path(__file__).parent
CASES = HERE / "cases.json"
RECORDINGS = HERE / "recordings"
TOP_N = 5


class RecordingCatalog(ShopifyGlobalCatalog):
    """The real catalog, keeping its raw answer for each page."""

    def __init__(self) -> None:
        super().__init__()
        self.pages: dict[str, Any] = {}

    def answer(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> dict[str, Any]:
        content = super().answer(query, country, currency, max_price, attributes, page)
        self.pages[page or "first"] = content
        return content


class ReplayCatalog:
    """Answers with the recorded pages, read by the real adapter's reader."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages

    def search(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> CatalogPage:
        return page_of(self.pages.get(page or "first", {}))


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg != "-v"]
    mode = args.pop(0) if args and args[0] in ("record", "candidates") else "replay"
    logging.basicConfig(level=logging.INFO if "-v" in sys.argv else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cases = [case for case in json.loads(CASES.read_text()) if not args or case["id"] in args]
    if mode == "candidates":
        show_candidates(cases)
        return

    results = []
    for case in cases:
        if mode == "record":
            stores = record(case)
        elif (RECORDINGS / f"{case['id']}.json").exists():
            stores = replay(case)
        else:
            print(f"{case['id']}: no recording, run `record` first")
            continue
        results.append((case, stores))
    show_scores(results)


def record(case: dict[str, Any]) -> list[str]:
    """Search the live catalog and save its answers."""
    catalog = RecordingCatalog()
    stores = search(case, catalog)
    RECORDINGS.mkdir(exist_ok=True)
    (RECORDINGS / f"{case['id']}.json").write_text(json.dumps(catalog.pages, ensure_ascii=False, indent=1))
    return stores


def replay(case: dict[str, Any]) -> list[str]:
    """Search the recorded answers: no network."""
    return search(case, ReplayCatalog(json.loads((RECORDINGS / f"{case['id']}.json").read_text())))


def search(case: dict[str, Any], catalog: RecordingCatalog | ReplayCatalog) -> list[str]:
    """The first TOP_N stores of the pool, in the order of their best offer."""
    max_price = Decimal(case["max_price"]) if case.get("max_price") else None
    pool = OfferPool(case["country"], case["currency"], max_price)
    search_into(pool, catalog, case["query"], {})
    stores = list(dict.fromkeys(offer.seller.url for offer in pool.offers))
    return stores[:TOP_N]


def show_scores(results: list[tuple[dict[str, Any], list[str]]]) -> None:
    """One line per case, then the totals.

    recall: share of the good stores that were returned. precision: share of returned stores that are not bad.
    """
    print(f"\n{'case':32} {'stores':>6} {'recall':>7} {'precision':>9}  returned (✓ good, ✗ bad)")
    recalls, precisions, covered = [], [], 0
    for case, stores in results:
        good, bad = set(case["good_stores"]), set(case["bad_stores"])
        covered += bool(stores)
        recall = f"{len(good & set(stores))}/{len(good)}" if good else "-"
        precision = f"{len(stores) - len(bad & set(stores))}/{len(stores)}" if stores and (good or bad) else "-"
        if good:
            recalls.append(len(good & set(stores)) / len(good))
        if stores and (good or bad):
            precisions.append(1 - len(bad & set(stores)) / len(stores))
        names = [store.removeprefix("https://") for store in stores]
        marked = [f"{n} ✓" if s in good else f"{n} ✗" if s in bad else n for n, s in zip(names, stores, strict=True)]
        print(f"{case['id']:32} {len(stores):>6} {recall:>7} {precision:>9}  {', '.join(marked) or 'none'}")

    def average(values: list[float]) -> str:
        return f"{sum(values) / len(values):.0%}" if values else "- (no labels yet)"

    print(f"\ncases with at least one store: {covered}/{len(results)}")
    print(f"recall@{TOP_N}: {average(recalls)}   precision: {average(precisions)}")


def show_candidates(cases: list[dict[str, Any]]) -> None:
    """Every store in the recordings, with its offers, to label it good or bad."""
    for case in cases:
        path = RECORDINGS / f"{case['id']}.json"
        if not path.exists():
            continue
        print(f"\n{case['id']}  ({case['query']!r}, {case['country']}, {case['currency']})")
        by_store: dict[str, list[Offer]] = {}
        for content in json.loads(path.read_text()).values():
            for offer in offers_of(content.get("products") or []):
                by_store.setdefault(offer.seller.url, []).append(offer)
        for url, offers in by_store.items():
            example = offers[0]
            print(f"  {url:45} {len(offers):>2} offers, e.g. {example.title[:40]} {example.price} {example.currency}")


if __name__ == "__main__":
    main()
