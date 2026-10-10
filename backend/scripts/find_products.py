"""Try the product search without the agent: your words go to the catalog as they are.

    uv run python -m backend.scripts.find_products "nike running shoes" --max-price 120 --size 43 --color Black

Options: --country FR --currency EUR --max-price --size --color --gender. No API key needed.
Shows the pool the search agent would hand to the recommendation step for this one search.
"""

import argparse
import logging
from decimal import Decimal

from backend.core.adapters.shopify_catalog import ShopifyGlobalCatalog
from backend.core.services.product_finder import OfferPool, search_into


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Shopify catalog and show the offers kept.")
    parser.add_argument("query", nargs="+", help='what to buy, e.g. "nike running shoes"')
    parser.add_argument("--country", default="FR", help="shopper's country code, e.g. FR")
    parser.add_argument("--currency", default="EUR", help="shopper's currency, e.g. EUR")
    parser.add_argument("--max-price", type=Decimal, help="spending limit, e.g. 120")
    parser.add_argument("--size", nargs="*", help="catalog Size filter, e.g. 43")
    parser.add_argument("--color", nargs="*", help="catalog Color filter, e.g. Black")
    parser.add_argument("--gender", nargs="*", help="catalog Target gender filter, e.g. Male")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # the adapter logs its own calls

    attributes = {
        name: values
        for name, values in [("Size", args.size), ("Color", args.color), ("Target gender", args.gender)]
        if values
    }
    pool = OfferPool(args.country, args.currency, args.max_price)
    for message in search_into(pool, ShopifyGlobalCatalog(), " ".join(args.query), attributes):
        print(f"catalog note: {message}")

    print(f"\n{len(pool.offers)} offers from {len({offer.seller.name for offer in pool.offers})} stores")
    for number, offer in enumerate(pool.offers, start=1):
        options = ", ".join(f"{name} {value}" for name, value in offer.options.items())
        print(f"{number:>2}. {offer.title} | {offer.seller.name} | {offer.price} {offer.currency} | {options or '-'}")


if __name__ == "__main__":
    main()
