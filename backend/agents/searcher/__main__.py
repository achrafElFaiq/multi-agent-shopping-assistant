"""Start the search agent as an A2A server: uv run python -m backend.agents.searcher

Needs OPENROUTER_API_KEY, LLM_MODEL and DATABASE_URL in .env (the model is the one the search agent uses).
Its Agent Card is then at http://localhost:8001/.well-known/agent-card.json
"""

import argparse
import logging

import uvicorn

from backend.agents.searcher.server import create_app
from backend.config.settings import load_settings
from backend.core.adapters.openrouter import make_chat_model
from backend.core.adapters.shopify_catalog import ShopifyGlobalCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the search agent as an A2A server.")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    url = f"http://localhost:{args.port}/"
    app = create_app(make_chat_model(load_settings()), ShopifyGlobalCatalog(), url)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
