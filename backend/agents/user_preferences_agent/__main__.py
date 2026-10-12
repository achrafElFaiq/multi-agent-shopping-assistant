"""Start the user preferences agent as an A2A server: uv run python -m backend.agents.user_preferences_agent

Needs OPENROUTER_API_KEY, LLM_MODEL and DATABASE_URL in .env, and PostgreSQL with pgvector running (past answers).
Preferences are mocks for now: the example ones of backend/seed.py, kept in memory.
Its Agent Card is then at http://localhost:8002/.well-known/agent-card.json
"""

import argparse
import logging

import uvicorn

from backend.agents.user_preferences_agent.server import create_app
from backend.config.settings import load_settings
from backend.core.adapters.fastembed_embedder import FastEmbedEmbedder
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.openrouter import make_chat_model
from backend.core.adapters.postgres_recommendations import PostgresRecommendations
from backend.seed import SEED_PREFERENCES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the user preferences agent as an A2A server.")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = load_settings()
    # Mock preferences for now. For PostgreSQL: PostgresPreferencesRepository(settings.database_url.get_secret_value())
    preferences = InMemoryPreferencesRepository(SEED_PREFERENCES)
    recommendations = PostgresRecommendations(settings.database_url.get_secret_value(), FastEmbedEmbedder())
    url = f"http://localhost:{args.port}/"
    app = create_app(make_chat_model(settings), preferences, recommendations, url)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
