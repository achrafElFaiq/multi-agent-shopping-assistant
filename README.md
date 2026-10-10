# AI Shopping Assistant

LangGraph shopping assistant with an OpenRouter model and PostgreSQL preferences.
One user, with preferences stored by category. A recommendation agent works with a search agent over A2A:
the search agent searches every Shopify store at once (Shopify Global Catalog, no API key), the recommendation
agent presents the offers one by one and, when the user says no, asks it to find better, telling it why.
Saving the answers and payment are currently mocks.

## Launch

Requires [uv](https://docs.astral.sh/uv/) and a running PostgreSQL server. Run commands from the repository root.

1. Install dependencies and configure `.env`:

   ```bash
   uv sync --locked
   cp -n .env.example .env
   ```

   Set `OPENROUTER_API_KEY` and `LLM_MODEL` in `.env`.

2. Create a database on your PostgreSQL server and set `DATABASE_URL` in `.env`.
   Replace the example credentials with your database name, username, and password.

3. Start the search agent (its A2A server) in one terminal:

   ```bash
   uv run python -m backend.agents.searcher
   ```

   Its Agent Card is at http://localhost:8001/.well-known/agent-card.json

4. Launch the shopping agent in another terminal:

   ```bash
   uv run python -m backend.agents.shopping "running shoes" \
     --limit 120 --currency EUR --deliver-by 2026-10-20
   ```

   It uses the example preferences of `backend/seed.py`, kept in memory for now, so steps 2 and
   `uv run python -m backend.seed` are only needed for PostgreSQL. Answer `yes` or `no` to each recommendation,
   and say why after a no. Add `--raw` to see the HTTP request and
   response bodies.

To try the catalog search alone, without a model or a key:
`uv run python -m backend.scripts.find_products "nike running shoes" --max-price 120 --size 43`

## Checks

Create a separate PostgreSQL database for tests and set `TEST_DATABASE_URL` in `.env`.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Tests use `TEST_DATABASE_URL` with temporary schemas, a scripted model and fake search agent and catalog.
No API key is needed. `uv run python -m tests.benchmark.run record` records 25 catalog searches once (free); without `record`, it replays them.
