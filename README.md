# AI Shopping Assistant

Three LangGraph agents talking over A2A, for one user with preferences stored by category:
- **master** (`backend/agents/master_agent`): talks to the user, delegates, presents offers one by one, pays.
- **user preferences** (`backend/agents/user_preferences_agent`, A2A on :8002): reads the stored preferences
  and past answers, asks the user when something essential is missing, saves lasting preferences.
- **product finder** (`backend/agents/product_finder_agent`, A2A on :8001): searches every Shopify store at once
  (Shopify Global Catalog, no API key).

When the user says no, the master asks why and asks the product finder to find better, telling it why.
Answers are kept in PostgreSQL (pgvector). Preferences and payment are currently mocks.
The previous single-graph version is still in `backend/agents/shopping`.

## Launch

Requires [uv](https://docs.astral.sh/uv/) and a running PostgreSQL server with pgvector. Run commands from the repository root.

1. Install dependencies and configure `.env`:

   ```bash
   uv sync --locked
   cp -n .env.example .env
   ```

   Set `OPENROUTER_API_KEY` and `LLM_MODEL` in `.env`.

2. Create a database on your PostgreSQL server and set `DATABASE_URL` in `.env`.
   Replace the example credentials with your database name, username, and password.

3. Start the two A2A agents, each in its own terminal:

   ```bash
   uv run python -m backend.agents.product_finder_agent     # :8001
   uv run python -m backend.agents.user_preferences_agent   # :8002
   ```

   Their Agent Cards are at http://localhost:8001/.well-known/agent-card.json and
   http://localhost:8002/.well-known/agent-card.json

4. Launch the master in a third terminal:

   ```bash
   uv run python -m backend.agents.master_agent "running shoes" --limit 120 --currency EUR
   ```

   Preferences are the example ones of `backend/seed.py`, kept in memory for now. Answer the preferences agent's
   questions, then `yes` or `no` to each offer, and say why after a no.

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
