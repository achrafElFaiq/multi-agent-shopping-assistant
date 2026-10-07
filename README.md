# AI Shopping Assistant

LangGraph shopping assistant with an OpenRouter model and PostgreSQL preferences.
One user, with preferences stored by category. Product search and payment are currently demos.

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

3. Seed example preferences and launch the agent:

   ```bash
   uv run python -m backend.seed
   uv run python -m backend.agents.shopping "running shoes" \
     --limit 120 --currency EUR --deliver-by 2026-10-20
   ```

   Enter `approve` or `cancel` when prompted. Add `--raw` to see the HTTP request and response bodies.

## Checks

Create a separate PostgreSQL database for tests and set `TEST_DATABASE_URL` in `.env`.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Tests use `TEST_DATABASE_URL` with temporary schemas and a scripted model. No API key is needed.
