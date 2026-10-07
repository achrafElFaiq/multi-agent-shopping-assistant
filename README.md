# AI Shopping Assistant

LangGraph shopping assistant with an OpenRouter model and PostgreSQL preferences.
One user, with preferences stored by category. Product search and payment are currently demos.

## Launch

Requires [uv](https://docs.astral.sh/uv/) and Docker. Run commands from the repository root.

1. Install dependencies and configure `.env`:

   ```bash
   uv sync --locked
   cp -n .env.example .env
   ```

   Set `OPENROUTER_API_KEY` in `.env`. `LLM_MODEL` selects the model.

2. Start PostgreSQL with the credentials from `.env.example`:

   ```bash
   docker run -d --name shopping-postgres \
     -e POSTGRES_USER=shopping \
     -e POSTGRES_PASSWORD=shopping \
     -e POSTGRES_DB=shopping \
     -p 127.0.0.1:5432:5432 \
     -v shopping-postgres-data:/var/lib/postgresql/data \
     postgres:17
   ```

   Once PostgreSQL is ready, create the test database:

   ```bash
   docker exec shopping-postgres createdb -U shopping shopping_test
   ```

   For an existing PostgreSQL server, configure `DATABASE_URL` and `TEST_DATABASE_URL` in `.env` and skip this step.

3. Seed example preferences and launch the agent:

   ```bash
   uv run python -m backend.seed
   uv run python -m backend.agents.shopping "running shoes" \
     --limit 120 --currency EUR --deliver-by 2026-10-20
   ```

   Enter `approve` or `cancel` when prompted. Add `--raw` to see the HTTP request and response bodies.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Tests use `TEST_DATABASE_URL` with temporary schemas and a scripted model. No API key is needed.
