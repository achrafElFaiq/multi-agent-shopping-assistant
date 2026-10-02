# AI Shopping Assistant

Tell it what you need and it does the shopping for you:

> "Buy me running shoes under €120, delivered by Friday."

It knows your tastes, searches several stores, picks the best option, asks for your OK, and pays — never more than you allow.

## How it works

One **LangGraph agent** made of four steps. Each step reads from a shared state and adds its own result.

```
request ─► user_preferences ─► product_search ─► approval ─► payment ─► result
                 │                   │               │           │
             Memory DB          Store agents        You     Payment services
```

| Step | What it does | Output | Planned tech |
|---|---|---|---|
| **1. User preferences** | Remembers what you bought and returned, and learns your sizes, colours, brands, budget, and what didn't work. Can explain why it thinks you like something. | `UserPreferences` | PostgreSQL, Qdrant, LLM agent with tools |
| **2. Product search** | Searches several stores at once for price, stock, and delivery date. Reads reviews and return policies, ranks the options, and explains its choice. | `Recommendation` | A2A, MCP, hybrid RAG |
| **3. Approval** | Double-checks the recommendation, shows you the product, store, price, and why — then **waits** until you approve or cancel. | `ApprovalDecision` | LangGraph `interrupt()` |
| **4. Payment** | Completes the checkout, only with your approval and never above your spending limit. | `PaymentResult` | AP2, ACP, Stripe test mode |

The data passed between steps is defined once, as Pydantic models, in [`schemas.py`](backend/agents/shopping/schemas.py). A real implementation of a step may change how it works, but must keep these shapes.

External services: a **memory database** (purchase history and preferences), **test stores** (Shopify development stores and simulated stores), and **Stripe test mode** — no real money is involved.

## Status

| | |
|---|---|
| ✅ | End-to-end graph with all four steps as stubs |
| ✅ | Real human-in-the-loop pause at approval |
| ✅ | Payment safety rules: no payment without approval, none above the limit |
| ✅ | End-to-end tests and CI (lint, types, tests) |
| 🚧 | User preferences as an LLM agent with tools ([#8](../../issues/8), [#10](../../issues/10), [#11](../../issues/11)) |
| 🚧 | PostgreSQL ([#9](../../issues/9)) |
| ⬜ | Product search, approval, and payment in depth |

**Build order:** basic version of all four steps → product search → approval → payment. User preferences is built in parallel.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs the right Python version for you.

```bash
git clone git@github.com:achrafElFaiq/multi-agent-shopping-assistant.git
cd multi-agent-shopping-assistant
uv sync                 # create .venv and install exact versions from uv.lock
cp .env.example .env    # then add your OpenRouter key to .env (never commit it)
```

The tests don't need a key: they use a fake model.

Run the checks — the same ones CI runs:

```bash
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy .           # type check (strict)
```

### Using the graph

```python
from langgraph.types import Command

from backend.agents.shopping.graph import build_graph
from backend.agents.shopping.schemas import ShoppingRequest

graph = build_graph()
config = {"configurable": {"thread_id": "order-1"}}

request = ShoppingRequest(user_id="user-1", query="running shoes", spending_limit="120")
result = graph.invoke({"request": request}, config)
print(result["__interrupt__"][0].value)  # the recommendation shown to the user

result = graph.invoke(Command(resume={"action": "approve"}), config)
print(result["payment"])  # status='paid' ...
```

## Project structure

```
backend/
  agents/shopping/
    schemas.py        # data passed between steps (Pydantic)
    state.py          # state shared by all nodes
    graph.py          # the four steps wired together
    nodes/            # one file per step
  core/ports/         # interfaces to external services (planned)
  core/adapters/      # their implementations (planned)
  mcp_servers/        # MCP servers (planned)
tests/
  e2e/                # full graph runs
```

## Contributing

### Dependencies

uv is the only tool for dependencies — no `pip install`.

```bash
uv add <package>          # app dependency
uv add --dev <package>    # dev tool
uv remove <package>
uv lock --upgrade         # upgrade, in its own commit
```

- Commit `pyproject.toml` and `uv.lock` **together**. Never edit `uv.lock` by hand.
- On a merge conflict in `uv.lock`, take either side, run `uv lock`, and commit the result.

### Issues

```
Title:  scope: what to do

What: one or two sentences.
Done when: how we know it's finished.
Blocked by: #N (if any)
```

### Commits

```
scope: what you did
```

Lowercase, imperative mood ("add", not "added"), one change per commit.

| Scope | Use for |
|---|---|
| `user_preferences` `product_search` `approval` `payment` | changes inside one step |
| `setup` | graph, state, schemas, project structure |
| `infra` | databases, Docker, deployment |
| `test` | tests only |
| `docs` | documentation |
| `chore` | tooling, CI, dependencies, cleanup |

Examples: `payment: block payments above limit`, `test: cover invalid approval answer`, `chore: install dependencies with uv in CI`.

### Pull requests

Work on a branch named after the scope (e.g. `user_preferences/agent-with-mock-tools`), open a PR with `Closes #N`, and merge once CI is green.

## Stack

Python 3.12 · LangGraph · Pydantic · FastAPI · PostgreSQL · Qdrant · Next.js · Docker · uv · pytest · ruff · mypy
