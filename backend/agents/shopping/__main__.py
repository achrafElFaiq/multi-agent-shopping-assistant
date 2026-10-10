"""Run the shopping agent in the terminal, with the real model, showing every step.

    uv run python -m backend.agents.shopping "running shoes" \
        --limit 120 --currency EUR --deliver-by 2026-10-20
    Add --raw to also print the raw HTTP calls to the model.

Needs OPENROUTER_API_KEY and LLM_MODEL in .env, and the search agent running in another
terminal: uv run python -m backend.agents.searcher
Preferences are mocks for now: the example ones of backend/seed.py, kept in memory.
"""

import argparse
import json
import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from backend.agents.shopping.graph import ShoppingGraph, build_graph
from backend.agents.shopping.schemas import ShoppingRequest
from backend.config.settings import load_settings
from backend.core.adapters.a2a_searcher import A2ASearcher
from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.adapters.openrouter import make_chat_model
from backend.seed import SEED_PREFERENCES

BLUE, PURPLE, YELLOW, GREEN, RED, DIM, BOLD, RESET = (
    "\033[34m", "\033[35m", "\033[33m", "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)  # fmt: skip


def short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"{DIM}… ({len(text)} chars){RESET}"


def pretty(body: bytes) -> str:
    try:
        return json.dumps(json.loads(body), indent=2, ensure_ascii=False)
    except ValueError:
        return body.decode(errors="replace")


def raw_http_client() -> httpx.Client:
    """An HTTP client that prints every request sent to the model and every response, in full.

    Headers are not printed: they contain the API key.
    """

    def show_request(request: httpx.Request) -> None:
        print(f"\n{RED}━━━━ HTTP REQUEST  {request.method} {request.url} ━━━━{RESET}")
        print(pretty(request.content))

    def show_response(response: httpx.Response) -> None:
        response.read()
        print(f"{RED}━━━━ HTTP RESPONSE  {response.status_code} ━━━━{RESET}")
        print(pretty(response.content))
        print(f"{RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")

    return httpx.Client(event_hooks={"request": [show_request], "response": [show_response]})


class Tracer(BaseCallbackHandler):
    """Prints every model call and tool call made inside the graph's nodes."""

    def __init__(self) -> None:
        self.model_calls = 0
        self.total_tokens = 0

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any) -> None:
        self.model_calls += 1
        conversation = messages[0]
        print(f"   {BLUE}→ model call #{self.model_calls}{RESET} {DIM}({len(conversation)} messages sent){RESET}")
        print(f"     {DIM}last message: {short(conversation[-1].content, 120)}{RESET}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        self.total_tokens += usage.get("total_tokens", 0)
        tokens = f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out tokens"
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(f"{c['name']}({short(c['args'], 80)})" for c in tool_calls)
            print(f"   {BLUE}← model asks for tools:{RESET} {names} {DIM}[{tokens}]{RESET}")
        else:
            content = getattr(message, "content", "") or "(empty reply: no more tools needed)"
            print(f"   {BLUE}← model answers:{RESET} {short(content, 200)} {DIM}[{tokens}]{RESET}")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        print(f"   {YELLOW}⚙ tool {serialized.get('name', '?')}{RESET}({short(kwargs.get('inputs') or input_str, 80)})")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        print(f"   {YELLOW}  returned:{RESET} {short(getattr(output, 'content', output))}")


def run_until_pause(graph: ShoppingGraph, graph_input: Any, config: RunnableConfig) -> None:
    """Streams the graph node by node, printing what each node adds to the state."""
    started = time.perf_counter()
    for update in graph.stream(graph_input, config, stream_mode="updates"):
        for node, output in update.items():
            elapsed = time.perf_counter() - started
            if node == "__interrupt__":
                print(f"{BOLD}{PURPLE}■ user input{RESET} {DIM}paused, waiting for you{RESET}")
                continue
            print(f"{BOLD}{PURPLE}■ {node}{RESET} {DIM}done in {elapsed:.1f}s{RESET}")
            for key, value in (output or {}).items():
                print(f"   {GREEN}+ state.{key}{RESET} = {short(value.model_dump(mode='json'), 600)}")
            print()
            started = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shopping agent in the terminal, showing every step.")
    parser.add_argument("query", help='what to buy, e.g. "running shoes"')
    parser.add_argument("--limit", required=True, type=Decimal, help="spending limit, e.g. 120")
    parser.add_argument("--currency", required=True, help="currency of the limit, e.g. EUR")
    parser.add_argument("--deliver-by", required=True, type=date.fromisoformat, help="deadline, e.g. 2026-10-20")
    parser.add_argument("--raw", action="store_true", help="also print the full HTTP requests and responses")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=f"   {DIM}%(message)s{RESET}")
    logging.getLogger("backend.core.adapters.memory_recommendations").setLevel(logging.INFO)  # the mock actions
    logging.getLogger("backend.core.adapters.a2a_searcher").setLevel(logging.INFO)  # the A2A messages

    settings = load_settings()
    http_client = raw_http_client() if args.raw else None
    # Mock preferences for now. For PostgreSQL: PostgresPreferencesRepository(settings.database_url.get_secret_value())
    repository = InMemoryPreferencesRepository(SEED_PREFERENCES)
    graph = build_graph(
        make_chat_model(settings, http_client),
        repository,
        A2ASearcher(settings.searcher_url),
        InMemoryRecommendations(),
    )
    tracer = Tracer()
    thread_id = str(uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "callbacks": [tracer]}
    request = ShoppingRequest(
        query=args.query,
        spending_limit=args.limit,
        currency=args.currency,
        deliver_by=args.deliver_by,
    )

    print(f"{DIM}model:   {settings.llm_model}")
    print(f"thread:  {thread_id}{RESET}")
    print(f"{BOLD}request: {request.query}{RESET}", end=" ")
    print(f"(limit {request.spending_limit} {request.currency}, by {request.deliver_by})\n")

    run_until_pause(graph, {"request": request}, config)

    while (state := graph.get_state(config)).interrupts:
        question = state.interrupts[0].value
        if question.get("type") == "recommendation":
            offer = question["offer"]
            options = ", ".join(f"{name} {value}" for name, value in offer["options"].items())
            print(f"   {BOLD}#{question['number']} {offer['title']}{RESET} at {offer['store']}")
            print(f"   {BOLD}{offer['price']} {offer['currency']}{RESET}" + (f"  ({options})" if options else ""))
            print(f"   {offer['url']}")
            print(f"   {GREEN}{question['pitch']}{RESET}\n")
            if question.get("error"):
                print(f"   {RED}{question['error']}{RESET}")
            answer = input(f"{BOLD}Do you want it? [yes/no] {RESET}").strip()
        else:  # a question from the preferences agent, or "why not?" after a no
            answer = input(f"{BOLD}{question['question']} {RESET}").strip()
        print()
        run_until_pause(graph, Command(resume=answer), config)

    payment = state.values["payment"]
    color = {"paid": GREEN, "cancelled": YELLOW, "blocked": RED}[payment.status]
    print(f"{BOLD}result: {color}{payment.status}{RESET}", end="   ")
    print(f"{DIM}model calls: {tracer.model_calls}, tokens: {tracer.total_tokens}{RESET}")


if __name__ == "__main__":
    main()
