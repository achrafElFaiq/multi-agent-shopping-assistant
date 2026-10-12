"""Run the master agent in the terminal, showing every tool it calls.

    uv run python -m backend.agents.master_agent "running shoes" --limit 120 --currency EUR

Needs OPENROUTER_API_KEY, LLM_MODEL and DATABASE_URL in .env, PostgreSQL with pgvector running, and the two
other agents running, each in its own terminal:
    uv run python -m backend.agents.product_finder_agent
    uv run python -m backend.agents.user_preferences_agent
"""

import argparse
import logging
from decimal import Decimal
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from backend.agents.master_agent.agent import MasterGraph, build_master, first_messages
from backend.agents.master_agent.tools import payment_of
from backend.config.settings import load_settings
from backend.core.adapters.a2a_preferences import A2APreferencesAgent
from backend.core.adapters.a2a_searcher import A2ASearcher
from backend.core.adapters.fastembed_embedder import FastEmbedEmbedder
from backend.core.adapters.openrouter import make_chat_model
from backend.core.adapters.postgres_recommendations import PostgresRecommendations
from backend.core.models.shopping import ShoppingRequest

BLUE, YELLOW, GREEN, RED, DIM, BOLD, RESET = (
    "\033[34m", "\033[33m", "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)  # fmt: skip


def short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + f"{DIM}… ({len(text)} chars){RESET}"


def run_until_pause(master: MasterGraph, master_input: Any, config: RunnableConfig) -> None:
    """Prints each tool the master asks for and what it returned, and any text the model answered instead."""
    for update in master.stream(master_input, config, stream_mode="updates"):
        for node, output in update.items():
            if node == "__interrupt__":
                continue
            for message in output["messages"]:
                if isinstance(message, AIMessage):
                    if not message.tool_calls:
                        print(f"{RED}  model answered with text, no tool:{RESET} {short(str(message.content))}")
                    for call in message.tool_calls:
                        print(f"{BLUE}→ {call['name']}{RESET}({short(str(call['args']), 120)})")
                elif isinstance(message, ToolMessage):
                    print(f"{YELLOW}  returned:{RESET} {short(str(message.content))}\n")


def ask(question: dict[str, Any]) -> str:
    """Shows what the master asks the user, and reads the answer."""
    if question.get("error"):
        print(f"   {RED}{question['error']}{RESET}")
    if question.get("type") != "recommendation":  # a question from the preferences agent, or "why not?"
        return input(f"{BOLD}{question['question']} {RESET}").strip()
    offer = question["offer"]
    options = ", ".join(f"{name} {value}" for name, value in offer["options"].items())
    print(f"   {BOLD}#{question['number']} {offer['title']}{RESET} at {offer['store']}")
    print(f"   {BOLD}{offer['price']} {offer['currency']}{RESET}" + (f"  ({options})" if options else ""))
    print(f"   {offer['url']}")
    print(f"   {GREEN}{question['pitch']}{RESET}\n")
    return input(f"{BOLD}Do you want it? [yes/no] {RESET}").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the master agent in the terminal.")
    parser.add_argument("query", help='what to buy, e.g. "running shoes"')
    parser.add_argument("--limit", required=True, type=Decimal, help="spending limit, e.g. 120")
    parser.add_argument("--currency", required=True, help="currency of the limit, e.g. EUR")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=f"   {DIM}%(message)s{RESET}")
    logging.getLogger("backend.core.adapters.a2a_searcher").setLevel(logging.INFO)  # the A2A messages
    logging.getLogger("backend.core.adapters.a2a_preferences").setLevel(logging.INFO)

    settings = load_settings()
    model = make_chat_model(settings)
    recommendations = PostgresRecommendations(settings.database_url.get_secret_value(), FastEmbedEmbedder())
    preferences_agent = A2APreferencesAgent(settings.preferences_url)
    request = ShoppingRequest(query=args.query, spending_limit=args.limit, currency=args.currency)
    master = build_master(model, preferences_agent, A2ASearcher(settings.searcher_url), recommendations, request)

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
    print(f"{DIM}model: {settings.llm_model}{RESET}")
    print(f"{BOLD}request: {request.query}{RESET} (limit {request.spending_limit} {request.currency})\n")

    run_until_pause(master, first_messages(request), config)
    while (state := master.get_state(config)).interrupts:
        answer = ask(state.interrupts[0].value)
        print()
        run_until_pause(master, Command(resume=answer), config)

    payment = payment_of(state.values["messages"])
    status = payment.status if payment else "cancelled"  # the master gave up: no payment
    color = {"paid": GREEN, "cancelled": YELLOW, "blocked": RED}[status]
    print(f"{BOLD}result: {color}{status}{RESET}")


if __name__ == "__main__":
    main()
